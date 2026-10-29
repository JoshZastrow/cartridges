"""The Teacher: an LLM on vLLM (Modal) with a long context block in its prompt.

Copied from `cartridges.cartridge.teacher` at commit a9a7588. Changed: the
journal-specific assumptions are parameters. `instructions` (the cartridge
module's citation-heavy INSTRUCTIONS) defaults to empty; the context file is a
path on the `shared-teacher-cache` Volume chosen per Teacher (`context_path`,
"" for none) instead of a journal file baked into the image; the Qwen chat
template's system turn and stop token are the defaults of `system_turn` and
`stop_token` rather than hard-coded. The HF `TeacherScorer` (parked, never
validated on GPU) and the cartridge smoke entrypoints are not copied.

The teacher prompt is a context block (its own system turn) followed by the
conversation, rendered by the chat template exactly as without context:

    <|im_start|>system\\n{instructions}\\n\\n{context}<|im_end|>\\n   <- prefix_ids
    [<|im_start|>system\\n...<|im_end|>\\n]  (the caller's system turn, if any)
    <|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...   <- prompt_ids

`render()` builds both, so every caller tokenizes the conversation the same
way. Thinking is off by default (chat_template_kwargs enable_thinking=False).

`Teacher` generates with vLLM's AsyncLLM. Automatic prefix caching keeps the
context block's KV after the first request, so later requests only prefill
their own suffix. With `logprobs=k` each generated token also carries the
teacher's raw top-k logprobs. Results are `Completion`s.

Put a context file on the Volume, deploy once, then call from anywhere:

    uv run modal volume put shared-teacher-cache journal.md /context/journal.md
    uv run modal deploy -m shared.teacher
    teacher = teacher_cls("Qwen/Qwen3.5-4B", context_path="context/journal.md", instructions="...")
    outs = teacher.generate.remote(messages, max_tokens=256, n=2, logprobs=20)
"""

# No `from __future__ import annotations`: Modal reads the modal.parameter
# annotations at class creation and needs real types.
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import modal
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MINUTES = 60
APP_NAME = "shared-teacher"
SMOKE_MODEL_ID = "Qwen/Qwen3.5-4B"
VOLUME_NAME = "shared-teacher-cache"  # HF weights under hf/, context files anywhere else
VOLUME_PATH = "/cache"
COMPILE_CACHE_PATH = "/root/.cache/vllm"  # vLLM torch.compile + Triton caches, kept across cold starts

QWEN_SYSTEM_TURN = "<|im_start|>system\n{content}<|im_end|>\n"
QWEN_STOP_TOKEN = "<|im_end|>"
NO_THINKING = {"enable_thinking": False}
# Qwen3 model card, instruct (non-thinking) mode.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
MAX_TOP_K = 20  # vLLM's default max_logprobs
# Modal gpu= per model; models not listed need an explicit gpu.
GPU_FOR_MODEL = {SMOKE_MODEL_ID: "L4", "Qwen/Qwen3.8-27B": "H200"}


class Completion(BaseModel):
    """One generated answer. prompt_ids is the conversation without the context
    block; topk_* are empty unless logprobs > 0."""

    text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    topk_ids: list[list[int]] = []
    topk_logprobs: list[list[float]] = []
    finish_reason: str | None = None
    ttft_s: float = 0.0  # request in -> first token out, on the container
    latency_s: float = 0.0


# ---------------------------------------------------------------- prompts


def context_block(context: str, instructions: str = "", system_turn: str = QWEN_SYSTEM_TURN) -> str:
    """The instructions and context as one system turn of the chat template."""
    content = f"{instructions}\n\n{context}".strip()
    return system_turn.format(content=content)


def render(
    tokenizer, messages: list[dict], context: str | None = None, instructions: str = "",
    system_turn: str = QWEN_SYSTEM_TURN, chat_template_kwargs: dict = NO_THINKING,
) -> tuple[list[int], list[int]]:
    """(prefix_ids, prompt_ids) for a conversation that ends with a user turn
    (it may start with its own system turn).

    prefix_ids is the context block ([] when context is None); prompt_ids is the
    chat-templated conversation plus the assistant header. They are tokenized
    separately, so prompt_ids does not depend on the context and
    prefix_ids + prompt_ids is the teacher's prompt."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    if context is None:
        return [], prompt_ids
    return tokenizer.encode(context_block(context, instructions, system_turn), add_special_tokens=False), prompt_ids


def completion_ids(tokenizer, completion: str | list[int], stop_token: str = QWEN_STOP_TOKEN) -> list[int]:
    """Token ids of an assistant answer, ending with `stop_token`, so a student
    also learns where to stop. Token ids pass through."""
    if not isinstance(completion, str):
        return list(completion)
    return tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(stop_token)]


def sampling_kwargs(
    n: int, max_tokens: int, logprobs: int, temperature: float | None, seed: int | None,
    json_schema: dict | None, structured_outputs_cls,
) -> dict:
    """vLLM SamplingParams kwargs for Teacher.generate: SAMPLING (temperature
    overridable); temperature 0.0 is plain greedy; json_schema becomes
    structured_outputs=StructuredOutputsParams(json=schema)."""
    if temperature == 0.0:
        sampling = {"temperature": 0.0}
    else:
        sampling = {**SAMPLING, **({"temperature": temperature} if temperature is not None else {})}
    kwargs = {"n": n, "max_tokens": max_tokens, "logprobs": logprobs or None, "seed": seed, **sampling}
    if json_schema is not None:
        kwargs["structured_outputs"] = structured_outputs_cls(json=json_schema)
    return kwargs


def topk_from_vllm(position_logprobs, k: int) -> tuple[list[list[int]], list[list[float]]]:
    """vLLM sample logprobs (one {token_id: Logprob} per generated token, which
    also holds the sampled token when it is outside the top k) -> top-k ids and
    logprobs, most likely first."""
    ids, lps = [], []
    for position in position_logprobs:
        best = sorted(position.items(), key=lambda item: -item[1].logprob)[:k]
        ids.append([token_id for token_id, _ in best])
        lps.append([lp.logprob for _, lp in best])
    return ids, lps


# ---------------------------------------------------------------- Modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.29.0",
        "transformers==5.17.0",  # vllm 0.29 needs >=5.10.4
        "flash-linear-attention==0.5.2",
        "hf_transfer",
        "wandb",
    )
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "TRITON_CACHE_DIR": f"{COMPILE_CACHE_PATH}/triton"})
    .add_local_python_source("shared")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
compile_cache = modal.Volume.from_name("shared-teacher-compile-cache", create_if_missing=True)


def _secrets() -> list:
    from shared import tracking

    return [modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("HF_TOKEN"),
            modal.Secret.from_dict(tracking.modal_env())]


secrets = _secrets()


@app.cls(
    gpu=["H200", "H100"],
    max_containers=1,  # queued calls would otherwise start a second GPU while the first is in @enter
    scaledown_window=5 * MINUTES,
    volumes={VOLUME_PATH: volume, COMPILE_CACHE_PATH: compile_cache},
    secrets=secrets,
    timeout=30 * MINUTES,  # per call
    startup_timeout=40 * MINUTES,  # first start downloads the weights and prefills the context
    retries=0,  # a failed call fails; callers bound waits with .get(timeout=...)
)
@modal.concurrent(max_inputs=64)
class Teacher:
    """Generation with the context block in the prompt (or without it:
    with_context=False is the base model). Calls run concurrently; vLLM
    batches them and reuses the cached context."""

    model_id: str = modal.parameter(default=SMOKE_MODEL_ID)
    revision: str = modal.parameter(default="main")
    context_path: str = modal.parameter(default="")  # on the Volume; "" = no context block
    instructions: str = modal.parameter(default="")
    max_num_seqs: int = modal.parameter(default=64)
    max_model_len: int = modal.parameter(default=262144)

    @modal.enter()
    async def start(self):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
        from vllm.v1.engine.async_llm import AsyncLLM

        self.context = (Path(VOLUME_PATH) / self.context_path).read_text() if self.context_path else None
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=self.model_id,
                revision=self.revision,
                dtype="bfloat16",
                max_model_len=self.max_model_len,
                enable_prefix_caching=True,
                language_model_only=True,
                max_num_seqs=self.max_num_seqs,
                max_logprobs=MAX_TOP_K,
                gpu_memory_utilization=0.92,
            )
        )
        self._stats_task = asyncio.create_task(self._log_stats_forever())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.prefix_ids, _ = render(self.tokenizer, [{"role": "user", "content": "."}], self.context, self.instructions)
        volume.commit()  # keep the downloaded weights
        self.TokensPrompt, self.SamplingParams = TokensPrompt, SamplingParams
        self.prefill_s = 0.0
        if self.prefix_ids:
            # Prefill up to the last cache-block boundary inside the context block,
            # so later requests hit the cache there.
            cache_config = self.engine.vllm_config.cache_config
            block = cache_config.mamba_block_size or cache_config.block_size
            t0 = time.perf_counter()
            await self._run(self.prefix_ids[: len(self.prefix_ids) // block * block], SamplingParams(max_tokens=1))
            self.prefill_s = time.perf_counter() - t0
        compile_cache.commit()
        logger.info("prefilled %d context tokens in %.1fs", len(self.prefix_ids), self.prefill_s)

    async def _log_stats_forever(self, every_s: float = 10.0):
        while True:
            await asyncio.sleep(every_s)
            await self.engine.do_log_stats()

    async def _run(self, prompt_ids: list[int], params):
        """(final RequestOutput, seconds to the first output token)."""
        final, ttft, t0 = None, None, time.perf_counter()
        async for out in self.engine.generate(self.TokensPrompt(prompt_token_ids=prompt_ids), params, uuid.uuid4().hex):
            ttft = ttft if ttft is not None else time.perf_counter() - t0
            final = out
        return final, ttft

    @modal.method()
    def info(self) -> dict:
        """Context size, cold prefill time and this container's id (for stop(container_ids=...))."""
        return {"container_id": os.environ.get("MODAL_TASK_ID"), "model_id": self.model_id,
                "prefix_tokens": len(self.prefix_ids), "prefill_s": self.prefill_s}

    @modal.method()
    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        n: int = 1,
        logprobs: int = 0,
        with_context: bool = True,
        temperature: float | None = None,
        seed: int | None = None,
        json_schema: dict | None = None,
    ) -> list[Completion]:
        """n sampled answers to the conversation (which ends with a user turn).
        Sampling follows SAMPLING; temperature=0.0 is plain greedy decoding.
        logprobs=k (<= 20) also returns the teacher's raw top-k logprobs at every
        generated token. json_schema constrains the answer to JSON matching it."""
        from vllm.sampling_params import StructuredOutputsParams

        _, prompt_ids = render(self.tokenizer, messages)
        full = (self.prefix_ids if with_context else []) + prompt_ids
        params = self.SamplingParams(**sampling_kwargs(n, max_tokens, logprobs, temperature, seed, json_schema,
                                                       StructuredOutputsParams))
        t0 = time.perf_counter()
        out, ttft = await self._run(full, params)
        latency = time.perf_counter() - t0
        completions = []
        for choice in out.outputs:
            topk_ids, topk_lps = topk_from_vllm(choice.logprobs, logprobs) if logprobs else ([], [])
            completions.append(Completion(
                text=choice.text, prompt_ids=prompt_ids, completion_ids=list(choice.token_ids),
                topk_ids=topk_ids, topk_logprobs=topk_lps, finish_reason=choice.finish_reason,
                ttft_s=ttft, latency_s=latency,
            ))
        return completions


def teacher_cls(model_id: str = SMOKE_MODEL_ID, gpu: str | None = None, max_inputs: int | None = None, **params):
    """A handle on the deployed Teacher for `model_id` (gpu from GPU_FOR_MODEL
    unless given). Every distinct option or parameter makes a separate
    container pool, so callers sharing a Teacher should pass the same ones.
    max_inputs sets both Modal's input concurrency and vLLM's max_num_seqs."""
    cls = modal.Cls.from_name(APP_NAME, "Teacher").with_options(gpu=gpu or GPU_FOR_MODEL[model_id])
    if max_inputs is not None:
        cls = cls.with_concurrency(max_inputs=max_inputs)
        params["max_num_seqs"] = max_inputs
    return cls(model_id=model_id, **params)


def running_containers(rows: list[dict], app_name: str = APP_NAME) -> list[str]:
    """Container ids of `app_name` in `modal container list --json` rows. Raises
    on any other row shape, so a CLI format change can't make stop() silently
    match nothing."""
    for row in rows:
        if not {"container_id", "app_name"} <= row.keys():
            raise RuntimeError(f"unexpected `modal container list --json` row: {row}")
    return [r["container_id"] for r in rows if r["app_name"] == app_name]


def _list_containers(run, app_name: str) -> list[str]:
    import json

    listed = run(["modal", "container", "list", "--json"], capture_output=True, text=True, check=True)
    return running_containers(json.loads(listed.stdout or "[]"), app_name)


def stop(
    app_name: str = APP_NAME, run=None, recheck_s: float = 15.0, sleep=time.sleep,
    container_ids: list[str] | None = None,
) -> list[str]:
    """Stop this app's running containers now instead of paying for the idle
    scaledown window; returns the stopped ids. `run` defaults to
    subprocess.run. Cancel outstanding calls first (generate_all does on any
    interrupt): Modal reschedules an unfinished input of a stopped container
    onto a new one, which stop() reports as an ERROR after `recheck_s`.
    container_ids: stop only these, so other sessions' containers survive."""
    import subprocess

    run = run or subprocess.run
    ids = _list_containers(run, app_name)
    if container_ids is not None:
        ids = [c for c in ids if c in set(container_ids)]
    for container_id in ids:
        # --yes: without a TTY, the confirmation prompt aborts the stop.
        run(["modal", "container", "stop", "--yes", container_id], capture_output=True, text=True, check=True)
    if ids and recheck_s > 0:
        sleep(recheck_s)
        new = [c for c in _list_containers(run, app_name) if c not in ids]
        if new:
            logger.error("stop(): new %s container(s) %s after stopping %s: queued inputs were rescheduled. "
                         "Cancel them, or run `modal app stop --yes %s` and redeploy.", app_name, new, ids, app_name)
    return ids


def generate_all(
    teacher, conversations: list[list[dict]], window: int = 128, **kwargs
) -> list[list[Completion] | None]:
    """Teacher.generate for every conversation, results in order; a failed call
    gives None. Up to `window` requests run at once, each in its own worker
    thread (spawn, then a blocking get). On any interrupt every unfinished call
    is cancelled before the exception propagates, so nothing is left queued to
    restart a container after stop()."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    calls: dict[int, object] = {}
    cancelled = threading.Event()

    def run(i: int, conversation: list[dict]):
        if cancelled.is_set():
            return None
        call = teacher.generate.spawn(conversation, **kwargs)
        calls[i] = call
        while True:
            if cancelled.is_set():
                return None
            try:
                return call.get(timeout=30)
            except TimeoutError:
                continue

    pool = ThreadPoolExecutor(max_workers=max(1, min(window, len(conversations))))
    futures = [pool.submit(run, i, c) for i, c in enumerate(conversations)]
    try:
        results = []
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception:
                logger.exception("generate_all: call %s failed", getattr(calls.get(i), "object_id", "?"))
                results.append(None)
        return results
    except BaseException:
        cancelled.set()
        n = 0
        for call in list(calls.values()):
            try:
                call.cancel()
                n += 1
            except Exception:
                logger.exception("generate_all: could not cancel call %s", getattr(call, "object_id", "?"))
        logger.warning("generate_all: interrupted; cancelled %d of %d spawned calls", n, len(calls))
        raise
    finally:
        cancelled.set()
        pool.shutdown(wait=False, cancel_futures=True)


def modal_generate(teacher):
    """`generate(batch, **kw) -> list[Completion | None]` over a Teacher handle:
    the first completion of each conversation (shared.judge's `generate`).
    Copied from `cartridges.cartridge.data.modal_generate`."""

    def generate(batch: list[list[dict]], **kw) -> list:
        return [out[0] if out else None for out in generate_all(teacher, batch, **kw)]

    return generate
