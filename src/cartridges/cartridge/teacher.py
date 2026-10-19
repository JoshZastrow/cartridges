"""The teacher: the base model with the journal context in its prompt.

The teacher prompt is a context block (its own system turn) followed by the
conversation, rendered by the chat template exactly as without context:

    <|im_start|>system\n{INSTRUCTIONS}\n\n{context}<|im_end|>\n   <- prefix_ids (~205k tokens)
    [<|im_start|>system\n...<|im_end|>\n]  (the caller's system turn, if any)
    <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n   <- prompt_ids

The student (base model + cartridge) sees exactly `prompt_ids`, with the
cartridge in place of the context block, and the base model alone ("base"
mode, eval's baseline and judge) sees `prompt_ids` with nothing before it.
`render()` builds both, so every stage tokenizes the conversation the same way.
The context block only says what the journals are; answer format and
citation rules belong to the conversation (e.g. evals.ANSWER_SYSTEM_PROMPT), so
all three systems get the same instructions.

Thinking is off everywhere (enable_thinking=False): Qwen3.8 thinks by default,
and distillation targets and eval answers should be the direct answer.

Two Modal classes share one image:

- `Teacher` (vLLM, AsyncLLM) generates. Automatic prefix caching (hybrid
  mamba_cache_mode "align") keeps the context block's KV/state after the
  first request, so later requests only prefill their own suffix. With
  `logprobs=k` each generated token also carries the teacher's raw top-k
  logprobs: this is the main source of distillation targets.
- `TeacherScorer` (HF transformers) scores a given completion:
  `logprobs(messages, completion, top_k)`. vLLM cannot do this cheaply, since
  a request with prompt_logprobs skips the prefix cache and recomputes all
  ~205k context tokens. The scorer prefills the context once into a hybrid
  cache and scores each completion on a fork of it (attention KV shared,
  linear-attention states cloned).

Both return `prompt_ids`, `completion_ids`, `topk_ids`, `topk_logprobs` with
the names of `train_distill.Example`.

Deploy once, then call from anywhere:

    uv run modal deploy -m cartridges.cartridge.teacher  # module mode: results unpickle as cartridges.cartridge.teacher.*
    teacher = teacher_cls("Teacher", "Qwen/Qwen3.5-4B")
    outs = teacher.generate.remote(messages, max_tokens=256, n=2, logprobs=20)

Smoke test of Teacher (Qwen3.5-4B, one L4):

    uv run modal run -m cartridges.cartridge.teacher::smoke
"""

# No `from __future__ import annotations`: Modal reads the modal.parameter
# annotations at class creation and needs real types.
import asyncio
import copy
import logging
import os
import time
import uuid
from pathlib import Path

import modal
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MINUTES = 60
APP_NAME = "cartridges-teacher"
MODEL_ID = "Qwen/Qwen3.8-27B"
SMOKE_MODEL_ID = "Qwen/Qwen3.5-4B"
# Same journals as timeline.md, with a "## [<journal>:p.<N>] date: ..." heading per
# page: the anchors citations refer to (datagen.qa.Passage.text uses the same format).
CONTEXT_DIR = "data/journals/output/2026_10_13"
CONTEXT_FILE = "teacher_context.md"
REMOTE_CONTEXT_DIR = "/context"
VOLUME_PATH = "/cache"
# vLLM's torch.compile/AOT cache, plus Triton's kernel cache (TRITON_CACHE_DIR):
# without them every cold start recompiles (~31 s) and re-JITs the Qwen GDN
# kernels in vLLM's Triton warmup (~2 min on L4).
COMPILE_CACHE_PATH = "/root/.cache/vllm"

INSTRUCTIONS = (
    "Below are the user's own handwritten journals, transcribed by OCR, in timeline order; "
    "the user is their author. "
    'Each page starts with a heading like "## [Morning Pages:p.63] date: 2024-11-28". '
    'Answer the user\'s questions about their life from the journals, speaking to them as "you". '
    "Cite the page behind each claim about the user's life, even in brief answers, exactly as its "
    "heading names it, e.g. [Morning Pages:p.63] or [Auto biography:p.12]; never a page number alone. "
    "For questionnaire items, cite the entries behind your rating. "
    "If the journals don't show it, say so and give your best estimate as uncertain. "
    "Never present the question's wording as a journal quote. "
    "Keep answers focused, typically 300-500 words; go longer only when the question calls for "
    "a timeline, a comparison across periods, or extensive context. No filler or repetition."
)
# Qwen3.8 model card, instruct (non-thinking) mode.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
MAX_TOP_K = 20  # vLLM's default max_logprobs
# Default GPU per model (Modal gpu= strings, as in tracking.GPU_RATES_USD_PER_HOUR).
GPU_FOR_MODEL = {SMOKE_MODEL_ID: "L4", MODEL_ID: "H200"}  # H200: KV headroom for many 205k-context streams
# Teacher's container settings are the 27B session's (2026-10-15 post-mortem):
# - gpu: H200, falling back to H100 when no H200 is free (an H100 fits the 27B
#   model + 13.6 GB context cache, with fewer concurrent streams). Only the
#   decorator takes a fallback list; with_options takes one GPU string.
# - max_containers=1: while the first container is still in @enter, queued
#   calls would otherwise make Modal start a second GPU that serves nothing.
# - scaledown_window=5 min: stages run back-to-back from one runbook, which
#   ends with stop(); a long idle window only bills an idle GPU.
# 4B callers override the GPU through teacher_options().
TEACHER_OPTIONS = {"gpu": ["H200", "H100"], "max_containers": 1, "scaledown_window": 5 * MINUTES}


class Completion(BaseModel):
    """One generated answer. prompt_ids is the conversation without the context
    block (what the student sees); topk_* are empty unless logprobs > 0."""

    text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    topk_ids: list[list[int]] = []
    topk_logprobs: list[list[float]] = []
    finish_reason: str | None = None
    ttft_s: float = 0.0  # request in -> first token out, on the container
    latency_s: float = 0.0


class TopK(BaseModel):
    """The teacher's top-k next-token logprobs at each completion position:
    position t is the distribution that predicts completion_ids[t]."""

    prompt_ids: list[int]
    completion_ids: list[int]
    topk_ids: list[list[int]]
    topk_logprobs: list[list[float]]
    token_logprobs: list[float] = []  # logprob of completion_ids[t] itself (teacher-forced)


# ---------------------------------------------------------------- prompts


def load_context(name: str = CONTEXT_FILE, root: str | Path = CONTEXT_DIR) -> str:
    return (Path(root) / name).read_text()


def context_block(context: str) -> str:
    """The context as a system turn, as the Qwen chat template renders one."""
    content = f"{INSTRUCTIONS}\n\n{context}".strip()
    return f"<|im_start|>system\n{content}<|im_end|>\n"


def render(tokenizer, messages: list[dict], context: str | None = None) -> tuple[list[int], list[int]]:
    """(prefix_ids, prompt_ids) for a conversation that ends with a user turn
    (it may start with its own system turn).

    prefix_ids is the context block ([] when context is None); prompt_ids is the
    chat-templated conversation plus the non-thinking assistant header. They
    are tokenized separately, so prompt_ids does not depend on the context and
    prefix_ids + prompt_ids is the teacher's prompt."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    if context is None:
        return [], prompt_ids
    return tokenizer.encode(context_block(context), add_special_tokens=False), prompt_ids


def completion_ids(tokenizer, completion: str | list[int]) -> list[int]:
    """Token ids of an assistant answer, ending with <|im_end|> (the model's stop
    token), so the student also learns where to stop. Token ids pass through."""
    if not isinstance(completion, str):
        return list(completion)
    return tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]


def cache_bytes(text_config, n_tokens: int, kv_bytes: int = 2, state_bytes: int = 4) -> int:
    """Memory of the hybrid cache for an n-token context: attention KV grows with
    n, the linear-attention (Gated DeltaNet) recurrent and conv states do not.
    bf16 KV and fp32 recurrent state (mamba_ssm_dtype) by default; conv states
    are counted at the conv kernel width, as HF keeps them."""
    c = text_config
    n_full = c.layer_types.count("full_attention")
    n_linear = c.layer_types.count("linear_attention")
    kv = n_full * 2 * c.num_key_value_heads * c.head_dim * n_tokens * kv_bytes
    recurrent = c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * state_bytes
    conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
    conv = conv_dim * c.linear_conv_kernel_dim * kv_bytes
    return kv + n_linear * (recurrent + conv)


def sampling_kwargs(
    n: int, max_tokens: int, logprobs: int, temperature: float | None, seed: int | None,
    json_schema: dict | None, structured_outputs_cls,
) -> dict:
    """vLLM SamplingParams kwargs for Teacher.generate: SAMPLING (temperature
    overridable); temperature 0.0 is plain greedy; json_schema becomes
    structured_outputs=StructuredOutputsParams(json=schema) (vLLM 0.29)."""
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


# ---------------------------------------------------------------- HF scorer


def fork_cache(cache):
    """A copy of a prefilled cache that a forward pass can extend without
    changing the original. Attention keys/values are shared (DynamicLayer.update
    concatenates into new tensors); linear-attention conv/recurrent states, which
    are updated in place, are cloned (~150 MB for 27B)."""
    memo = {}
    for layer in cache.layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                memo[id(tensor)] = tensor
    return copy.deepcopy(cache, memo)


class Scorer:
    """Teacher logprobs with HF transformers: the context block is prefilled once
    (in chunks) and each completion is scored on a fork of that cache."""

    def __init__(self, model, tokenizer=None, prefill_chunk: int = 4096):
        self.model, self.tokenizer, self.prefill_chunk = model, tokenizer, prefill_chunk
        self.prefix_ids: list[int] = []
        self.cache = None

    def prefill(self, prefix_ids: list[int]) -> float:
        """Run the context block through the model once; returns seconds."""
        import torch
        from transformers import DynamicCache

        device = self.model.device
        t0 = time.perf_counter()
        cache = DynamicCache(config=self.model.config)
        with torch.inference_mode():
            for start in range(0, len(prefix_ids), self.prefill_chunk):
                chunk = torch.tensor([prefix_ids[start : start + self.prefill_chunk]], device=device)
                self.model(input_ids=chunk, past_key_values=cache, use_cache=True, logits_to_keep=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.prefix_ids, self.cache = list(prefix_ids), cache
        return time.perf_counter() - t0

    def cache_bytes(self) -> int:
        """Bytes held by the prefilled cache (attention KV + linear states)."""
        total = 0
        for layer in self.cache.layers if self.cache is not None else []:
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
            for name in ("conv_states", "recurrent_states"):
                for tensor in (getattr(layer, name, None) or {}).values():
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
        return total

    def score(
        self, prompt_ids: list[int], completion_ids: list[int], top_k: int = MAX_TOP_K, with_context: bool = True,
        chunk: int = 512,
    ) -> tuple[list[list[int]], list[list[float]], list[float]]:
        """(top-k ids, top-k logprobs, chosen-token logprobs) of the distributions
        that predict each completion token, given prefix (if with_context) +
        prompt + the completion so far. One forward; the vocabulary-wide
        log-softmax runs `chunk` positions at a time to bound memory."""
        import torch

        if with_context and self.cache is None:
            raise RuntimeError("prefill() the context before scoring with it")
        T = len(completion_ids)
        ids = torch.tensor([prompt_ids + completion_ids], device=self.model.device)
        targets = torch.tensor(completion_ids, device=self.model.device)
        cache = fork_cache(self.cache) if with_context else None
        top_ids, top_lps, chosen = [], [], []
        with torch.inference_mode():
            logits = self.model(input_ids=ids, past_key_values=cache, use_cache=with_context, logits_to_keep=T + 1).logits
            for start in range(0, T, chunk):
                logprobs = torch.log_softmax(logits[0, start:min(start + chunk, T)].float(), dim=-1)
                top = logprobs.topk(top_k, dim=-1)
                top_ids += top.indices.tolist()
                top_lps += top.values.tolist()
                chosen += logprobs.gather(-1, targets[start:start + chunk, None])[:, 0].tolist()
        return top_ids, top_lps, chosen

    def score_ids(
        self, prompt_ids: list[int], completion_ids: list[int], top_k: int = MAX_TOP_K, with_context: bool = True
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Top-k logprobs of the distributions that predict each completion token,
        given prefix (if with_context) + prompt + the completion so far."""
        top_ids, top_lps, _ = self.score(prompt_ids, completion_ids, top_k, with_context)
        return top_ids, top_lps

    def logprobs(
        self, messages: list[dict], completion: str | list[int], top_k: int = MAX_TOP_K, with_context: bool = True
    ) -> TopK:
        _, prompt_ids = render(self.tokenizer, messages)
        completion = completion_ids(self.tokenizer, completion)
        topk_ids, topk_logprobs, chosen = self.score(prompt_ids, completion, top_k, with_context)
        return TopK(prompt_ids=prompt_ids, completion_ids=completion, topk_ids=topk_ids, topk_logprobs=topk_logprobs,
                    token_logprobs=chosen)


# The name the scorer registers flash_attention under. It must not contain "flash":
# transformers treats any such name as a request for a flash-attn kernel and
# tries to import one (2026-10-16: "Could not find ... implementation at
# `teacher_flash`" crash-looped the first 27B TeacherScorer start).
SCORER_ATTENTION = "teacher_varlen"


def flash_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers attention for the scorer (batch 1, no padding): vLLM's bundled
    flash-attn, causal aligned to the end of the keys, GQA native. SDPA cannot
    serve this model here: with head_dim 256 and a mask over a ~205k-token
    cache it falls back to the math kernel, which materializes
    queries x heads x keys scores and repeats the KV for GQA (OOM). Since this
    name is not a transformers mask implementation, no mask is built."""
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported

    q, k, v = (t[0].transpose(0, 1).contiguous() for t in (query, key, value))  # [tokens, heads, dim]
    cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=q.device)
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=q.shape[0], cu_seqlens_q=cu_q, max_seqlen_k=k.shape[0], cu_seqlens_k=cu_k,
        softmax_scale=scaling, causal=True, fa_version=3 if is_fa_version_supported(3) else 2,
    )
    return out.unsqueeze(0), None  # [1, tokens, heads, dim]


# ---------------------------------------------------------------- Modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.29.0",
        "transformers==5.17.0",  # vllm 0.29 needs >=5.10.4; vllm 0.31 caps it <5.18
        "flash-linear-attention==0.5.2",  # HF Gated DeltaNet kernels; torch fallback is >10x slower
        "hf_transfer",
        "wandb",
    )
    .env({
        "HF_HOME": f"{VOLUME_PATH}/hf",
        "TRITON_CACHE_DIR": f"{COMPILE_CACHE_PATH}/triton",
        # vLLM 0.29's default V2 model runner never takes cascade steps (it builds
        # attention metadata with common_prefix_len=0), so disable_cascade_attn=False
        # only works on the V1 runner. Set to "1" to roll back to V2.
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    })
    .add_local_file(f"{CONTEXT_DIR}/{CONTEXT_FILE}", f"{REMOTE_CONTEXT_DIR}/{CONTEXT_FILE}")
    .add_local_python_source("cartridges")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("cartridges-cartridge-cache", create_if_missing=True)
compile_cache = modal.Volume.from_name("cartridges-teacher-compile-cache", create_if_missing=True)
def _secrets() -> list:
    from cartridges.cartridge import tracking

    # tracking.modal_env() carries the local git SHA/dirty flag into the container.
    return [modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("HF_TOKEN"),
            modal.Secret.from_dict(tracking.modal_env())]


secrets = _secrets()


SERVICE_GROUP = "service"


def _start_run(role: str, model_id: str, group: str, config: dict):
    """One W&B run per container: the Teacher's own health, prefill and GPU
    cost. In the default "service" group it goes to exp-<date>-teacher-service;
    an experiment's share of that cost comes from monitoring's time-window join."""
    from cartridges.cartridge import tracking

    if group in ("", SERVICE_GROUP):
        group = tracking.group_name("teacher-service")
    return tracking.init_run("teacher", model_id, group, config=config, role=role)


@app.cls(
    **TEACHER_OPTIONS,
    volumes={VOLUME_PATH: volume, COMPILE_CACHE_PATH: compile_cache},
    secrets=secrets,
    timeout=30 * MINUTES,  # per call
    startup_timeout=40 * MINUTES,  # first start downloads the weights and prefills the context
    retries=0,  # a failed call fails; callers bound waits with .get(timeout=...)
)
@modal.concurrent(max_inputs=64)
class Teacher:
    """Generation with the context block in the prompt (or without it:
    with_context=False is the base model, eval's baseline). Calls run
    concurrently; vLLM batches them and reuses the cached context."""

    model_id: str = modal.parameter(default=SMOKE_MODEL_ID)
    revision: str = modal.parameter(default="main")
    context_file: str = modal.parameter(default=CONTEXT_FILE)
    max_num_seqs: int = modal.parameter(default=64)
    group: str = modal.parameter(default=SERVICE_GROUP)  # pool identity: leave the default

    @modal.enter()
    async def start(self):
        from transformers import AutoConfig, AutoTokenizer
        from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
        from vllm.v1.engine.async_llm import AsyncLLM

        self.run = _start_run(
            "gen", self.model_id, self.group,
            {"model_id": self.model_id, "revision": self.revision, "context_file": self.context_file,
             "max_num_seqs": self.max_num_seqs, "engine": "vllm"},
        )
        self.context = load_context(self.context_file, REMOTE_CONTEXT_DIR)
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=self.model_id,
                revision=self.revision,
                dtype="bfloat16",
                max_model_len=262144,
                enable_prefix_caching=True,  # hybrid: mamba_cache_mode "align" by default
                language_model_only=True,  # no vision encoder, more room for KV
                max_num_seqs=self.max_num_seqs,
                max_logprobs=MAX_TOP_K,
                gpu_memory_utilization=0.92,
                # Opt in to cascade (shared-prefix) attention: with >= 8 streams
                # that all share the journal prefix, its KV is read once per
                # step instead of once per stream. Off by default in vLLM 0.29.
                # Steps that use it lose full CUDA graphs, and logprobs can
                # differ by ~1e-3 (LSE-merge rounding).
                disable_cascade_attn=False,
            )
        )
        config = self.engine.vllm_config
        print(f"teacher: cascade attention enabled = {not config.model_config.disable_cascade_attn}; "
              f"model runner = {'V2' if config.use_v2_model_runner else 'V1'}")
        # AsyncLLM only logs its periodic stats (running/waiting requests, prompt
        # and generation tok/s, prefix-cache hit rate) when someone calls
        # do_log_stats(); vLLM's API server does it every 10 s, so do the same.
        self._stats_task = asyncio.create_task(self._log_stats_forever())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.text_config = AutoConfig.from_pretrained(self.model_id, revision=self.revision).get_text_config()
        self.prefix_ids, _ = render(self.tokenizer, [{"role": "user", "content": "."}], self.context)
        volume.commit()  # keep the downloaded weights

        # Prefill the context once. In "align" mode vLLM keeps the linear-attention
        # state only at block boundaries, so prefill up to the last boundary
        # inside the context block: later requests then hit the cache there and
        # recompute less than one block plus their own tokens.
        cache_config = self.engine.vllm_config.cache_config
        block = cache_config.mamba_block_size or cache_config.block_size
        aligned = len(self.prefix_ids) // block * block
        self.TokensPrompt, self.SamplingParams = TokensPrompt, SamplingParams
        t0 = time.perf_counter()
        await self._run(self.prefix_ids[:aligned], SamplingParams(max_tokens=1))
        self.prefill_s = time.perf_counter() - t0
        compile_cache.commit()  # compile/Triton caches for the next cold start
        self.run.log({"teacher/prefix_tokens": len(self.prefix_ids), "teacher/block_size": block,
                      "time/prefill_s": self.prefill_s, "teacher/cache_gb": cache_bytes(self.text_config, len(self.prefix_ids)) / 1e9})
        logger.info("prefilled %d context tokens (%d cached) in %.1fs", len(self.prefix_ids), aligned, self.prefill_s)

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
        """Context size, cold prefill time (first, uncached prefill of the context
        block), the context's cache memory (bf16 KV + fp32 linear states), and
        this container's id, so a session can later stop only its own container
        (stop(container_ids=[...])) while other sessions share the app."""
        return {
            "container_id": os.environ.get("MODAL_TASK_ID"),
            "model_id": self.model_id,
            "prefix_tokens": len(self.prefix_ids),
            "prefill_s": self.prefill_s,
            "cache_gb": cache_bytes(self.text_config, len(self.prefix_ids)) / 1e9,
        }

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
        Sampling follows SAMPLING; temperature=0.0 is plain greedy decoding (no
        penalties), e.g. for the judge. logprobs=k (<= 20) also returns the
        teacher's raw top-k logprobs (temperature 1, before top-p/top-k/penalties)
        at every generated token. json_schema constrains the answer to JSON
        matching that schema (vLLM structured outputs)."""
        from vllm.sampling_params import StructuredOutputsParams

        _, prompt_ids = render(self.tokenizer, messages)
        full = (self.prefix_ids if with_context else []) + prompt_ids
        kwargs = sampling_kwargs(n, max_tokens, logprobs, temperature, seed, json_schema, StructuredOutputsParams)
        params = self.SamplingParams(**kwargs)
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
        self.run.log({
            "gen/ttft_s": ttft,
            "gen/latency_s": latency,
            "gen/prompt_tokens": len(full),
            "gen/output_tokens": sum(len(c.completion_ids) for c in completions),
            "gen/cached_tokens": getattr(out, "num_cached_tokens", None) or 0,
        })
        return completions

    @modal.exit()
    def stop(self):
        self.run.finish()


@app.cls(
    gpu="L4",
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=30 * MINUTES,
    startup_timeout=40 * MINUTES,
    scaledown_window=5 * MINUTES,
    retries=0,  # a failed call fails; callers bound waits with .get(timeout=...)
)
class TeacherScorer:
    """Teacher logprobs of given completions (HF transformers, batch 1: a
    forward over ~1-2k suffix tokens against the ~205k-token cache is already
    compute-bound). Run more containers for more throughput.

    NOT validated on GPU, and parked (lead, 2026-10-13): nothing on the
    critical path needs it (distillation uses Teacher.generate(logprobs=20)).
    The first GPU try OOMed in transformers' SDPA math fallback; the
    flash_attention backend above is the untested fix. Scorer itself is
    covered by the CPU tests."""

    model_id: str = modal.parameter(default=SMOKE_MODEL_ID)
    revision: str = modal.parameter(default="main")
    context_file: str = modal.parameter(default=CONTEXT_FILE)
    group: str = modal.parameter(default=SERVICE_GROUP)  # pool identity: leave the default

    @modal.enter()
    def start(self):
        import torch
        from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

        self.run = _start_run(
            "score", self.model_id, self.group,
            {"model_id": self.model_id, "revision": self.revision, "context_file": self.context_file, "engine": "hf"},
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        AttentionInterface.register(SCORER_ATTENTION, flash_attention)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, dtype=torch.bfloat16, attn_implementation=SCORER_ATTENTION
        )
        model = model.to("cuda").eval()
        volume.commit()
        self.scorer = Scorer(model, tokenizer)
        prefix_ids, _ = render(tokenizer, [{"role": "user", "content": "."}], load_context(self.context_file, REMOTE_CONTEXT_DIR))
        self.prefill_s = self.scorer.prefill(prefix_ids)
        self.run.log({"teacher/prefix_tokens": len(prefix_ids), "time/prefill_s": self.prefill_s,
                      "teacher/cache_gb": self.scorer.cache_bytes() / 1e9})
        self.run.log_gpu_memory()
        logger.info("prefilled %d context tokens in %.1fs", len(prefix_ids), self.prefill_s)

    @modal.method()
    def info(self) -> dict:
        return {"container_id": os.environ.get("MODAL_TASK_ID"), "model_id": self.model_id,
                "prefix_tokens": len(self.scorer.prefix_ids), "prefill_s": self.prefill_s,
                "cache_gb": self.scorer.cache_bytes() / 1e9}

    @modal.method()
    def logprobs(
        self, messages: list[dict], completion: str | list[int], top_k: int = MAX_TOP_K, with_context: bool = True
    ) -> TopK:
        t0 = time.perf_counter()
        result = self.scorer.logprobs(messages, completion, top_k, with_context)
        self.run.log({"score/latency_s": time.perf_counter() - t0,
                      "score/tokens": len(result.prompt_ids) + len(result.completion_ids)})
        return result

    @modal.method()
    def score_rows(self, rows: list[dict], with_context: bool = True) -> list[list[float]]:
        """Teacher-forced chosen-token logprobs of each row's completion_ids,
        given its prompt_ids (Target Store rows: the label's exact tokens, no
        re-rendering), with or without the journal context. For eval's CGC
        metric (base vs teacher NLL on the same tokens)."""
        t0 = time.perf_counter()
        out = [self.scorer.score(r["prompt_ids"], r["completion_ids"], 1, with_context)[2] for r in rows]
        self.run.log({"score/rows": len(rows), "score/latency_s": time.perf_counter() - t0,
                      "score/tokens": sum(len(r["completion_ids"]) for r in rows)})
        return out

    @modal.exit()
    def stop(self):
        self.run.finish()


@app.function(volumes={VOLUME_PATH: volume}, secrets=secrets, timeout=60 * MINUTES)
def download_weights(model_id: str = MODEL_ID, revision: str = "main") -> dict:
    """Download a model into the Volume's HF cache (where Teacher looks) from a
    CPU-only container, so a GPU container never pays for the download:

        uv run modal run -m cartridges.cartridge.teacher::download_weights --model-id Qwen/Qwen3.8-27B
    """
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    path = snapshot_download(model_id, revision=revision)
    seconds = time.perf_counter() - t0
    volume.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"model_id": model_id, "path": path, "gb": size / 1e9, "seconds": seconds}


def teacher_options(model_id: str, gpu: str | None = None, scaledown_window: int | None = None) -> dict:
    """with_options for a model's Teacher. The 27B model uses the decorator's
    TEACHER_OPTIONS as they are (no options: one container pool for every
    caller); other models get their GPU from GPU_FOR_MODEL."""
    options = {}
    if gpu is not None or model_id != MODEL_ID:
        options["gpu"] = gpu or GPU_FOR_MODEL[model_id]
    if scaledown_window is not None:
        options["scaledown_window"] = scaledown_window
    return options


def teacher_cls(
    name: str = "Teacher", model_id: str = SMOKE_MODEL_ID, gpu: str | None = None,
    scaledown_window: int | None = None, max_inputs: int | None = None, **params,
):
    """A handle on the deployed Teacher or TeacherScorer for `model_id`
    (options from teacher_options). Every option or parameter you pass makes a
    separate container pool, so 27B callers should pass none; `group` is
    dropped (experiment cost is attributed by time window, not by pool).

    max_inputs: concurrent requests per container (default 64), set as both
    Modal's input concurrency and vLLM's max_num_seqs. 128 fits an H100 for 4B
    with the 205k journal and 3072-token answers (~33 GB of ~60 GiB KV); the L4
    does not (keep 64 there, and fewer for long with-context answers)."""
    if name == "TeacherScorer" and model_id == MODEL_ID and gpu is None:
        gpu = "H200"  # its decorator's L4 can't hold 27B (52 GB weights + 13.6 GB context cache)
    options = teacher_options(model_id, gpu, scaledown_window)
    if name == "TeacherScorer":
        # Its decorator sets no cap: concurrent score_rows calls would otherwise start
        # one GPU container each (each re-prefilling the 205k context).
        options["max_containers"] = 1
    # `group` used to be passed here, and each value made its own container pool
    # (2026-10-15: a stray probe got its own L4). It is ignored now.
    if params.pop("group", None) is not None:
        logger.warning("teacher_cls: `group` is ignored; the Teacher runs in one service pool per model")
    cls = modal.Cls.from_name(APP_NAME, name)
    if options:
        cls = cls.with_options(**options)
    if max_inputs is not None:
        cls = cls.with_concurrency(max_inputs=max_inputs)
        params["max_num_seqs"] = max_inputs
    return cls(model_id=model_id, **params)


def running_containers(rows: list[dict], app_name: str = APP_NAME) -> list[str]:
    """Container ids of `app_name` in `modal container list --json` rows:
    snake_case keys (the CLI snake-cases its column names), start_time
    "Pending" while a container is starting. Raises on any other row shape, so
    a CLI format change can't make stop() silently match nothing."""
    for row in rows:
        if not {"container_id", "app_name"} <= row.keys():
            raise RuntimeError(f"unexpected `modal container list --json` row: {row}")
    return [r["container_id"] for r in rows if r["app_name"] == app_name]


def _list_containers(run, app_name: str) -> list[str]:
    import json

    listed = run(["modal", "container", "list", "--json"], capture_output=True, text=True, check=True)
    rows = json.loads(listed.stdout or "[]")
    ids = running_containers(rows, app_name)
    if rows and not ids:
        logger.warning("stop(): %d running container(s), none of app %r: %s",
                       len(rows), app_name, sorted({r["app_name"] for r in rows}))
    return ids


def stop(
    app_name: str = APP_NAME, run=None, recheck_s: float = 15.0, sleep=time.sleep,
    container_ids: list[str] | None = None,
) -> list[str]:
    """Stop this app's running containers now, instead of paying for the idle
    scaledown window. Call it at the end of a GPU session (the runbook does).
    Returns the stopped container ids. `run` defaults to subprocess.run.

    Cancel outstanding calls first (generate_all does it on any interrupt):
    Modal reschedules an unfinished input of a stopped container onto a NEW
    container (seen 2026-10-15, twice). After stopping, stop() waits
    `recheck_s` and logs an ERROR if a new container of the app has appeared,
    so a rescheduled input can't run unnoticed; stop it with
    `modal app stop --yes cartridges-teacher` (then redeploy).

    container_ids: stop only these (of the app's running containers). Without
    it, stop() stops EVERY container of the app, including ones other sessions
    are using (e.g. the user's serve gateway on the default 4B pool), so pass
    the session's own container id whenever another session may be live."""
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
            logger.error(
                "stop(): new %s container(s) %s after stopping %s: queued inputs were rescheduled. "
                "Cancel them, or run `modal app stop --yes %s` and redeploy.", app_name, new, ids, app_name,
            )
    return ids


def generate_all(
    teacher, conversations: list[list[dict]], window: int = 128, **kwargs
) -> list[list[Completion] | None]:
    """Teacher.generate for every conversation, results in order; a failed call
    gives None. Up to `window` requests run at once, each in its own worker
    thread (spawn, then a blocking get): every Modal spawn/get is a ~150 ms
    round trip, so doing them one after another cost ~80 s per 256 requests
    (datagen's profile, 2026-10-16). On any interrupt (KeyboardInterrupt, a
    timeout raised by the caller, ...) every unfinished call is cancelled
    before the exception propagates, so nothing is left queued to restart a
    container after stop()."""
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


@app.local_entrypoint()
def smoke(model_id: str = SMOKE_MODEL_ID, gpu: str = "", n_questions: int = 4, burst: int = 32):
    """Prefill the context, then time cached generation: one stream, then
    `burst` concurrent requests (the throughput the cost estimate uses)."""
    from cartridges.cartridge import tracking

    gpu = gpu or GPU_FOR_MODEL[model_id]
    # One container: ~10 min start + ~5 min of requests + the 5 min scaledown window.
    tracking.check_budget(20 / 60 * tracking.GPU_RATES_USD_PER_HOUR[gpu])
    questions = [
        "What was the author listening to on the morning of 2021-11-19?",
        "Which journals does this collection include?",
        "What mood ratings did the author record most often in late 2021?",
        "Who is Allie, according to the journals?",
    ][:n_questions]
    teacher = Teacher.with_options(gpu=gpu)(model_id=model_id)

    t0 = time.perf_counter()
    print("gen info", teacher.info.remote(), f"(container ready after {time.perf_counter() - t0:.0f}s)")
    for i, q in enumerate(questions):
        t0 = time.perf_counter()
        out = teacher.generate.remote([{"role": "user", "content": q}], max_tokens=128, logprobs=20)[0]
        print(f"[{i}] {time.perf_counter() - t0:.2f}s {len(out.completion_ids)} tok: {out.text[:200]!r}")
    base = teacher.generate.remote([{"role": "user", "content": questions[0]}], max_tokens=64, with_context=False)[0]
    print("base:", repr(base.text[:200]))

    t0 = time.perf_counter()
    kwargs = {"max_tokens": 128, "logprobs": 20, "temperature": 0.0}
    outs = list(teacher.generate.map([[{"role": "user", "content": questions[i % len(questions)] + f" ({i})"}]
                                       for i in range(burst)], kwargs=kwargs))
    wall, tokens = time.perf_counter() - t0, sum(len(o[0].completion_ids) for o in outs)
    print(f"burst: {burst} requests, {tokens} tokens in {wall:.1f}s = {tokens / wall:.0f} tok/s; "
          f"mean ttft {sum(o[0].ttft_s for o in outs) / burst:.2f}s")


def check_deployed(model_id: str = SMOKE_MODEL_ID) -> Completion:
    """Call the deployed app the way clients do (modal.Cls.from_name) and check
    that the result unpickles as this module's Completion; a script-mode deploy
    pickles it under module "teacher" instead. Run after every deploy:

        uv run python -m cartridges.cartridge.teacher
    """
    out = teacher_cls("Teacher", model_id).generate.remote(
        [{"role": "user", "content": "Say OK."}], max_tokens=8, temperature=0.0, with_context=False
    )[0]
    # By module name: under `python -m` this file is __main__, not the client's module.
    assert type(out).__module__ == "cartridges.cartridge.teacher", type(out)
    return out


if __name__ == "__main__":
    print(check_deployed())
