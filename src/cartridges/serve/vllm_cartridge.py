"""CartridgeEngine: cartridges on vLLM (the Teacher's engine and image), one
container pool per base model for every cartridge of that model.

The base model loads once; each request names its cartridge (a volume path,
exported with `cartridges cartridge_export_vllm`), and CartridgeConnector
(serve/vllm_connector.py) copies the export into the request's first p
positions, which the prompt fills with placeholder tokens vLLM never computes:

    prompt = [PLACEHOLDER] * p + teacher.render(tokenizer, messages)[1]

so the conversation starts at position p, as in HF's Cartridge.generate.
Decoding is greedy unless a temperature is given (then the Teacher's sampling
at that temperature). v1: prefix caching off, TP=1.

    uv run modal deploy -m cartridges.serve.vllm_cartridge
    engine = engine_cls("Qwen/Qwen3.5-4B")
    for event in engine.generate_stream.remote_gen("/vol/cartridges/exp-20261013-smoke/grpo", messages):
        ...
"""

# No `from __future__ import annotations`: Modal reads the modal.parameter
# annotations at class creation and needs real types.
import asyncio
import os
import time
import uuid
from pathlib import Path

import modal

from cartridges.cartridge import teacher
from cartridges.cartridge.export_vllm import EXPORT_DIR, STATE, load_manifest
from cartridges.cartridge.train_distill import VOLUME_PATH as CARTRIDGE_VOLUME_PATH

MINUTES = 60
APP_NAME = "cartridges-cartridge-vllm"
PLACEHOLDER = 0  # any id: vLLM never computes the cartridge's positions
MAX_NUM_SEQS = 64  # the Teacher's default; also the Modal input concurrency

GPU_FOR_MODEL = {teacher.MODEL_ID: "H200"}  # others: the decorator's L4/L40S

app = modal.App(APP_NAME, image=teacher.image)


def engine_kwargs(model_id: str, revision: str = "main", **overrides) -> dict:
    """vLLM engine arguments: the Teacher's (teacher.Teacher.start), so the demo
    compares two identical engines, except prefix caching (off in v1: the
    placeholder blocks of different cartridges would hash alike) and the
    cartridge connector. Also the parity check's, so both run the same engine."""
    from vllm.config import KVTransferConfig

    return {
        "model": model_id,
        "revision": revision,
        "dtype": "bfloat16",
        "max_model_len": 262144,
        "enable_prefix_caching": False,
        "language_model_only": True,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_logprobs": teacher.MAX_TOP_K,
        "gpu_memory_utilization": 0.92,  # the ~8% left also holds the connector's GPU store (<= 8 x 0.27 GB at 27B)
        "disable_cascade_attn": False,
        "kv_transfer_config": KVTransferConfig(
            kv_connector="CartridgeConnector",
            kv_connector_module_path="cartridges.serve.vllm_connector",
            kv_role="kv_consumer",
            kv_load_failure_policy="fail",  # never recompute the placeholders: that gives fluent nonsense
        ),
        **overrides,
    }


def engine_path(cartridge: str) -> str:
    """A cartridge's volume path (/vol/cartridges/..., as training writes it) as
    this container mounts the volume (teacher.VOLUME_PATH)."""
    return teacher.VOLUME_PATH + cartridge.removeprefix(CARTRIDGE_VOLUME_PATH)


def check_manifest(manifest: dict, model_id: str, vllm_version: str) -> None:
    """Refuse an export made for another base model or vLLM version."""
    if manifest["base_model"] != model_id:
        raise ValueError(f"cartridge {manifest['cartridge_id']} is for {manifest['base_model']}, this engine runs {model_id}")
    if manifest["vllm_version"] != vllm_version:
        raise ValueError(f"cartridge {manifest['cartridge_id']} was exported for vLLM {manifest['vllm_version']}, "
                         f"this engine runs {vllm_version}; re-export it")


def transfer_params(cartridge: str, model_id: str, vllm_version: str) -> dict:
    """The connector's kv_transfer_params for a cartridge's export (the volume
    is reloaded once when no export is visible yet)."""
    path = engine_path(cartridge)
    try:
        manifest = load_manifest(path)
    except FileNotFoundError:
        teacher.volume.reload()
        try:
            manifest = load_manifest(path)
        except FileNotFoundError:
            raise ValueError(f"no vLLM export of {cartridge}; run `cartridges cartridge_export_vllm "
                             f"--cartridge {cartridge}`") from None
    check_manifest(manifest, model_id, vllm_version)
    return {"cartridge_id": manifest["cartridge_id"], "path": str(Path(path) / EXPORT_DIR / STATE), "p": manifest["p"]}


@app.cls(
    gpu=["L4", "L40S"],  # the 4B default; L4s are often short
    max_containers=1,  # one pool per base model; a second container would serve nothing while the first starts
    scaledown_window=5 * MINUTES,
    volumes={teacher.VOLUME_PATH: teacher.volume, teacher.COMPILE_CACHE_PATH: teacher.compile_cache},
    secrets=teacher.secrets,
    timeout=30 * MINUTES,  # per call
    startup_timeout=30 * MINUTES,
    retries=0,  # a failed call fails; callers bound waits with .get(timeout=...)
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
class CartridgeEngine:
    """Chat generation with any exported cartridge of `model_id`."""

    model_id: str = modal.parameter(default=teacher.SMOKE_MODEL_ID)
    gpu: str = modal.parameter(default="")  # set by engine_cls with with_options(gpu=...); "": detected, for cost logging

    @modal.enter()
    async def start(self):
        import vllm
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
        from vllm.v1.engine.async_llm import AsyncLLM

        from cartridges.cartridge import tracking

        self.run = tracking.init_run(
            "eval", self.model_id, tracking.group_name("cartridge-vllm"),
            config={"model_id": self.model_id, "engine": "vllm", "max_num_seqs": MAX_NUM_SEQS},
            gpu=self.gpu or None, role="cartridge-vllm",
        )
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs(self.model_id)))
        self._stats_task = asyncio.create_task(self._log_stats_forever())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.vllm_version = vllm.__version__
        self.TokensPrompt, self.SamplingParams = TokensPrompt, SamplingParams
        self.DELTA, self.StructuredOutputsParams = RequestOutputKind.DELTA, StructuredOutputsParams
        teacher.compile_cache.commit()
        self.run.log_gpu_memory()

    async def _log_stats_forever(self, every_s: float = 10.0):
        while True:
            await asyncio.sleep(every_s)
            await self.engine.do_log_stats()

    async def _events(self, cartridge: str, messages: list[dict], max_tokens: int, temperature: float | None):
        params = transfer_params(cartridge, self.model_id, self.vllm_version)
        _, prompt_ids = teacher.render(self.tokenizer, messages)
        sampling = self.SamplingParams(
            **teacher.sampling_kwargs(1, max_tokens, 0, temperature or 0.0, None, None, self.StructuredOutputsParams),
            output_kind=self.DELTA, extra_args={"kv_transfer_params": params},
        )
        prompt = self.TokensPrompt(prompt_token_ids=[PLACEHOLDER] * params["p"] + prompt_ids)
        n, finish_reason, ttft, t0 = 0, None, None, time.perf_counter()
        async for out in self.engine.generate(prompt, sampling, uuid.uuid4().hex):
            choice = out.outputs[0]
            n += len(choice.token_ids)
            finish_reason = choice.finish_reason or finish_reason
            if choice.token_ids:
                ttft = ttft if ttft is not None else time.perf_counter() - t0
                yield {"text": choice.text, "tokens": len(choice.token_ids)}
        if finish_reason == "error":
            raise RuntimeError(f"vLLM failed the request for {params['cartridge_id']}")
        latency = time.perf_counter() - t0
        self.run.log({"gen/ttft_s": ttft or 0.0, "gen/latency_s": latency, "gen/prompt_tokens": len(prompt_ids),
                      "gen/output_tokens": n, "gen/decode_tok_s": (n - 1) / (latency - ttft) if n > 1 else 0.0})
        yield {"finish_reason": finish_reason, "prompt_tokens": len(prompt_ids), "completion_tokens": n}

    @modal.method()
    def info(self) -> dict:
        return {"container_id": os.environ.get("MODAL_TASK_ID"), "model_id": self.model_id,
                "vllm_version": self.vllm_version}

    @modal.method()
    async def generate(
        self, cartridge: str, messages: list[dict], max_tokens: int = 512, temperature: float | None = None
    ) -> dict:
        """One answer: {text, finish_reason, prompt_tokens, completion_tokens},
        as CartridgeServer.generate. prompt_tokens excludes the cartridge. No
        strip_think here (evals needs rapidfuzz, which the Teacher image lacks);
        thinking is off in render() and the gateway strips <think> anyway."""
        text, final = [], {}
        async for event in self._events(cartridge, messages, max_tokens, temperature):
            if "text" in event:
                text.append(event["text"])
            else:
                final = event
        return {"text": "".join(text), **final}

    @modal.method()
    async def generate_stream(
        self, cartridge: str, messages: list[dict], max_tokens: int = 512, temperature: float | None = None
    ):
        """`generate` streamed (call with .remote_gen): {text, tokens} per engine
        step, then {finish_reason, prompt_tokens, completion_tokens}."""
        async for event in self._events(cartridge, messages, max_tokens, temperature):
            yield event

    @modal.exit()
    def stop(self):
        self.run.finish()


def engine_cls(model_id: str = teacher.SMOKE_MODEL_ID, gpu: str | None = None):
    """A handle on the deployed CartridgeEngine for `model_id` (GPU from
    GPU_FOR_MODEL, else the decorator's); one container pool per model."""
    gpu = gpu or GPU_FOR_MODEL.get(model_id)
    cls = modal.Cls.from_name(APP_NAME, "CartridgeEngine")
    if gpu is None:
        return cls(model_id=model_id)
    return cls.with_options(gpu=gpu)(model_id=model_id, gpu=gpu)
