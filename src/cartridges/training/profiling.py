"""Measure, profile, and speed up single-page OCR inference on Modal.

Own Modal app ("cartridges-profiling"), reusing training's image and model
loaders. Two tools:

1. Single-page profiler (docs/plans/profiling.md, modes `generate` and
   `forward`): one model.infer() call exactly as `_run_inference` makes it,
   recorded with torch.profiler + stage labels + a 50 ms GPU sensor; writes
   trace.json.gz, stacks_{cpu,gpu}.folded, stages.json, ops.txt, gpu.csv,
   output.txt and report.html to /cache/profiling/<run_name>/:

   uv run modal run -m cartridges.training.profiling::run_profile \\
       --mode generate --split test --index 0 \\
       --model-source artifact --hf-revision run-liw4qzol --run-name prof-eval-ft-test0
   uv run python -m cartridges.training.profiling fetch prof-eval-ft-test0  # -> profiles/

2. 3-page timing bench (`run_bench`): loads a model once, then applies a list
   of *cumulative* optimization steps, re-measuring the same 3 fixed test
   pages after each (no traces):

    baseline   the current path: model.infer(), HF generate, HF no_repeat_ngram
    ngram      HF's NoRepeatNGramLogitsProcessor -> exact incremental equivalent
    merge      fold the LoRA into the base weights (merge_and_unload) -- fast, but
               changes greedy outputs; the fast path below leaves it unmerged
    moe_sync   MoE expert loop: one GPU->CPU sync per layer instead of per expert
    cuda_graph static KV cache + the 1-token decode step captured as a CUDA graph
               (routed experts run per-expert from stacked weights, LoRA unmerged)
    batched_moe  inside the graph, run the k routed experts as batched matmuls

Every step still goes through the model's own `infer()` (same prompt build,
image tiling, and output decoding) -- only `generate()` underneath it changes,
so outputs are directly comparable to the current path.

Run (detached; prints a CALL_ID to poll). The fastest output-preserving path
measured (Unsloth load, unmerged adapter, ~124 decode tok/s on an A100):
uv run modal run --detach -m cartridges.training.profiling::run_bench \\
    --run-name fast --loader unsloth --steps moe_sync,cuda_graph,batched_moe
uv run python -m cartridges.training.profiling poll <CALL_ID> profiles/fast.json
uv run python -m cartridges.training.profiling_report profiles/fast.json
"""

from __future__ import annotations

import contextlib
import inspect
import json
import sys
import time

import modal
from pydantic import BaseModel

from cartridges.training.config import EvalConfig, ModelSource, TrainConfig
from cartridges.training.finetune import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    _stage,
    hf_secret,
    image,
    logger,
    model_cache,
    wandb_secret,
)
from cartridges.training.fast_decode import (
    RANGES,
    CudaGraphDecoder,
    _IncrementalNoRepeatNGram,
    _moe_modules,
    _patch_moe,
)
from cartridges.training.perf_metrics import (
    GenerationTiming,
    mean_in_window,
    parse_nvidia_smi_line,
)

MINUTES = 60
PROFILING_DIR = "/cache/profiling"  # the only place on the shared volume we write

# Fixed test-split pages, picked by reference token length (DeepSeek tokenizer):
# p10 / median / p90 of the 78 test pages = 87 / 249 / 611 tokens.
PAGES = {"short": 25, "median": 19, "long": 48}
STACK_PASS_MAX_NEW_TOKENS = 4

STEPS = ("baseline", "ngram", "merge", "moe_sync", "cuda_graph", "batched_moe")

app = modal.App("cartridges-profiling", image=image)


class ProfileConfig(BaseModel):
    run_name: str
    # "unsloth": training's FastVisionModel load (the current inference path).
    # "plain": transformers AutoModel, bf16, no Unsloth patches.
    loader: str = "unsloth"
    steps: list[str] = ["baseline"]
    # Also measure the base model (no adapter) at baseline before attaching it.
    measure_base: bool = False
    # Any `run-<wandb_id>` tag (or branch/sha) in TrainConfig.hf_repo_id;
    # attached unmerged to a fresh base load. "" = base model only.
    hf_revision: str = "run-liw4qzol"
    # Same clamp `_capped_generate` applies on the current eval path.
    max_new_tokens: int = TrainConfig().max_new_tokens
    # Measure every test-split page, not just the 3 fixed ones (a corpus-level
    # CER check that a speedup didn't change transcription quality).
    all_test_pages: bool = False


# --- GPU utilization sampling ----------------------------------------------


class GpuSampler:
    """`nvidia-smi -lms <interval>` in the background; samples stamped with
    this process's perf_counter clock so they line up with generation timers."""

    def __init__(self, interval_ms: int = 200):
        import subprocess
        import threading

        self.samples: list[tuple[float, float, float]] = []  # (t, util %, mem MiB)
        self.proc = subprocess.Popen(
            [
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", "-lms", str(interval_ms),
            ],
            stdout=subprocess.PIPE, text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            parsed = parse_nvidia_smi_line(line)
            if parsed:
                self.samples.append((time.perf_counter(), *parsed))

    def util(self, t0: float, t1: float) -> float | None:
        return mean_in_window([(t, u) for t, u, _ in self.samples], t0, t1)

    def stop(self):
        self.proc.terminate()


# --- generate() interception -------------------------------------------------


class _FirstTokenTimer:
    """HF logits processor that records when the first token's logits exist
    (i.e. prefill finished). Duck-typed: LogitsProcessorList just calls it."""

    def __init__(self):
        self.t_first: float | None = None

    def mark(self):
        import torch

        if self.t_first is None:
            torch.cuda.synchronize()
            self.t_first = time.perf_counter()

    def __call__(self, input_ids, scores):
        self.mark()
        return scores


class _State:
    """What the generate() wrapper should do; mutated by the optimization steps."""

    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens
        self.fast_ngram = False
        self.graph_decoder: CudaGraphDecoder | None = None
        # generate() kwarg overrides over infer()'s hardcoded ones (DecodingConfig)
        self.decoding: dict = {}
        # `forward` profile mode: one prompt forward pass, no decoding
        self.forward_only = False
        self.last: dict = {}
        # CPU wall time of each logits-processor-list call in the current
        # generate(), and the same split by processor class
        self.processor_s: list[float] = []
        self.processor_times: dict[str, list[float]] = {}


def _install_stage_hooks(inner) -> None:
    """Label vision encoder / projector / language-model calls inside infer()."""
    lm = inner.model
    lm.register_forward_pre_hook(lambda module, args: RANGES.lm_call())
    labelled = [(f"vision_encode.{attr}", getattr(lm, attr)) for attr in ("sam_model", "qwen2_model", "projector")]
    labelled += [(f"lm.layer{i:02d}", layer) for i, layer in enumerate(lm.layers)]
    # DeepseekV2MoE (routing + expert loop + shared experts), in every layer
    # after first_k_dense_replace; one label so its total per step is direct.
    labelled += [("lm.moe", layer.mlp) for layer in lm.layers if hasattr(layer.mlp, "experts")]
    for name, module in labelled:
        module.register_forward_pre_hook(lambda m, a, name=name: RANGES.push(name))
        module.register_forward_hook(lambda m, a, o, name=name: RANGES.pop(name))


def _time_logits_processors(state: _State) -> None:
    """Replace HF's LogitsProcessorList.__call__ (called once per generate
    step with every processor, e.g. NoRepeatNGram) with an equivalent loop
    that labels and times each processor separately."""
    from transformers.generation.logits_process import LogitsProcessorList

    def timed(self, input_ids, scores, **kwargs):
        RANGES.push("logits_processors")
        start = time.perf_counter()
        try:
            for processor in self:
                name = type(processor).__name__
                RANGES.push(f"logits_processor.{name}")
                t0 = time.perf_counter()
                # HF passes extra kwargs only to processors that take them
                # (none of infer()'s do); same rule here.
                if kwargs and len(inspect.signature(processor.__call__).parameters) > 2:
                    scores = processor(input_ids, scores, **kwargs)
                else:
                    scores = processor(input_ids, scores)
                state.processor_times.setdefault(name, []).append(time.perf_counter() - t0)
                RANGES.pop(f"logits_processor.{name}")
            return scores
        finally:
            state.processor_s.append(time.perf_counter() - start)
            RANGES.pop("logits_processors")

    LogitsProcessorList.__call__ = timed


@contextlib.contextmanager
def _instrumented_generate(inner, state: _State):
    """Patch `inner.generate` (what `infer()` calls -- note: on a PeftModel,
    infer() runs on the *inner* model, so patching the PeftModel's generate,
    as `_capped_generate` does, never takes effect) to clamp max_new_tokens,
    time prefill vs decode, and route through the current optimizations."""
    import torch
    from transformers import LogitsProcessorList

    original = inner.generate

    def generate(input_ids, **kwargs):
        RANGES.pop("preprocess")  # opened by _infer(): prompt build + image tiling
        RANGES.reset_calls()
        state.processor_s = []
        state.processor_times = {}
        kwargs.update(state.decoding)
        kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", state.max_new_tokens), state.max_new_tokens)
        timer = _FirstTokenTimer()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        RANGES.push("generate_setup")  # closed by the first language-model call
        if state.forward_only:
            out = inner(
                input_ids=input_ids, images=kwargs["images"], images_seq_mask=kwargs["images_seq_mask"],
                images_spatial_crop=kwargs["images_spatial_crop"], use_cache=False,
            )
            timer.mark()
            output_ids = torch.cat([input_ids, out.logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        elif state.graph_decoder is not None:
            output_ids = state.graph_decoder.generate(input_ids, timer=timer, state=state, **kwargs)
        else:
            processors = []
            if state.fast_ngram:
                ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
                if ngram_size:
                    processors.append(_IncrementalNoRepeatNGram(ngram_size))
            processors.append(timer)
            output_ids = original(input_ids, logits_processor=LogitsProcessorList(processors), **kwargs)
        RANGES.end_lm()
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        # The first processor call belongs to prefill (and absorbs its sync).
        decode_processor_s = state.processor_s[1:]
        state.last = {
            "t_start": t_start,
            "t_first": timer.t_first,
            "t_end": t_end,
            # host timestamps of each LM call: [prefill, step 1, step 2, ...]
            "lm_call_times": list(RANGES.call_times),
            "processor_ms_per_step": {
                name: 1e3 * sum(times[1:]) / len(times[1:]) for name, times in state.processor_times.items()
                if len(times) > 1
            },
            "n_output_tokens": int(output_ids.shape[1] - input_ids.shape[1]),
            "n_prompt_tokens": int(input_ids.shape[1]),
            "logits_processor_ms_per_step": (
                1e3 * sum(decode_processor_s) / len(decode_processor_s) if decode_processor_s else None
            ),
        }
        return output_ids

    inner.generate = generate
    try:
        yield
    finally:
        inner.generate = original


# --- model loading -----------------------------------------------------------


def _load(config: ProfileConfig):
    """Returns (model, tokenizer); see `_inner()` for the module whose
    infer()/generate() actually run (the PeftModel's base once wrapped)."""
    if config.loader == "unsloth":
        from cartridges.training.finetune import _load_base_model
        from unsloth import FastVisionModel

        model, tokenizer = _load_base_model(TrainConfig())
        FastVisionModel.for_inference(model)
        return model, tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    base_model_dir = TrainConfig().base_model_dir
    sys.path.insert(0, "/cache")
    with _stage("load_plain_model", base_model_dir=base_model_dir):
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", use_safetensors=True,
        ).cuda().eval()
    logger.info("attn_implementation=%s", model.config._attn_implementation)
    return model, tokenizer


def _attach_adapter(model, config: ProfileConfig):
    from peft import PeftModel

    eval_config = EvalConfig(model_source=ModelSource.ARTIFACT, hf_revision=config.hf_revision)
    with _stage("attach_adapter", hf_revision=config.hf_revision):
        model = PeftModel.from_pretrained(
            model, eval_config.train_config.hf_repo_id, revision=eval_config.hf_revision
        )
    if config.loader == "unsloth":
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
    return model


def _inner(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


# --- measurement -------------------------------------------------------------


def _infer(model, tokenizer, pil_image, tmpdir: str, max_new_tokens: int):
    """One page exactly like `finetune._run_inference`: image to a temp file,
    then `model.infer(...)` with the same kwargs, inside `_capped_generate`.
    (On a PeftModel that cap never reaches the inner generate() infer() calls;
    `_instrumented_generate` applies it there instead.)"""
    from pathlib import Path

    from cartridges.training.collator import FREE_OCR_PROMPT
    from cartridges.training.finetune import _capped_generate

    train_config = TrainConfig()
    RANGES.push("save_image")
    image_path = Path(tmpdir) / "page.png"
    pil_image.save(image_path)
    RANGES.pop("save_image")
    with _capped_generate(model, max_new_tokens):
        RANGES.push("preprocess")  # closed by the generate() wrapper
        try:
            return model.infer(
                tokenizer, prompt=FREE_OCR_PROMPT, image_file=str(image_path), output_path=tmpdir,
                base_size=train_config.base_size, image_size=train_config.image_size,
                crop_mode=train_config.crop_mode, save_results=False, test_compress=False,
                eval_mode=True,
            )
        finally:
            RANGES.pop("preprocess")  # no-op unless infer() failed before generate()


def _label_decode_text(tokenizer) -> None:
    """infer()'s output detokenization runs through tokenizer.decode."""
    original = tokenizer.decode

    def decode(*args, **kwargs):
        RANGES.push("decode_text")
        try:
            return original(*args, **kwargs)
        finally:
            RANGES.pop("decode_text")

    tokenizer.decode = decode


def _measure(label, model, tokenizer, pages, state, sampler, reference_hyps) -> list[dict]:
    import tempfile

    import jiwer
    import torch

    inner = _inner(model)
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir, _instrumented_generate(inner, state):
        with _stage("warmup", step=label):
            _infer(model, tokenizer, pages["short"]["image"], tmpdir, state.max_new_tokens)
        for name, page in pages.items():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hypothesis = _infer(model, tokenizer, page["image"], tmpdir, state.max_new_tokens)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            gen = state.last
            timing = GenerationTiming(gen["t_start"], gen["t_first"], gen["t_end"], gen["n_output_tokens"])
            ref_hyp = reference_hyps.get(name)
            row = {
                "step": label,
                "page": name,
                "index": page["index"],
                "prompt_tokens": gen["n_prompt_tokens"],
                "output_tokens": gen["n_output_tokens"],
                "prefill_s": timing.prefill_s,
                "decode_tok_s": timing.decode_tok_s,
                "generate_s": timing.generate_s,
                "page_wall_s": wall,
                "preprocess_s": wall - timing.generate_s,
                "logits_processor_ms_per_step": gen["logits_processor_ms_per_step"],
                "gpu_util_pct": sampler.util(gen["t_start"], gen["t_end"]),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "cer_vs_reference": jiwer.cer(page["text"], hypothesis),
                "cer_vs_baseline": jiwer.cer(ref_hyp, hypothesis) if ref_hyp is not None else None,
                "matches_baseline": hypothesis == ref_hyp if ref_hyp is not None else None,
                "hypothesis": hypothesis,
            }
            logger.info(
                "%s %s: prefill=%.2fs decode=%s tok/s out=%d wall=%.2fs util=%s proc=%sms match=%s",
                label, name, row["prefill_s"],
                f"{row['decode_tok_s']:.1f}" if row["decode_tok_s"] else "n/a",
                row["output_tokens"], wall, row["gpu_util_pct"],
                row["logits_processor_ms_per_step"], row["matches_baseline"],
            )
            rows.append(row)
    return rows


def _gzip(path: str) -> str:
    import gzip
    import os
    import shutil

    with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)
    return path + ".gz"


def _apply_step(step: str, model, state: _State):
    """Apply one optimization (see module docstring); returns the (possibly new) model."""
    with _stage("apply_step", step=step):
        if step == "ngram":
            state.fast_ngram = True
        elif step == "merge":
            model = model.merge_and_unload()
        elif step == "moe_sync":
            logger.info("patched MoE classes: %d", _patch_moe(model))
        elif step == "cuda_graph":
            _patch_moe(model)
            state.graph_decoder = CudaGraphDecoder(_inner(model))
        elif step == "batched_moe":  # requires cuda_graph earlier
            for moe in _moe_modules(model):
                moe._batched = True
            state.graph_decoder.graph = None  # re-capture with the new MoE path
        elif step != "baseline":
            raise ValueError(f"unknown step {step!r}; choose from {STEPS}")
    return model


def _install_instrumentation(model, tokenizer, state: _State) -> None:
    """Stage labels (vision / prefill / decode_step / logits processors /
    decode_text) and per-processor timing. The hooked modules survive
    PeftModel wrapping and merging."""
    _install_stage_hooks(_inner(model))
    _time_logits_processors(state)
    _label_decode_text(tokenizer)


@app.function(
    gpu="A100",  # same GPU as evaluate()
    timeout=4 * 60 * MINUTES,  # all_test_pages runs the slow current path twice
    volumes={"/cache": model_cache},
    secrets=[hf_secret],
)
def profile_inference(config: ProfileConfig) -> dict:
    """3-page timing bench (no traces): every step in `config.steps` applied
    cumulatively, the fixed pages re-measured after each."""
    import os

    import torch
    from datasets import load_from_disk

    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {unknown}; choose from {STEPS}")

    with _stage("load_dataset", split="test"):
        test = load_from_disk(TrainConfig().dataset_dir)["test"]
        page_indices = dict(PAGES)
        if config.all_test_pages:
            page_indices.update({f"test_{i}": i for i in range(len(test)) if i not in PAGES.values()})
        pages = {name: {"index": i, **test[i]} for name, i in page_indices.items()}

    os.makedirs(f"{PROFILING_DIR}/results", exist_ok=True)
    result_path = f"{PROFILING_DIR}/results/{config.run_name}.json"
    result = {"config": config.model_dump(), "gpu": torch.cuda.get_device_name(), "rows": []}

    def save():
        with open(result_path, "w") as f:
            json.dump(result, f, indent=1)
        model_cache.commit()

    sampler = GpuSampler()
    state = _State(config.max_new_tokens)
    try:
        model, tokenizer = _load(config)
        _install_instrumentation(model, tokenizer, state)
        if config.measure_base:
            result["rows"] += _measure("base/baseline", model, tokenizer, pages, state, sampler, {})
            save()

        if config.hf_revision:  # "" = run the steps on the base model, no adapter
            model = _attach_adapter(model, config)
        reference_hyps: dict[str, str] = {}
        for step in config.steps:
            model = _apply_step(step, model, state)
            rows = _measure(step, model, tokenizer, pages, state, sampler, reference_hyps)
            # Every later step in this run is compared to the first one.
            reference_hyps = reference_hyps or {r["page"]: r["hypothesis"] for r in rows}
            result["rows"] += rows
            save()
    finally:
        sampler.stop()
    return result


# --- single-page profiler (docs/plans/profiling.md, modes generate/forward) --


class DecodingConfig(BaseModel):
    """Overrides for the generate() kwargs infer() hardcodes (eval_mode:
    greedy, no_repeat_ngram_size=35, eos_token_id=tokenizer.eos_token_id).
    None keeps infer()'s value. Same fields as the DecodingConfig being added
    to training/config.py on main (kept local here to avoid a conflict)."""

    no_repeat_ngram_size: int | None = None
    repetition_penalty: float | None = None
    eos_token_id: int | None = None


class PageProfileConfig(BaseModel):
    run_name: str
    mode: str = "generate"  # "generate": one model.infer(); "forward": prompt forward only
    split: str = "test"
    index: int = 0
    model_source: ModelSource = ModelSource.BASE
    hf_revision: str | None = None  # required for model_source=artifact
    merge_adapter: bool = False
    max_new_tokens: int = 64
    decoding: DecodingConfig = DecodingConfig()
    with_stack: bool = True
    # Optional speedups from STEPS (e.g. ["moe_sync", "cuda_graph",
    # "batched_moe"]) applied before the pass, to profile the fast path too.
    optimizations: list[str] = []
    log_wandb: bool = True


def _load_for_profile(config: PageProfileConfig):
    """Model load with the existing eval functions, as `evaluate()` does."""
    from unsloth import FastVisionModel

    from cartridges.training.finetune import _load_artifact_model, _load_base_model

    if config.model_source == ModelSource.ARTIFACT:
        if not config.hf_revision:
            raise ValueError("hf_revision is required when model_source='artifact'")
        model, tokenizer = _load_artifact_model(
            EvalConfig(model_source=ModelSource.ARTIFACT, hf_revision=config.hf_revision)
        )
    else:
        model, tokenizer = _load_base_model(TrainConfig())
    merge_s = None
    if config.merge_adapter:
        if config.model_source != ModelSource.ARTIFACT:
            raise ValueError("merge_adapter needs model_source='artifact'")
        start = time.monotonic()
        with _stage("merge_adapter", hf_revision=config.hf_revision):
            model = model.merge_and_unload()  # into the bf16 weights of this fresh load
        merge_s = time.monotonic() - start
    FastVisionModel.for_inference(model)  # as _run_inference does
    return model, tokenizer, merge_s


def _flat_numbers(d: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat_numbers(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


@app.function(
    gpu="A100",
    timeout=30 * MINUTES,
    # a with_stack Chrome trace of a 64-token eager decode is ~1 GB of JSON
    # before gzip; trace_stats() loads it whole
    memory=32 * 1024,
    volumes={"/cache": model_cache},
    secrets=[hf_secret, wandb_secret],
)
def profile_page(config: PageProfileConfig) -> dict:
    """One page, one recorded pass -> /cache/profiling/<run_name>/:
    trace.json.gz, stacks_cpu.folded, stacks_gpu.folded, stages.json, ops.txt,
    gpu.csv, output.txt, report.html."""
    import os
    import tempfile

    import torch
    from datasets import load_from_disk
    from torch.profiler import ProfilerActivity, profile

    from cartridges.training.perf_metrics import (
        collapse_folded,
        op_tables,
        stage_coverage,
        step_durations,
        summarize,
        trace_stats,
    )
    from cartridges.training.profile_page_report import render_report

    if config.mode not in ("generate", "forward"):
        raise ValueError("mode must be 'generate' or 'forward' (train-step lives in profiling_train.py)")
    out_dir = f"{PROFILING_DIR}/{config.run_name}"
    os.makedirs(out_dir, exist_ok=True)

    with _stage("load_page", split=config.split, index=config.index):
        page = load_from_disk(TrainConfig().dataset_dir)[config.split][config.index]
    load_start = time.monotonic()
    model, tokenizer, merge_s = _load_for_profile(config)
    load_s = time.monotonic() - load_start

    state = _State(config.max_new_tokens)
    state.decoding = config.decoding.model_dump(exclude_none=True)
    state.forward_only = config.mode == "forward"
    _install_instrumentation(model, tokenizer, state)
    for step in config.optimizations:
        model = _apply_step(step, model, state)

    def timed_infer(tmpdir: str, cap: int):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = _infer(model, tokenizer, page["image"], tmpdir, cap)
        torch.cuda.synchronize()
        return text, t0, time.perf_counter()

    def decode_tok_s(gen: dict) -> float | None:
        return GenerationTiming(gen["t_start"], gen["t_first"], gen["t_end"], gen["n_output_tokens"]).decode_tok_s

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    inner = _inner(model)
    with tempfile.TemporaryDirectory() as tmpdir, _instrumented_generate(inner, state):
        with _stage("warmup", max_new_tokens=8):
            _infer(model, tokenizer, page["image"], tmpdir, 8)
        with _stage("unprofiled_pass", max_new_tokens=config.max_new_tokens):
            unprofiled_text, _, _ = timed_infer(tmpdir, config.max_new_tokens)
            unprofiled = dict(state.last)
        torch.cuda.reset_peak_memory_stats()
        sampler = GpuSampler(interval_ms=50)
        try:
            # Timeline pass: no Python stacks. A with_stack Chrome trace of
            # 64 eager decode steps (~250k events) took >10 min to export.
            with _stage("profiled_pass", record_shapes=True), profile(
                activities=activities, record_shapes=True
            ) as prof:
                text, t0, t1 = timed_infer(tmpdir, config.max_new_tokens)
        finally:
            sampler.stop()
        gen = dict(state.last)
        # Export before any other profiler session starts: exporting after the
        # stack pass below produced ts=0/dur=0 for every CUDA runtime, kernel
        # and memcpy event of this pass.
        with _stage("export_trace", out_dir=out_dir):
            prof.export_chrome_trace(f"{out_dir}/trace.json")
            with open(f"{out_dir}/trace.json") as f:
                events = json.load(f)["traceEvents"]
            stats = trace_stats(events)
            # from the trace JSON, not prof.key_averages() (~8 min of Python here)
            with open(f"{out_dir}/ops.txt", "w") as f:
                f.write(op_tables(events, n=30))
            del events
            _gzip(f"{out_dir}/trace.json")
        stack_prof = None
        if config.with_stack:
            # Separate pass for the flame graphs only (stack capture also
            # inflates CPU timings, so it stays out of the timeline above).
            # export_stacks is slow (Python over every stacked event: >10 min
            # for 64 eager steps), so this pass decodes fewer tokens; the
            # per-step pattern repeats, so flame-graph proportions hold.
            with _stage("stack_pass", max_new_tokens=STACK_PASS_MAX_NEW_TOKENS), profile(
                activities=activities, with_stack=True,
                # without verbose=True, export_stacks writes empty files
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            ) as stack_prof:
                RANGES.enabled = False
                try:
                    stack_text, _, _ = timed_infer(tmpdir, min(config.max_new_tokens, STACK_PASS_MAX_NEW_TOKENS))
                finally:
                    RANGES.enabled = True

    with _stage("write_outputs", out_dir=out_dir):
        with open(f"{out_dir}/output.txt", "w") as f:
            f.write(text)
        if stack_prof is not None:
            for name, metric in (("stacks_cpu", "self_cpu_time_total"), ("stacks_gpu", "self_cuda_time_total")):
                raw = f"{out_dir}/{name}.raw"
                stack_prof.export_stacks(raw, metric)
                # export_stacks writes one line per event (~400 MB); collapse
                # identical stacks into the standard folded format
                with open(raw) as f:
                    folded = collapse_folded(f)
                with open(f"{out_dir}/{name}.folded", "w") as f:
                    f.write("\n".join(folded) + "\n")
                os.remove(raw)
        gpu_rows = [(t - t0, util, mem) for t, util, mem in sampler.samples if t0 - 0.5 <= t <= t1 + 0.5]
        with open(f"{out_dir}/gpu.csv", "w") as f:
            f.write("t_s,util_pct,mem_mib\n")
            f.writelines(f"{t:.3f},{u:.0f},{m:.0f}\n" for t, u, m in gpu_rows)

        steps_s = step_durations(gen["lm_call_times"][1:], gen["t_end"])
        stage_ms = stats.pop("stage_ms")
        wall_ms = 1e3 * (t1 - t0)
        stages = {
            "config": config.model_dump(mode="json"),
            "gpu": torch.cuda.get_device_name(),
            "load_s": load_s,
            "merge_s": merge_s,
            "wall_ms": wall_ms,
            "ttft_ms": 1e3 * (gen["t_first"] - gen["t_start"]),
            "prompt_tokens": gen["n_prompt_tokens"],
            "output_tokens": gen["n_output_tokens"],
            "stage_ms": stage_ms,
            "stage_coverage": stage_coverage(stage_ms, wall_ms),
            "decode_step_ms": summarize([1e3 * s for s in steps_s]),
            "decode_step_ms_all": [round(1e3 * s, 3) for s in steps_s],
            "logits_processor_ms_per_step": gen["processor_ms_per_step"],
            "trace_stats": stats,
            "gpu_util_mean_pct": mean_in_window([(t, u) for t, u, _ in gpu_rows], 0.0, t1 - t0),
            "decode_tok_s_profiled": decode_tok_s(gen),
            "decode_tok_s_unprofiled": decode_tok_s(unprofiled),
            "output_matches_unprofiled": text == unprofiled_text,
            # stack pass decodes fewer tokens: compare against the same prefix
            "stack_pass_output_is_prefix": unprofiled_text.startswith(stack_text) if stack_prof is not None else None,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        with open(f"{out_dir}/stages.json", "w") as f:
            json.dump(stages, f, indent=1)
        with open(f"{out_dir}/report.html", "w") as f:
            f.write(render_report(stages, gpu_rows, open(f"{out_dir}/ops.txt").read()))
        model_cache.commit()

    if config.log_wandb:
        # Summary numbers only -- no traces, no page text.
        with _stage("log_wandb", group="infra-profiling"):
            import wandb

            run = wandb.init(
                entity=WANDB_ENTITY, project=WANDB_PROJECT, group="infra-profiling",
                name=config.run_name, job_type="profile", config=config.model_dump(mode="json"),
            )
            run.summary.update(_flat_numbers({k: v for k, v in stages.items() if k not in ("config", "decode_step_ms_all")}))
            run.finish()

    logger.info(
        "profile %s: wall=%.0fms ttft=%.0fms steps=%s coverage=%s match_unprofiled=%s stats=%s",
        config.run_name, wall_ms, stages["ttft_ms"], stages["decode_step_ms"], stages["stage_coverage"],
        stages["output_matches_unprofiled"], stats,
    )
    return {k: v for k, v in stages.items() if k != "decode_step_ms_all"}


# --- local entrypoints / CLI -------------------------------------------------


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@app.local_entrypoint()
def run_profile(
    run_name: str,
    mode: str = "generate",
    split: str = "test",
    index: int = 0,
    model_source: str = "base",
    hf_revision: str = "",
    merge_adapter: bool = False,
    max_new_tokens: int = 64,
    no_repeat_ngram_size: int = -1,  # -1 = keep infer()'s value (35)
    repetition_penalty: float = -1.0,  # -1 = keep infer()'s value (none)
    eos_token_id: int = -1,  # -1 = keep infer()'s value
    with_stack: bool = True,
    optimizations: str = "",
    log_wandb: bool = True,
):
    """Single-page profiler (docs/plans/profiling.md). Spawns the job, polls
    with short reconnects, then copies the outputs to profiles/<run_name>/.
    With --detach the job survives a dropped client; re-fetch with
    `python -m cartridges.training.profiling fetch <run_name>`."""
    decoding = DecodingConfig(
        no_repeat_ngram_size=None if no_repeat_ngram_size < 0 else no_repeat_ngram_size,
        repetition_penalty=None if repetition_penalty < 0 else repetition_penalty,
        eos_token_id=None if eos_token_id < 0 else eos_token_id,
    )
    config = PageProfileConfig(
        run_name=run_name, mode=mode, split=split, index=index, model_source=ModelSource(model_source),
        hf_revision=hf_revision or None, merge_adapter=merge_adapter, max_new_tokens=max_new_tokens,
        decoding=decoding, with_stack=with_stack, optimizations=_split(optimizations), log_wandb=log_wandb,
    )
    call = profile_page.spawn(config)
    print(f"CALL_ID {call.object_id}", flush=True)
    _wait(call.object_id)
    _fetch(run_name)


@app.local_entrypoint()
def run_bench(
    run_name: str,
    loader: str = "unsloth",
    steps: str = "baseline",
    measure_base: bool = False,
    hf_revision: str = "run-liw4qzol",
    all_test_pages: bool = False,
):
    """3-page timing bench across cumulative optimization steps (detached; poll it)."""
    config = ProfileConfig(
        run_name=run_name, loader=loader, steps=_split(steps), measure_base=measure_base,
        hf_revision=hf_revision, all_test_pages=all_test_pages,
    )
    call = profile_inference.spawn(config)
    print(f"CALL_ID {call.object_id}")


def _wait(call_id: str):
    """Short reconnects instead of one long blocking .get() (see finetune.py's
    note on dropped long-lived connections)."""
    while True:
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=60)
        except TimeoutError:
            print("still running...", flush=True)


def _poll(call_id: str, out_path: str) -> None:
    result = _wait(call_id)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out_path}")


def _fetch(run_name: str, local_root: str = "profiles") -> None:
    """Copy /cache/profiling/<run_name>/ from the volume to profiles/<run_name>/."""
    from pathlib import Path

    volume = modal.Volume.from_name("cartridges-training-cache")
    remote = f"profiling/{run_name}"
    for entry in volume.listdir(remote, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest = Path(local_root) / run_name / Path(entry.path).relative_to(remote)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        print(f"fetched {dest}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "poll":
        _poll(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "fetch":
        _fetch(sys.argv[2])
    else:
        sys.exit(
            "usage: python -m cartridges.training.profiling poll <CALL_ID> <out.json>\n"
            "       python -m cartridges.training.profiling fetch <run_name>"
        )
