"""Measure, profile, and speed up single-page OCR inference on Modal.

Own Modal app ("cartridges-profiling"), reusing training's image and model
loader. One call loads a model once, then applies a list of *cumulative*
optimization steps, re-measuring the same 3 fixed test pages after each:

    baseline   the current path: model.infer(), HF generate, HF no_repeat_ngram
    ngram      HF's NoRepeatNGramLogitsProcessor -> exact incremental equivalent
    merge      fold the LoRA into the base weights (merge_and_unload)
    moe_sync   MoE expert loop: one GPU->CPU sync per layer instead of per expert
    cuda_graph static KV cache + the 1-token decode step captured as a CUDA graph

Every step still goes through the model's own `infer()` (same prompt build,
image tiling, and output decoding) -- only `generate()` underneath it changes,
so outputs are directly comparable to the current path.

Run (detached; prints a CALL_ID to poll):
uv run modal run --detach -m cartridges.training.profiling::run_profile \\
    --loader unsloth --steps baseline,ngram,merge,moe_sync --run-name unsloth
uv run python -m cartridges.training.profiling poll <CALL_ID> profiles/unsloth.json
"""

from __future__ import annotations

import contextlib
import json
import sys
import time

import modal
from pydantic import BaseModel

from cartridges.training.config import EvalConfig, ModelSource, TrainConfig
from cartridges.training.finetune import _stage, hf_secret, image, logger, model_cache
from cartridges.training.perf_metrics import (
    GenerationTiming,
    NoRepeatNGramBlocker,
    mean_in_window,
    parse_nvidia_smi_line,
)

MINUTES = 60
PROFILING_DIR = "/cache/profiling"  # the only place on the shared volume we write

# Fixed test-split pages, picked by reference token length (DeepSeek tokenizer):
# p10 / median / p90 of the 78 test pages = 87 / 249 / 611 tokens.
PAGES = {"short": 25, "median": 19, "long": 48}
PROFILE_PAGE = "median"
# torch.profiler with with_stack on a full page makes a multi-hundred-MB
# trace; 48 decode steps is plenty to see the per-step pattern.
PROFILE_MAX_NEW_TOKENS = 48

STEPS = ("baseline", "ngram", "merge", "moe_sync", "cuda_graph")

app = modal.App("cartridges-profiling", image=image)


class ProfileConfig(BaseModel):
    run_name: str
    # "unsloth": training's FastVisionModel load (the current inference path).
    # "plain": transformers AutoModel, bf16, no Unsloth patches.
    loader: str = "unsloth"
    steps: list[str] = ["baseline"]
    # Also measure the base model (no adapter) at baseline before attaching it.
    measure_base: bool = False
    hf_revision: str = "run-liw4qzol"
    profile_steps: list[str] = []
    # Same clamp `_capped_generate` applies on the current eval path.
    max_new_tokens: int = 1024


# --- GPU utilization sampling ----------------------------------------------


class GpuSampler:
    """`nvidia-smi -lms 200` in the background; samples stamped with this
    process's perf_counter clock so they line up with generation timers."""

    def __init__(self):
        import subprocess
        import threading

        self.samples: list[tuple[float, float, float]] = []  # (t, util %, mem MiB)
        self.proc = subprocess.Popen(
            [
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits", "-lms", "200",
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


class _IncrementalNoRepeatNGram:
    """Drop-in for HF's NoRepeatNGramLogitsProcessor (batch 1), backed by
    `NoRepeatNGramBlocker`'s incremental n-gram index."""

    def __init__(self, ngram_size: int):
        self.ngram_size = ngram_size
        self.blocker: NoRepeatNGramBlocker | None = None

    def __call__(self, input_ids, scores):
        if self.blocker is None:
            self.blocker = NoRepeatNGramBlocker(self.ngram_size, input_ids[0].tolist())
        else:
            for token in input_ids[0, len(self.blocker.tokens) :].tolist():
                self.blocker.append(token)
        banned = self.blocker.banned()
        if banned:
            scores[0, banned] = -float("inf")
        return scores


class _State:
    """What the generate() wrapper should do; mutated by the optimization steps."""

    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens
        self.fast_ngram = False
        self.graph_decoder: CudaGraphDecoder | None = None
        self.last: dict = {}


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
        kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", state.max_new_tokens), state.max_new_tokens)
        timer = _FirstTokenTimer()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        if state.graph_decoder is not None:
            output_ids = state.graph_decoder.generate(input_ids, timer=timer, **kwargs)
        else:
            processors = []
            if state.fast_ngram:
                ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
                if ngram_size:
                    processors.append(_IncrementalNoRepeatNGram(ngram_size))
            processors.append(timer)
            output_ids = original(input_ids, logits_processor=LogitsProcessorList(processors), **kwargs)
        torch.cuda.synchronize()
        state.last = {
            "t_start": t_start,
            "t_first": timer.t_first,
            "t_end": time.perf_counter(),
            "n_output_tokens": int(output_ids.shape[1] - input_ids.shape[1]),
            "n_prompt_tokens": int(input_ids.shape[1]),
        }
        return output_ids

    inner.generate = generate
    try:
        yield
    finally:
        inner.generate = original


# --- optimizations -----------------------------------------------------------


def _moe_one_sync(self, hidden_states, topk_ids, topk_weight):
    """transformers' DeepseekV2MoE.moe with two changes:

    1. `tokens_per_expert.tolist()` once, instead of iterating the CUDA tensor
       (`if num_tokens == 0` on each element = one blocking GPU->CPU sync per
       expert: 64 experts x 11 MoE layers ~ 700 syncs per decoded token).
       Same math, bit-identical outputs.
    2. If `CudaGraphDecoder` stacked the expert weights (`_w_gate_up`/`_w_down`)
       and this is a single token, gather the top-k experts' weights and run
       them as two batched matmuls: no host sync and static shapes, so the
       decode step can be captured in a CUDA graph.
    """
    import torch

    if hidden_states.shape[0] == 1 and getattr(self, "_w_gate_up", None) is not None:
        idx = topk_ids[0]
        x = hidden_states[0][None, :, None].expand(idx.shape[0], -1, -1)
        gate, up = torch.bmm(self._w_gate_up[idx], x)[..., 0].chunk(2, dim=-1)
        act = self.experts[0].act_fn(gate) * up
        out = torch.bmm(self._w_down[idx], act[..., None])[..., 0]  # [k, hidden]
        return (out.type(topk_weight.dtype) * topk_weight[0, :, None]).sum(0, keepdim=True).type(out.dtype)

    cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0).tolist()
    indicies = topk_ids.view(-1).argsort()
    sorted_tokens = hidden_states[indicies // topk_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            continue
        end_idx = start_idx + num_tokens
        outputs.append(self.experts[i + self.ep_rank * self.experts_per_rank](sorted_tokens[start_idx:end_idx]))
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[indicies] = outs
    return (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def _moe_modules(model):
    return [m for m in model.modules() if hasattr(m, "experts") and hasattr(m, "moe")]


def _patch_moe(model) -> int:
    classes = {type(m) for m in _moe_modules(model)}
    for cls in classes:
        cls.moe = _moe_one_sync
    return len(classes)


class CudaGraphDecoder:
    """Greedy decode with a static KV cache and the 1-token step replayed
    from a CUDA graph. Prefill still runs the model's own forward (vision
    encoder + full prompt) eagerly, into the same static cache.

    Decoding semantics match `infer()`'s generate call: greedy, stop at
    eos_token_id (included in the output), max_new_tokens, and the same
    no-repeat-n-gram rule (checked on the CPU each step; the argmax token
    has to come back to the host anyway to check it and for EOS).
    """

    # Prompt is ~1,150 tokens max (257 global + up to 6x144 crop tokens + text)
    # plus max_new_tokens <= 1024; one fixed size = one graph for every page.
    MAX_CACHE_LEN = 2560

    def __init__(self, model):
        import torch
        from transformers import StaticCache

        self.model = model
        self.cache = StaticCache(config=model.config, max_cache_len=self.MAX_CACHE_LEN)
        self.ids = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.pos = torch.zeros((1,), dtype=torch.long, device="cuda")
        self.graph = None
        self.logits = None
        self._stack_expert_weights()

    def _stack_expert_weights(self):
        """Stack each MoE layer's expert weights into [E, 2I, H] / [E, H, I]
        tensors and re-point the expert Linears at views of them (no extra
        memory, prefill keeps using the per-expert modules)."""
        import torch

        for moe in _moe_modules(self.model):
            experts = moe.experts
            w_gate_up = torch.stack(
                [torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0) for e in experts]
            ).contiguous()
            w_down = torch.stack([e.down_proj.weight for e in experts]).contiguous()
            inter = experts[0].gate_proj.weight.shape[0]
            for i, e in enumerate(experts):
                e.gate_proj.weight = torch.nn.Parameter(w_gate_up[i, :inter], requires_grad=False)
                e.up_proj.weight = torch.nn.Parameter(w_gate_up[i, inter:], requires_grad=False)
                e.down_proj.weight = torch.nn.Parameter(w_down[i], requires_grad=False)
            moe._w_gate_up, moe._w_down = w_gate_up, w_down
        torch.cuda.empty_cache()

    def _step(self):
        from transformers import DeepseekV2Model

        lm = self.model.model
        out = DeepseekV2Model.forward(
            lm,
            inputs_embeds=lm.embed_tokens(self.ids),
            position_ids=self.pos[None],
            past_key_values=self.cache,
            cache_position=self.pos,
            use_cache=True,
        )
        return self.model.lm_head(out.last_hidden_state[:, -1]).float()

    def _capture(self):
        import torch

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # also lazily allocates the static cache
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step()
        self.cache.reset()

    def generate(self, input_ids, *, timer, images, images_seq_mask, images_spatial_crop,
                 eos_token_id, max_new_tokens, no_repeat_ngram_size=0, **_ignored):
        import torch

        if self.graph is None:
            # First call happens inside infer()'s autocast/no_grad context, so
            # the graph is captured under the same numerics as eager decode.
            with _stage("capture_cuda_graph", max_cache_len=self.MAX_CACHE_LEN):
                self._capture()
        prompt_len = input_ids.shape[1]
        if prompt_len + max_new_tokens > self.MAX_CACHE_LEN:
            raise ValueError(f"prompt {prompt_len} + {max_new_tokens} new tokens > {self.MAX_CACHE_LEN}")

        self.cache.reset()
        prefill = self.model(
            input_ids=input_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            past_key_values=self.cache,
            position_ids=torch.arange(prompt_len, device=input_ids.device)[None],
            use_cache=True,
        )
        logits = prefill.logits[:, -1]
        timer.mark()

        blocker = NoRepeatNGramBlocker(no_repeat_ngram_size, input_ids[0].tolist()) if no_repeat_ngram_size else None
        tokens: list[int] = []
        for step in range(max_new_tokens):
            if blocker is not None:
                banned = blocker.banned()
                if banned:
                    logits = logits.clone()
                    logits[0, banned] = -float("inf")
            token = int(logits.argmax(dim=-1))
            tokens.append(token)
            if blocker is not None:
                blocker.append(token)
            if token == eos_token_id or step == max_new_tokens - 1:
                break
            self.ids.fill_(token)
            self.pos.fill_(prompt_len + step)
            self.graph.replay()
            logits = self.logits
        new = torch.tensor([tokens], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, new], dim=1)


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


def _infer(inner, tokenizer, pil_image, tmpdir: str):
    from pathlib import Path

    from cartridges.training.collator import FREE_OCR_PROMPT

    train_config = TrainConfig()
    # Same per-page disk round trip as `_run_inference` (infer() only takes a path).
    image_path = Path(tmpdir) / "page.png"
    pil_image.save(image_path)
    return inner.infer(
        tokenizer, prompt=FREE_OCR_PROMPT, image_file=str(image_path), output_path=tmpdir,
        base_size=train_config.base_size, image_size=train_config.image_size,
        crop_mode=train_config.crop_mode, save_results=False, test_compress=False, eval_mode=True,
    )


def _measure(label, inner, tokenizer, pages, state, sampler, reference_hyps) -> list[dict]:
    import tempfile

    import jiwer
    import torch

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir, _instrumented_generate(inner, state):
        with _stage("warmup", step=label):
            _infer(inner, tokenizer, pages["short"]["image"], tmpdir)
        for name, page in pages.items():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hypothesis = _infer(inner, tokenizer, page["image"], tmpdir)
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
                "gpu_util_pct": sampler.util(gen["t_start"], gen["t_end"]),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "cer_vs_reference": jiwer.cer(page["text"], hypothesis),
                "cer_vs_baseline": jiwer.cer(ref_hyp, hypothesis) if ref_hyp is not None else None,
                "matches_baseline": hypothesis == ref_hyp if ref_hyp is not None else None,
                "hypothesis": hypothesis,
            }
            logger.info(
                "%s %s: prefill=%.2fs decode=%s tok/s out=%d wall=%.2fs util=%s match=%s",
                label, name, row["prefill_s"],
                f"{row['decode_tok_s']:.1f}" if row["decode_tok_s"] else "n/a",
                row["output_tokens"], wall, row["gpu_util_pct"], row["matches_baseline"],
            )
            rows.append(row)
    return rows


def _profile(label, inner, tokenizer, page, state, run_name) -> dict:
    """torch.profiler on one page -> gzipped Chrome trace + op tables."""
    import gzip
    import os
    import shutil
    import tempfile

    import torch
    from torch.profiler import ProfilerActivity, profile

    os.makedirs(f"{PROFILING_DIR}/traces", exist_ok=True)
    trace_path = f"{PROFILING_DIR}/traces/{run_name}_{label}.json"
    saved_cap = state.max_new_tokens
    state.max_new_tokens = PROFILE_MAX_NEW_TOKENS
    try:
        with tempfile.TemporaryDirectory() as tmpdir, _instrumented_generate(inner, state):
            _infer(inner, tokenizer, page["image"], tmpdir)  # warm
            with _stage("torch_profiler", step=label), profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, with_stack=True,
            ) as prof:
                _infer(inner, tokenizer, page["image"], tmpdir)
            gen = dict(state.last)
    finally:
        state.max_new_tokens = saved_cap

    prof.export_chrome_trace(trace_path)
    with open(trace_path, "rb") as src, gzip.open(trace_path + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(trace_path)

    averages = prof.key_averages()
    device_total_us = sum(e.self_device_time_total for e in averages)
    counts = {e.key: e.count for e in averages}
    return {
        "step": label,
        "trace": trace_path + ".gz",
        "generate_s": gen["t_end"] - gen["t_start"],
        "prefill_s": gen["t_first"] - gen["t_start"],
        "output_tokens": gen["n_output_tokens"],
        "gpu_kernel_time_s": device_total_us / 1e6,
        "sync_counts": {
            k: counts.get(k, 0)
            for k in ("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize",
                      "cudaMemcpyAsync", "cudaLaunchKernel", "cudaGraphLaunch")
        },
        "top_cuda": averages.table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=60),
        "top_cpu": averages.table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=60),
    }


@app.function(
    gpu="A100",  # same GPU as evaluate()
    timeout=90 * MINUTES,
    volumes={"/cache": model_cache},
    secrets=[hf_secret],
)
def profile_inference(config: ProfileConfig) -> dict:
    import os

    import torch
    from datasets import load_from_disk

    unknown = set(config.steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {unknown}; choose from {STEPS}")

    with _stage("load_dataset", split="test"):
        test = load_from_disk(TrainConfig().dataset_dir)["test"]
        pages = {name: {"index": i, **test[i]} for name, i in PAGES.items()}

    os.makedirs(f"{PROFILING_DIR}/results", exist_ok=True)
    result_path = f"{PROFILING_DIR}/results/{config.run_name}.json"
    result = {"config": config.model_dump(), "gpu": torch.cuda.get_device_name(), "rows": [], "profiles": []}

    def save():
        with open(result_path, "w") as f:
            json.dump(result, f, indent=1)
        model_cache.commit()

    sampler = GpuSampler()
    state = _State(config.max_new_tokens)
    try:
        model, tokenizer = _load(config)
        if config.measure_base:
            result["rows"] += _measure("base/baseline", _inner(model), tokenizer, pages, state, sampler, {})
            save()

        model = _attach_adapter(model, config)
        reference_hyps: dict[str, str] = {}
        for step in config.steps:
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
            inner = _inner(model)
            rows = _measure(step, inner, tokenizer, pages, state, sampler, reference_hyps)
            # Every later step in this run is compared to the first one.
            reference_hyps = reference_hyps or {r["page"]: r["hypothesis"] for r in rows}
            result["rows"] += rows
            save()
            if step in config.profile_steps:
                result["profiles"].append(
                    _profile(step, inner, tokenizer, pages[PROFILE_PAGE], state, config.run_name)
                )
                save()
    finally:
        sampler.stop()
    return result


@app.local_entrypoint()
def run_profile(
    run_name: str,
    loader: str = "unsloth",
    steps: str = "baseline",
    measure_base: bool = False,
    profile_steps: str = "",
    hf_revision: str = "run-liw4qzol",
):
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    config = ProfileConfig(
        run_name=run_name, loader=loader, steps=split(steps), measure_base=measure_base,
        profile_steps=split(profile_steps), hf_revision=hf_revision,
    )
    call = profile_inference.spawn(config)
    print(f"CALL_ID {call.object_id}")


def _poll(call_id: str, out_path: str) -> None:
    """Short reconnects instead of one long blocking .get() (see finetune.py's
    note on dropped long-lived connections)."""
    while True:
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=60)
            break
        except TimeoutError:
            print("still running...", flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "poll":
        _poll(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: python -m cartridges.training.profiling poll <CALL_ID> <out.json>")
