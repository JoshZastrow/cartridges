# Plan: bring OCR inference to modern tokens/second

**Status:** in progress (started 2026-10-08), worked by a background agent in a git worktree
**Related:** `docs/plans/loop-elimination.md` (runs in parallel on `main`; don't conflict with it)

## Problem

Fine-tuned DeepSeek-OCR-2 transcription on a Modal A100 takes ~25–30 s for a typical page
(median reference length 250 tokens, p90 639), i.e. roughly **8–10 output tokens/s**
including preprocessing and the ~1,300-token image prefill. The decoder is a small-active-parameter
MoE; single-stream decode on an A100 should be several times faster. Nothing has been profiled:
the only timings are per-page wall clock (`_run_inference`) and coarse `_stage` logs.

## Current inference path

`training/finetune.py`:
- `_load_base_model`: `unsloth.FastVisionModel.from_pretrained(..., use_gradient_checkpointing="unsloth", unsloth_force_compile=True)`
- `_load_artifact_model`: wraps it in `peft.PeftModel.from_pretrained` (adapter **unmerged**)
- `_run_inference`: `FastVisionModel.for_inference(model)`, then the model's own `model.infer(..., eval_mode=True)`
  per page, with `_capped_generate` clamping `max_new_tokens` to 1024
- `model.infer` (remote code, `/cache/deepseek_ocr/modeling_deepseekocr2.py:716`) calls `generate` with
  greedy decoding, `no_repeat_ngram_size=35`, `eos_token_id=1`

`serve/inference.py` reuses `_load_base_model` and `_capped_generate`.

## Suspects (unverified)

1. `no_repeat_ngram_size=35`: Hugging Face's `NoRepeatNGramLogitsProcessor` is Python-side and rescans the
   full sequence (prompt + generated) every step, so the cost grows roughly with length squared.
2. Unmerged LoRA: extra adapter matmuls on every projection at every step.
3. Training-mode load (Unsloth gradient checkpointing, compile patches) carried into inference.
4. MoE expert routing in eager PyTorch (Python loop over experts).
5. Per-page overhead: image written to disk and re-read, preprocessing on CPU.

## Phases

### 1. Measure (2–3 pages)
- Pick 3 fixed pages from `data/train/hf_dataset` test split (short / median / long reference).
- Report per page: prefill time (time to first token), decode tokens/s, total wall, output token count,
  peak GPU memory. Use CUDA-synchronized timers, warm-up run first.
- **Check:** a table of baseline numbers for the current path (base model and adapter `run-liw4qzol`).

### 2. Profile
- `torch.profiler` (CPU + CUDA, `record_shapes`, `with_stack`) on one page → Chrome trace JSON,
  viewable as a flame chart in Perfetto (ui.perfetto.dev). Save traces locally.
- GPU utilization: sample `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 200`
  (or pynvml) during generation.
- **Check:** a breakdown of where decode time goes (top ops, CPU vs GPU time, idle gaps) and average GPU utilization.

### 3. Optimize (one change at a time, re-measure after each)
Candidates, cheapest first: drop or replace `no_repeat_ngram_size`; merge LoRA (`merge_and_unload`);
load for inference without Unsloth training patches (plain `transformers`, bf16, SDPA/flash attention);
static KV cache + `torch.compile`; batch pages; a dedicated engine (vLLM supports DeepSeek-OCR models;
check whether DeepSeek-OCR-2 is supported, and how to serve the LoRA: merged weights or native LoRA).

### 4. Verify
- Same 3 pages: outputs must match the baseline path (identical text, or CER within 0.5 points of the baseline
  output on each page). A speedup that changes transcriptions doesn't count.

## Target ("on par with modern inference")

- **Single-stream decode ≥ 100 output tokens/s** on an A100, adapter applied, at batch 1
  (excluding prefill), and a typical page (250 tokens) end-to-end in **≤ 5 s** including prefill.
- If a batched engine is used, also report batched throughput (pages/min).

## Constraints

- Work in a git worktree on a branch; keep changes in new files where possible
  (`src/cartridges/serve/` or a new `src/cartridges/training/profiling.py`). `finetune.py`, `config.py` and
  `collator.py` have uncommitted edits on `main` for the looping work.
- Modal: shared volume `cartridges-training-cache` is read-only for this work except under `/cache/profiling/`.
- Budget: ≤ $25 of Modal GPU time. Stop and report if blocked or over budget.
- Don't push to the Hugging Face repo or modify the W&B project's existing runs.

## Results log

| Step | Change | Prefill (s) | Decode tok/s | Page wall (s) | GPU util | Output match |
|---|---|---|---|---|---|---|
