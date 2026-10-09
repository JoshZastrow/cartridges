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

A100-40GB, batch 1, adapter `run-liw4qzol` unless noted. Numbers are for the **median page**
(test idx 19, 249-token reference); decode tok/s was within ±2% on all 3 pages for every step.
GPU util is the mean of `nvidia-smi -lms 200` samples during generate(). "Output match" is
CER(current-path output, step output) per page as short / median / long (test idx 25 / 19 / 48),
where the current path is `unsloth_v1` step `baseline`; 0 means identical text.
Raw JSON: `profiles/*.json`, which are not committed. Regenerate with the commands in `profiling.py`.

| Step | Change | Prefill (s) | Decode tok/s | Page wall (s) | GPU util | Output match (CER pts s/m/l) |
|---|---|---|---|---|---|---|
| 0 | base model, no adapter (Unsloth load) | 0.41 | 14.2 | 29.1 | 19% | n/a (different model) |
| 1 | **current path**: Unsloth load, unmerged LoRA, `infer()`, HF generate | 1.39 | 7.2 | 34.7 | 15% | reference |
| 2 | + incremental no-repeat-ngram (exact equivalent) | 1.58 | 6.3 | 39.8 | 13% | 0 / 0 / 0 |
| 3 | + merge LoRA (`merge_and_unload`) | 0.50 | 12.6 | 19.7 | 17% | 6.4 / 0.2 / 85.6 ✗ |
| 4 | + MoE loop: 1 sync per layer, not per expert (after merge) | 0.39 | 22.0 | 11.5 | 19% | 6.4 / 0.2 / 85.6 ✗ |
| 5 | MoE sync fix, LoRA **unmerged** (instead of 3–4) | 1.24 | 9.8 | 25.9 | 16% | 0 / 0 / 0 |
| 6 | + static KV cache + CUDA-graph decode step (per-expert ops, LoRA unmerged) | 0.67 | 84.0 | 3.77 | 78% | 0 / 0 / 72.6 |
| 7 | + routed experts as batched matmuls inside the graph | 0.79 | **124.7** | **2.96** | 66% | 0 / 0.2 / 79.4 |
| P1 | plain `transformers` bf16/SDPA load, otherwise = step 1 | 0.82 | 9.0 | 27.3 | 20% | 0 / 1.7 / 61.4 ✗ |
| P2 | plain load + merge + MoE fix + CUDA graph (merged, bmm experts) | 0.21 | 205.5 | 1.70 | 76% | 0 / 4.1 / 58.1 ✗ |
| P3 | plain load + CUDA graph, LoRA unmerged | 0.91 | 79.1 | 4.23 | 72% | 0 / 0.7 / 63.7 |

The long page (idx 48) loops on every variant, the current path included: 723 output tokens for a
611-token reference, CER 0.74 vs reference, repeating "We're in the same emotional space."
Any bf16-level numeric difference sends it into a different loop. The current path is also not
reproducible across containers. The first measured pass after an Unsloth load gave 6.4 / 5.0 / 0
CER points on short / median / long vs `unsloth_v1`'s baseline (runs `unsloth_v2` and `unsloth_v3`,
step 1 of each). Later passes in the same container match it exactly.

Profile (step 1, 48 decode steps, `torch.profiler`): 40,933 `aten::item` host syncs
(about 850 per token: `DeepseekV2MoE.moe` iterates a CUDA tensor and evaluates `if num_tokens == 0`
for each of 64 experts × 11 MoE layers), 175k kernel launches (about 3,600 per token),
GPU busy about 5% of the decode window, and 92% of the window spent in gaps longer than 20 µs.
CPU time is dominated by `cudaLaunchKernel`, `aten::mm/addmm` dispatch (unmerged-LoRA matmuls),
`cudaMemcpyAsync`/`cudaStreamSynchronize`, and `aten::eq`. GPU kernel time is about 2.0 s of the
19 s generate() under the profiler. Without the profiler, nvidia-smi shows 13–21% utilization.
Decode is launch- and sync-bound, not compute- or bandwidth-bound.

### Full test split (78 pages), run `unsloth_all`

All variants use the Unsloth load with the unmerged adapter. The first eager pass is a throwaway
because of the post-load drift described above. "Clean" means the 66 pages that don't loop
(not flagged as looped in the loop-eval baseline and not hitting the 1024-token cap).

| Variant | Identical to eager pass 2 | Clean pages within 0.5 CER pts | CER vs truth, clean / all | Decode tok/s | Median page (s) | 78 pages (min) |
|---|---|---|---|---|---|---|
| eager, pass 1 (moe_sync, same math as current path) | 76/78 | 64/66 | 0.261 / 6.82 | 14.9 | 17.8 | 31.9 |
| eager, pass 2 (reference) | 78/78 | 66/66 | 0.262 / 6.82 | 14.8 | 17.6 | 32.1 |
| CUDA graph, per-expert ops | 20/78 | 26/66 | 0.310 / 10.42 | 83.3 | 3.9 | 6.4 |
| CUDA graph, batched experts | 18/78 | 30/66 | 0.269 / 6.91 | 125.7 | 2.9 | 4.4 |
| coordinator's independent current-path run (`baseline-ft-liw4qzol`) vs eager pass 2 | 30/78 | 31/66 | n/a | n/a | n/a | n/a |

Two conclusions:
- Greedy outputs of this model and adapter differ across containers by about as much as the fast path
  differs from eager. Across containers, eager vs eager agrees on 30/78 pages, with a 10-point mean CER gap on clean pages.
- Even so, the per-page 0.5-point gate is **not** met at corpus level by either graph variant.
  Batched experts keep corpus quality close: clean CER 0.269 vs 0.262. The per-expert graph path is
  worse: clean CER 0.310. That gap is unexplained and needs investigating before it is used.

### Single-page profiler, test/0, 64 new tokens (`profiles/prof-eval-{base,ft}-test0/`)

| | base | adapter run-liw4qzol (unmerged) |
|---|---|---|
| decode tok/s, no profiler / profiler on | 14.1 / 7.0 | 7.6 / 3.9 |
| decode step wall, median, profiler on | 144 ms | 252 ms |
| time to first token (profiler on) | 0.68 s | 2.08 s |
| `DeepseekV2MoE` share of decode step wall | ~73% (104 ms) | ~82% (206 ms) |
| GPU busy during decode (kernel time / wall) | 7.5% | 6.4% |
| kernel launches per step | 1,936 | 3,346 |
| `.item()` host syncs per step | 826 | 826 |
| `cudaStreamSynchronize` per step | 829 | 829 |
| DtoH copies per step | 828 | 828 |
| NoRepeatNGram CPU per step | 6.1 ms | 5.7 ms |
| nvidia-smi util (50 ms samples) | 12% | 10% |
| stage labels cover wall time | 99.96% | 99.98% |
| output.txt == unprofiled run | yes | yes |
| profiler job time (acceptance: < 5 min) | 5 min 5 s | 28 min ✗ (`export_stacks` on the adapter's stack pass) |
