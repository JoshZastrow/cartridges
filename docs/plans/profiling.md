# Spec: single-page profiler (evaluation, forward pass, training step)

**Status:** approved 2026-10-08. Implementation in progress.
**Owners:** steps 1–14 (`generate`, `forward` modes): inference-speed agent, in its worktree.
Step 15 (`train-step` mode): `main`, file `src/cartridges/training/profiling_train.py`.
**Related:** `docs/plans/inference-speed.md`, `docs/plans/loop-elimination.md`.

## 1. Problem

| Quantity | Value |
|---|---|
| Decode speed, fine-tuned model (HF path) | 9.2 tokens/s, about 110 ms for each token |
| Output tokens for each page (median) | 252 |
| Time for each page | 252 × 110 ms ≈ 28 s |
| Time for 78 pages, one page at a time | ≈ 36 min |
| vLLM, 78 pages | 23 s, about 0.3 s for each page |

The GPU work for one token is about 1 ms. About 109 ms for each token is outside the GPU
work. The profiler must find these 109 ms. The profiler examines one page in one pass.

## 2. Spec

### 2.1 Purpose
The profiler records one pass of the model on one page. It shows the time for each stage.
It shows when the GPU is busy and when it is idle. It shows which CPU functions use the time.

### 2.2 Modes
| Mode | What the profiler records | Use |
|---|---|---|
| `generate` | One `model.infer()` call on one page: preprocessing, vision encoder, prefill, all decode steps | Find the 109 ms for each token |
| `forward` | One forward pass on the prompt only (image + prompt), with no decoding | Measure the vision encoder and prefill in isolation |
| `train-step` | One forward pass, one backward pass and one optimizer step on one batch | Measure training cost (batch size 1 and 2) |

### 2.3 Inputs
| Input | Type | Default | Description |
|---|---|---|---|
| `mode` | `generate` / `forward` / `train-step` | `generate` | Pass type |
| `split`, `index` | str, int | `test`, `0` (`train`, `0` for `train-step`) | The page in `data/train/hf_dataset` |
| `model_source` | `base` / `artifact` | `base` | Base model, or base model + adapter |
| `hf_revision` | str | none | Adapter revision, for example `run-liw4qzol`. Required for `artifact`. |
| `merge_adapter` | bool | `false` | Merge the adapter before the pass |
| `max_new_tokens` | int | `64` | Token limit for the recorded pass |
| `decoding` | `DecodingConfig` | none | Overrides for `no_repeat_ngram_size`, `repetition_penalty`, `eos_token_id` |
| `batch_size` | int | `1` | `train-step` only |
| `with_stack` | bool | `true` | Record Python call stacks (more overhead) |
| `run_name` | str | required | Name of the output folder and the W&B run |

### 2.4 Components that the profiler integrates with
The profiler does not change the code of the evaluation or training paths. It calls the same
functions and attaches to them from the outside.

| Component | File | Integration |
|---|---|---|
| Model load | `finetune.py::_load_base_model`, `_load_artifact_model`, `_load_base_and_lora` | Called without changes |
| Inference call | DeepSeek `model.infer()` (remote code) | Called with the same arguments as `_run_inference` |
| Generate loop | `finetune.py::_capped_generate` | Wrap `model.generate`; record the time of each decode step |
| Logits processors | HF `generate` (includes `no_repeat_ngram_size`) | Wrap each processor; record its CPU time |
| Model parts | Vision encoder, projector, language model layers | Forward hooks add a named label to each part in the trace |
| Training step | `DeepSeekOCRDataCollator`, 8-bit AdamW | Build one batch; run forward, backward and optimizer steps |
| GPU sensor | `nvidia-smi` (or pynvml) | Background process records GPU use and memory every 50 ms |

### 2.5 Outputs
Location: `/cache/profiling/<run_name>/` on the Modal volume `cartridges-training-cache`.
Copy to `profiles/<run_name>/` locally. Git ignores `profiles/`.

| File | Format | Contents | Viewer |
|---|---|---|---|
| `trace.json.gz` | Chrome trace | CPU and GPU timeline with stage labels | ui.perfetto.dev |
| `stacks_cpu.folded`, `stacks_gpu.folded` | Collapsed stacks (flame graph input) | CPU / GPU time by call stack | speedscope.app (or flamegraph.pl → SVG) |
| `stages.json` | JSON | Time for each stage; per decode step; time to first token; kernel launches and CPU-GPU syncs per step | Text editor, report |
| `ops.txt` | Text table | Top 30 operations by CPU time and by GPU time | Text editor |
| `gpu.csv` | CSV | GPU use (%) and memory (MB), every 50 ms | Report |
| `memory.pickle` | PyTorch memory snapshot | Memory allocations over time (`train-step` only) | pytorch.org/memory_viz |
| `output.txt` | Text | Transcription from the recorded pass (`generate` only) | Text editor |
| `report.html` | HTML | One page with all of the above | Web browser |

W&B gets the summary numbers from `stages.json`: group `infra-profiling`, run name `<run_name>`.
W&B does not get trace files or page text.

## 3. Procedure: run the profiler

1. Select one page. Use `test/0` for the first run.
2. Run the profiler (`generate` mode):
   ```bash
   uv run modal run -m cartridges.training.profiling::run_profile \
       --mode generate --split test --index 0 \
       --model-source artifact --hf-revision run-liw4qzol \
       --run-name prof-eval-ft-test0
   ```
   For a training step:
   ```bash
   uv run modal run -m cartridges.training.profiling_train::run_profile_train \
       --run-name prof-train-bs1 --batch-size 1
   ```
3. Wait for the command to finish. Expect less than 5 minutes.
4. Copy the results to your computer:
   ```bash
   uv run modal volume get cartridges-training-cache profiling/<run_name> profiles/
   ```
5. Do steps 2 to 4 again with `--model-source base`, to compare the two models.

## 4. Procedure: review the results

1. Open `profiles/<run_name>/report.html` in a web browser.
2. Read the stage bar. Find the stage with the longest time.
3. Read the decode step chart. Find the time for each token. Compare it with 1 ms.
4. Read the GPU use line. Find the periods when the GPU is idle.
5. Open `trace.json.gz` in ui.perfetto.dev. Find the GPU row. Measure the gaps between kernels in one decode step.
6. Open `stacks_cpu.folded` in speedscope.app. Find the widest blocks. These functions use the most CPU time.
7. Compare `output.txt` with the transcription from the evaluation run. The two texts must be the same.
8. For `train-step`: open `memory.pickle` in pytorch.org/memory_viz. Find the peak and the tensors at the peak.

## 5. Implementation steps

| Step | Task | Owner |
|---|---|---|
| 1 | Page loader: read one row from `hf_dataset` by split and index | Speed agent |
| 2 | Modal function `run_profile` in `src/cartridges/training/profiling.py`, app `cartridges-profiling`, A100 | Speed agent |
| 3 | Load the model with the existing functions; add the merge option | Speed agent |
| 4 | One warm-up pass on the same page, 8 tokens, not recorded | Speed agent |
| 5 | Forward hooks with `record_function` labels on vision encoder, projector, language model layers | Speed agent |
| 6 | Wrap `model.generate`; record time of first token and each decode step | Speed agent |
| 7 | Wrap each logits processor; record its CPU time per step | Speed agent |
| 8 | GPU sensor at 50 ms | Speed agent |
| 9 | `torch.profiler.profile` (CPU + CUDA) around the pass | Speed agent |
| 10 | Count kernel launches and CPU-GPU syncs per decode step from trace events | Speed agent |
| 11 | Write output files (2.5) to the volume | Speed agent |
| 12 | Fetch command (copy to `profiles/`) | Speed agent |
| 13 | Report generator: `report.html` from the output files | Speed agent |
| 14 | `forward` mode | Speed agent |
| 15 | `train-step` mode in `profiling_train.py`, with memory history | `main` |
| 16 | Tests for the pure Python parts | Each owner |

## 6. Acceptance criteria

- The profiler runs on one page in less than 5 minutes.
- `output.txt` is the same as the text from an unprofiled run.
- The sum of the stage times is within 10% of the wall time.
- `report.html` identifies where most of the 110 ms for each token goes.

## 7. Training is batched

`TrainConfig`: `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`. One forward
and backward pass has 2 pages. One optimizer step has 4 passes = 8 pages. Unsloth gradient
checkpointing recomputes the forward pass during the backward pass. The collator runs in 2
dataloader worker processes; `torch.profiler` does not see them.
`train-step` mode records one batch (batch size 1 or 2) in isolation. A later option records a
real `Trainer` step with a `TrainerCallback` that calls `prof.step()` in `on_step_end`.
