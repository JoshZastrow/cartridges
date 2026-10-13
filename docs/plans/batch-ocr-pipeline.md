# Plan: Batch OCR pipeline for handwritten notes

Status: in progress (branch `feat/batch-ocr-pipeline`).
Writing standard: ASD-STE100 (Simplified Technical English).

## 1. Purpose

This plan gives the steps to:

1. Promote a trained OCR adapter to a verified model.
2. Merge the adapter into the base weights.
3. Serve the merged model with vLLM on Modal.
4. Send all pages of one or more PDFs to the server at high concurrency.
5. Keep the results in Postgres, so that a second run does no GPU work.
6. Write the outputs: one transcript, one PageIndex tree, one teacher context.

An agent or a person starts each step with one CLI command.

## 2. Terms

| Term | Meaning |
|---|---|
| Adapter | The LoRA weights that training pushes to the HF Hub repo. |
| Revision | A git ref in the HF Hub repo: a tag (`run-*`, `date-*`, `verified-*`, `prod`) or a commit SHA. |
| Merged model | Base weights plus adapter, in one directory, on the Modal volume. |
| `file_key` | BLAKE2b hash of the PDF bytes. |
| `page_key` | BLAKE2b hash of `file_key`, page number, and DPI. |
| `model_key` | BLAKE2b hash of the served model SHA and all decoding settings. |
| Flagged page | A page whose text ends in a repetition loop. |

## 3. Design decisions

### 3.1 vLLM is the inference engine. Modal is the compute platform.

- Modal gives the GPU containers, the autoscaler, the volume, and the HTTPS URL.
- vLLM runs the model. It batches many pages on one GPU (continuous batching).
- vLLM 0.29 has the upstream architecture `DeepseekOCR2ForCausalLM`.
- Use the logits processor `NGramPerReqLogitsProcessor`. Use the official values:
  `ngram_size=30`, `window_size=90`, whitelist `<td>` and `</td>`.
- Do not use the HF `model.infer()` path for batch work. It does one page at a time.

### 3.2 Merge the adapter before you serve it.

- The adapter targets `q/k/v/o/gate/up/down_proj`. These names can also occur in the vision encoder.
- vLLM LoRA support for vision layers is not certain. A merged model removes this risk.
- The merge is a pure tensor operation on CPU: `W += (B @ A) * (alpha / r)`.
- The merge does not load unsloth. It reads and writes safetensors only.
- The merge stops with an error if an adapter module has no base weight.

### 3.3 One CLI process is the orchestrator. Do not use Prefect for this path.

- No orchestration server is necessary.
- The pipeline uses `asyncio` and a process pool. These are in the standard library.
- The new runtime dependency is `httpx` (it is already installed through `modal`).

### 3.4 Pipeline shape

```
PDF files
  -> file_key (metadata fast path, else BLAKE2b of the bytes)
  -> page_keys -> bulk cache lookup in Postgres
  -> job queue (cache misses only)
  -> R render tasks (process pool, PyMuPDF)          CPU stage
  -> bounded PNG queue (maxsize = queue_size)         backpressure
  -> N OCR workers (async HTTP to vLLM)               GPU stage
  -> loop check -> one Postgres row per page          crash-safe
  -> output writers (read all rows, write files)
```

- N workers set the number of requests in flight. Set N to about
  `containers x max_num_seqs`.
- The bounded queue stops the render stage when the GPU stage is slow.
  Thus memory stays bounded.
- Each worker writes its row when its page is complete. If the run stops,
  you lose only the pages in flight. The next run skips all saved pages.
- The pipeline does not keep errors in the cache. It keeps flagged pages, with
  `status = 'looped'`. Use `--retry_flagged` to send them again.

### 3.5 Cache keys

- `file_key`: if path, size, and `mtime_ns` agree with the `source_files` row,
  use the saved `file_key`. If not, hash the file again.
- `page_key = blake2b(file_key | page_number | dpi)`.
- `model_key = blake2b(served_model_sha | prompt | max_tokens | ngram settings)`.
- The pipeline gets `served_model_sha` from the server (`/v1/models`). Thus a
  cache row always agrees with the model that made it.
- The old table `page_transcriptions` stays. The old flow continues to use it.

### 3.6 Warm containers

- A container cold start occurs one time for each container. A warm container
  serves many requests at the same time.
- `scaledown_window` is 10 minutes (Modal maximum is 20 minutes).
- At the start of a run, the CLI sets `min_containers = K`. All K containers
  start in parallel while the CPU stage runs.
- At the end of a run, the CLI sets `min_containers = 0` (in a `finally` block).
- `cartridges serve_warm --containers K` keeps K containers warm on purpose.
  `cartridges serve_cool` sets the minimum back to 0.
- Follow-up: GPU memory snapshots, to make each cold start shorter.

### 3.7 Outputs

For each PDF, in `<out>/<pdf_stem>/`:

| File | Content |
|---|---|
| `transcript.md` | Golden format: `## p. N - date: D`, sections divided by `---`. |
| `pageindex.md` | Heading tree: `# <doc>`, then `## p. N - date: D`, then the page text. |
| `pageindex_tree.json` | PageIndex schema: `doc_name`, `line_count`, `structure[]`. Each node has `title`, `node_id` (4 digits), `line_num`, `start_index`, `end_index`, `text`, `nodes`. |
| `manifest.json` | `file_key`, `model_sha`, `model_key`, and the status of each page. |

For all PDFs, in `<out>/`:

| File | Content |
|---|---|
| `teacher_context.md` | All transcripts in one file, with `[doc:page]` anchors and a token estimate. Budget default: 1,000,000 tokens. |

PageIndex PDF mode needs a text layer. Scanned handwriting has no text layer.
Thus we write the markdown tree ourselves, in the PageIndex schema. PageIndex
`md_to_tree` can add LLM summaries later. This is optional.

## 3.8 The packaged OCR service (recipe)

- `serve/recipe.py` `OcrRecipe` holds all settings: merged weights SHA,
  preprocessing (1024 / 768 / crop), prompt, decoding, and postprocessing.
- The model id is the recipe hash. The cache key, the eval report, and the
  deploy record use this id.
- The server (`serve/vllm_server.py`) has `POST /ocr` (image in, text out),
  `GET /info`, and `GET /health`. vLLM listens on an internal port only.
- The eval (`model_eval`) and the pipeline (`notes_process`) both call `/ocr`.
  Thus the eval measures the model that production uses.
- Preprocessing check (2026-10-09): on all 78 test pages, the tile grid and the
  global and tile pixel tensors of vLLM 0.29.0 are bit-identical to the
  training collator. The prompt token ids on the live server are identical.

## 4. Commands

```
cartridges model_eval    [--revision run-<id>] [--split test] [--out eval_report.json]
cartridges model_parity  --revision run-<id> [--hf_call_id fc-...]
cartridges model_promote --revision run-<id> [--max_cer 0.20] [--max_loop_rate 0.02] [--dry_run]
cartridges model_merge   --revision prod
cartridges serve_deploy  --revision prod [--max_containers 8] [--ngram_size 10] [--repetition_penalty 1.05]
cartridges ocr           FILE.png|FILE.pdf [--page_number 1] [--revision prod] [--dpi 250]
cartridges serve_warm    --containers 4
cartridges serve_cool
cartridges serve_status
cartridges notes_process PDF... [--out notes_out] [--concurrency 64] [--containers 0]
                         [--dpi 250] [--dry_run] [--retry_flagged]
cartridges notes_collate [--out notes_out] [--budget 1000000]
```

`notes_process` writes one JSON summary to stdout. Logs go to stderr.

| Exit code | Meaning |
|---|---|
| 0 | All pages are OK. |
| 1 | Fatal error. The run did not complete. |
| 2 | The run completed, but one or more pages are flagged or failed. |

## 5. Phases

### Phase 0: Numeric parity (go / no-go)

Purpose: show that vLLM gives the same results as the training eval path
(`finetune._run_inference`: unsloth + PEFT, adapter not merged, `model.infer`).

Command: `cartridges model_parity --revision <rev>` (file `serve/parity.py`).

1. Deploy the revision: `serve_deploy --revision <rev>`. This also tests the merge.
2. The command sends the test-split images (the same PNG bytes that eval uses)
   to vLLM. At the same time, it runs `evaluate()` on Modal.
3. The vLLM request uses the eval settings, not the production settings:
   plain prompt `"<image>\nFree OCR. "`, `no_repeat_ngram` 35 over the full
   output, no whitelist, `max_tokens` = `TrainConfig.max_new_tokens`.
4. The report gives, for each example: CER against the reference for each path,
   CER of vLLM against HF, exact match, and the first character that differs.

Pass condition:
- The mean CER of vLLM is within 1 point of the mean CER of HF.
- The mean CER of vLLM against HF is less than 2%.

Bit-for-bit equality is not expected (different kernels, merged bf16 weights).

If parity fails, examine `first_diff` and the prompt. If it still fails, use
the fallback: HF batched `generate()` with `@modal.batched`.

Result (2026-10-13): parity fails, and the cause is on the HF side. In
transformers 4.56.2 the HF Qwen2 vision encoder runs with a plain causal mask;
vLLM runs the design mask. vLLM is the production path. See
`docs/hf-vllm-parity.md`.

Throughput (after parity passes): run `notes_process` at concurrency 8, 32,
and 64. Read `pages/s` in the log.

### Phase 1: Model registry and promotion

Files: `src/cartridges/serve/registry.py`, `src/cartridges/serve/merge.py`.

1. `resolve_revision(alias)` gives the commit SHA.
2. `check_promotion(predictions, max_cer, max_loop_rate)` gives pass or fail.
3. `promote` runs the existing `evaluate`, checks the thresholds, then makes the
   tag `verified-<date>-<sha7>` and moves the tag `prod`.
4. `merge` writes `/cache/models/<sha>/` on the volume `cartridges-training-cache`.

Verify: unit tests for `check_promotion` and the merge key map.

### Phase 2: Inference server

File: `src/cartridges/serve/vllm_server.py`.

1. One Modal function `serve` runs `vllm serve` behind `@modal.web_server`.
2. The model SHA goes into the image environment at deploy time. A new SHA makes
   new containers.
3. `--served-model-name` is the SHA. The client reads it from `/v1/models`.
4. `VLLM_API_KEY` comes from the Modal secret `cartridges-ocr-api-key`.

Verify: `serve_status` shows the URL and the served SHA.

### Phase 3: Cache v2

File: `src/cartridges/storage/cache.py` (add, do not change the old functions).

1. Tables `ocr_pages` and `source_files`.
2. Async store with bulk read and one upsert for each page.

Verify: key tests with no database. One integration test that runs only if
`DATABASE_URL` is set.

### Phase 4: Batch pipeline

Files: `src/cartridges/batch/keys.py`, `backends.py`, `pipeline.py`.

Verify: tests with a fake backend and an in-memory store:
- The second run sends no requests.
- A retry after a failure succeeds.
- The pipeline does not keep errors.
- Flagged pages are sent again only with `retry_flagged`.

### Phase 5: Output writers

File: `src/cartridges/batch/outputs.py`.

Verify: golden tests on a 3-page fixture. The tree nodes agree with the page numbers.

### Phase 6: CLI

File: `src/cartridges/cli.py` (add methods).

Verify: `notes_process <fixture> --dry_run` gives valid JSON.

### Phase status

| Phase | Code | Unit tests | Live check on Modal |
|---|---|---|---|
| 0 Spike | n/a | n/a | Not done. Do this first. |
| 1 Registry and promotion | Done | Done | Not done |
| 2 Inference server | Done | Command only | Not done |
| 3 Cache v2 | Done | Gated test (`CARTRIDGES_TEST_DATABASE_URL`) | Not done |
| 4 Batch pipeline | Done | Done (fake backend, in-memory store) | Not done |
| 5 Output writers | Done | Done | n/a |
| 6 CLI | Done | `--help` only | Not done |

## 5a. Setup (one time)

1. Make the API key secret. Use the same key on your computer:
   ```
   modal secret create cartridges-ocr-api-key VLLM_API_KEY=<key>
   export VLLM_API_KEY=<key>
   ```
2. Make sure that the base model is on the volume `cartridges-training-cache`
   at `/cache/deepseek_ocr` (training `download_model` puts it there).
3. Set `DATABASE_URL` (Postgres). The pipeline makes its tables if necessary.
4. Run the commands from the main checkout. `model_promote` imports the
   training app, and that app needs `data/train/hf_dataset`.

## 5b. Phase 0 procedure

Note: the adapter of W&B run `divine-energy-17` (`vrp2edvi`) does not exist.
The run is older than the HF Hub repo, and it saved no model artifact. The
only trained adapter on the Hub is `run-liw4qzol` (`dry-fire-27`).

1. `cartridges serve_deploy --revision <rev>`
2. `cartridges serve_warm --containers 1`
3. `cartridges model_parity --revision <rev>`
4. Read the summary. Examine `parity_report.json` for each example.
5. `cartridges serve_cool`

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| The merged weights do not load in vLLM (key names). | Medium | The merge keeps the base key names. Phase 0 spike. |
| vLLM output is different from `model.infer()`. | Medium | Phase 0 CER check. |
| Repetition loops make requests slow. | Medium | `max_tokens` cap, n-gram processor, flag and retry. |
| Upload bandwidth limits throughput (large PNGs). | Medium | Default DPI 250. Measure in Phase 0. |
| Cold start of each container (1 to 3 minutes). | Medium | Pre-warm K containers. Snapshots later. |
| Postgres round trips are slow. | Low | Bulk read. One short write for each page. |

## 7. Out of scope

- The selection of the teacher model.
- Search over the PageIndex tree. An agent can read `pageindex_tree.json` directly.
- Date extraction in the new pipeline. Dates show as `unknown` until a later phase.
