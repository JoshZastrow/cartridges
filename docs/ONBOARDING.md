# Onboarding Guide: cartridges

Generated from the Understand-Anything knowledge graph at
`.understand-anything/knowledge-graph.json` (892 nodes, 1710 edges, 9 layers,
10 tour steps; analyzed at commit `d328753`).

## 1. Project Overview

**Name:** cartridges

**Description:** Training and serving platform for journal cartridges: OCR
fine-tuning of handwritten journals, a Task Store of questions, a vLLM Teacher
on Modal that labels them, distillation of the Teacher into a trainable
KV-cache cartridge, and a LibreChat gateway that serves Teacher or cartridge.

**Languages:** Python (primary), TypeScript, JavaScript, shell, Makefile, YAML,
TOML, JSON, HTML, Markdown.

**Frameworks:** FastAPI, Pydantic, Uvicorn, Pytest, Prefect, Modal, Playwright,
Marimo, PyTorch, Transformers, Weights & Biases, Weave, GitHub Actions.

**End to end.** Scanned handwritten journal PDFs are transcribed by a
LoRA-fine-tuned vision-language OCR model served on Modal behind vLLM
(`src/cartridges/training/`, `src/cartridges/serve/`). The batch pipeline
(`src/cartridges/batch/`) caches every page in Postgres, dates the pages, and
merges all journals into one chronological `timeline.md` and a
`teacher_context.md` with stable page anchors. Questions about the journals
live in the Postgres Task Store (`src/cartridges/cartridge/tasks/`), fed by
the author's own seed questions, mined ChatGPT history, psychology test items,
and a Synthesizer that drafts new tasks. The Teacher, a vLLM-served Qwen model
on Modal with the entire journal context in its prompt
(`src/cartridges/cartridge/teacher.py`), answers sampled tasks and writes
top-k logprob targets to Parquet. Stage 1 distills those targets into a
trainable KV-cache cartridge for a frozen base model
(`src/cartridges/cartridge/cache.py`, `train_distill.py`); Stage 2 runs
asynchronous GRPO (`grpo.py`). Finally an OpenAI-compatible FastAPI gateway
(`src/cartridges/serve/gateway.py`) routes LibreChat requests to the Teacher
or to a deployed cartridge server on Modal.

## 2. Architecture Layers

### CLI Entry & Logging (`layer:entry`)

The cartridges CLI entry point, package root, and shared logging configuration
that dispatch into the OCR, Task Store, Teacher, and cartridge subcommands.

- `src/cartridges/cli.py`: Fire-based CLI exposing the whole platform.
- `src/cartridges/logging/config.py`: single place that configures stdlib logging.
- `src/cartridges/logging/__init__.py`: re-exports `configure_logging`.
- `src/cartridges/__init__.py`: package marker with the project docstring.

### OCR Fine-Tuning (`layer:ocr-training`)

Modal-based OCR fine-tuning of vision-language models on handwritten journals:
dataset/collator, encoder-mask patch, training loops, profiling, review
reports, plus the finetune/dataset-split configs and build notebooks.

- `src/cartridges/training/finetune.py`: Modal LoRA fine-tuning app with W&B logging and k-fold evals.
- `src/cartridges/training/config.py`: Pydantic configs for download, train, eval and decoding.
- `src/cartridges/training/collator.py`: DeepSeek-OCR vision collator ported from the Unsloth notebook.
- `src/cartridges/training/dataset.py`: builds the hand-labeled image-to-text dataset with a frozen split.
- `src/cartridges/training/fast_decode.py`: faster single-stream decoding (CUDA graphs, n-gram blocker).
- `configs/finetune/*.yaml` and `configs/dataset/split.json`: experiment configs and the frozen train/test split.

### OCR Serving & Data Pipeline (`layer:ocr-data-pipeline`)

vLLM OCR service, LoRA merge and HF registry, batch OCR pipeline with page
cache and timeline assembly, Prefect datagen flows, and teacher QA dataset
generation from journal text.

- `src/cartridges/serve/vllm_server.py`: packaged OCR service on Modal (`/ocr`, `/info`, `/health`).
- `src/cartridges/serve/recipe.py`: immutable `OcrRecipe` whose hash is the served model id.
- `src/cartridges/serve/registry.py`: HF Hub tag/SHA resolution and prod promotion.
- `src/cartridges/batch/pipeline.py`: PDF to per-page `PageRow` with process pool, async workers, retries.
- `src/cartridges/storage/cache.py`: Postgres page cache and the `source_files` / `ocr_pages` tables.
- `src/cartridges/batch/timeline.py` and `src/cartridges/batch/outputs.py`: timeline merge and `teacher_context.md`.

### Task Store (`layer:task-store`)

Postgres-backed store of questions and labels: seeding, synthesis, QA
generation, sampling, validation, psych/chat loaders, judge/CGC/neighbor
evaluators, export, and the LLM-judge rubrics.

- `src/cartridges/cartridge/tasks/types.py`: frozen Pydantic records and Literal vocabularies (highest fan-in file).
- `src/cartridges/cartridge/tasks/store.py`: the only module that talks to Postgres.
- `src/cartridges/cartridge/tasks/synthesize.py`: the Task Synthesizer.
- `src/cartridges/cartridge/tasks/sampler.py`: pure, deterministic stratified sampling.
- `src/cartridges/cartridge/tasks/label.py`: Teacher labeling stage writing Parquet targets.
- `src/cartridges/cartridge/tasks/cgc.py`: context gap closed, the single-value cartridge metric.

### Cartridge Training & Teacher (`layer:cartridge`)

The vLLM Teacher on Modal that labels tasks, self-study data generation,
KV-cache cartridge model, distillation and GRPO trainers, evals, W&B/Weave
tracking, cartridge serving, and drift analysis scripts.

- `src/cartridges/cartridge/teacher.py`: the Teacher, HF Scorer, and shared `render()`.
- `src/cartridges/cartridge/cache.py`: the `Cartridge` (KV prefix plus DeltaNet states).
- `src/cartridges/cartridge/train_distill.py`: Stage 1 context distillation on Modal.
- `src/cartridges/cartridge/grpo.py`: Stage 2 asynchronous GRPO.
- `src/cartridges/cartridge/evals.py`: eval sets, scorers, grounded reward, base/ICL/cartridge comparisons.
- `src/cartridges/cartridge/tracking.py`: W&B naming, metadata, GPU cost budgets.

### Chat Gateway & LibreChat (`layer:chat-serving`)

OpenAI-compatible FastAPI gateway with Modal backends and chat model registry,
the LibreChat stack lifecycle (Makefile, start/stop, mongo, yaml), and its
Playwright e2e tests and CI workflow.

- `src/cartridges/serve/gateway.py`: `/v1/models` and `/v1/chat/completions`.
- `src/cartridges/serve/backends.py`: Modal backend adapters with timeouts and `TransientBackendError`.
- `src/cartridges/serve/chat_models.py`: registry of exposed chat model ids.
- `librechat/librechat.yaml`, `librechat/start.sh`, `librechat/stop.sh`: LibreChat config and lifecycle.
- `e2e/librechat.spec.ts` and `playwright.config.ts`: Playwright chat test.
- `.github/workflows/playwright.yml`: CI for the e2e suite.

### Test Suite (`layer:tests`)

Pytest unit, integration, and Modal GPU tests covering the OCR pipeline, Task
Store modules, cartridge training, Teacher, gateway, and metrics.

- `tests/test_cartridge_train.py`: distillation loss, cartridge-only gradients, checkpoint parity.
- `tests/test_cartridge_tasks_cli.py`: whole tasks CLI against FakeStore and FakeTeacher.
- `tests/test_cartridge_tasks_label.py`: label validation, citations, shard writing.
- `tests/test_gateway.py`: gateway routing, streaming framing, retries.
- `tests/test_batch_pipeline.py`: cache keys, idempotent reruns, collation.
- `tests/cartridge_cache_modal.py` and `tests/encoder_mask_modal.py`: GPU checks that run on Modal.

### Documentation & Reports (`layer:documentation`)

README, design plans for the batch OCR, cartridge, label-queue and LibreChat
pipelines, data-quality investigations, and the generated cartridge report
with its HTML diagrams and build script.

- `docs/plans/cartridge-pipeline.md`: team contract for distillation followed by async GRPO.
- `docs/plans/task-db.md`: Task Store, Sampler, Synthesizer, labeler and Target Store design.
- `docs/plans/batch-ocr-pipeline.md`: batch OCR design (vLLM on Modal, cache keys, recipe).
- `docs/plans/librechat.md`: LibreChat stack contract (ports, keys, owners, done criteria).
- `docs/report/summary.md` and `docs/report/recipe.md`: the pipeline review and ML system design.
- `docs/report/build_report.py`: renders the report markdown into HTML and PDF.

### Project & Experiment Config (`layer:config`)

Python/Node project manifests, Prime Intellect eval/GEPA/RL experiment configs
per model family, Neon database config, and the docs_to_markdown verifiers
environment package.

- `pyproject.toml`: project definition, dependencies, and the `cartridges` console script.
- `package.json`: Node dev tooling for the LibreChat e2e stack.
- `configs/eval/*.toml`, `configs/gepa/*.toml`, `configs/rl/*.toml`: Prime Intellect experiment configs.
- `environments/docs_to_markdown/docs_to_markdown/taskset.py`: verifiers.v1 taskset over `cartridges.datagen`.
- `neon.ts` and `skills-lock.json`: Neon config and pinned agent skills.

## 3. Key Concepts

- **Fire CLI as the runbook.** `src/cartridges/cli.py` turns the public
  methods of one class into subcommands (`serve_deploy`, `notes_process`,
  `tasks_label`, `cartridge_train`, `gateway_serve`, and so on). Heavy modules
  are imported lazily inside each method, so the static import graph from
  `cli.py` reaches only the logging package.
- **Frozen Pydantic models with Literal vocabularies.** The Task Store types
  (`tasks/types.py`) and the gateway's `ChatModel` and `Generation` records are
  frozen, and kinds, origins, splits and statuses are `Literal` fields. An
  invalid value fails at construction rather than deep inside a training run.
- **One module owns the database.** `tasks/store.py` is the only Task Store
  module that talks to Postgres and defines an idempotent `cartridge` schema.
  `storage/cache.py` plays the same role for the OCR page cache.
- **Pure, deterministic sampling.** `tasks/sampler.py` takes rows and returns
  rows, using largest-remainder quotas and a canonical query hash so the same
  query yields the same eval/exemplar/train selection on any machine.
- **Modal apps, volumes and classes.** Fine-tuning (`training/finetune.py`),
  the OCR service (`serve/vllm_server.py`), the Teacher
  (`cartridge/teacher.py`), distillation (`cartridge/train_distill.py`) and
  cartridge serving (`cartridge/serve_cartridge.py`) are all Modal apps. Model
  weights, Parquet target shards and trained cartridges live on Modal Volumes.
  `training/config.py` keeps paths as `str` so configs pickle across the Modal
  boundary.
- **Recipe-hash model ids for OCR serving.** `serve/recipe.py` freezes model
  SHA, preprocessing, prompt, decoding and postprocessing into an immutable
  `OcrRecipe`; its hash is the served model id and part of the page cache key,
  so a change in any recipe field invalidates cached transcriptions.
- **Content-addressed OCR cache.** `batch/keys.py` hashes PDF bytes and
  (file, page, DPI) with BLAKE2b; `storage/cache.py` keys transcriptions on
  (page_hash, model_id) so reruns never re-OCR a page.
- **The Teacher's prefilled journal context.** The Teacher answers with the
  entire `teacher_context.md` in its prompt. `teacher.render()` is shared by
  Teacher, student and base systems so prompts differ only in the context
  block. Distillation targets are top-k logprobs stored as columnar Parquet
  (`tasks/targets.py`), not plain text.
- **The cartridge as trainable KV prefix plus DeltaNet states.**
  `cartridge/cache.py` defines a trainable KV prefix for the attention layers
  and initial conv/recurrent states for the Gated DeltaNet layers of a frozen
  qwen3_5 model, initialized from a prefill and saved as safetensors. Only
  cartridge tensors receive gradients; the base model is frozen under gradient
  checkpointing.
- **Two training stages.** Stage 1 (`train_distill.py`) minimizes cross-entropy
  against the Teacher's top-k distribution. Stage 2 (`grpo.py`) runs async
  GRPO with a generation thread, a staleness-aware buffer and a clipped
  importance-ratio loss; `evals.py` supplies the grounded reward.
- **OpenAI-compatible gateway.** `serve/gateway.py` speaks the chat-completions
  dialect including SSE chunks and a final `[DONE]`. Modal does not stream, so
  the gateway chunks finished text, strips think tags, reports usage, and
  retries only transient transport errors. `serve/chat_models.py` maps each
  exposed model id to a teacher, cartridge or mock backend; adding a cartridge
  is one registry row.
- **W&B conventions and cost budgets.** `cartridge/tracking.py` enforces run
  naming and grouping, records config/git/Modal metadata, and accounts GPU
  cost against a budget.
- **Fakes over mocks in tests.** Hand-written `FakeStore`, `FakeTeacher` and
  `FakeVolume` classes let CLI and labeling tests run end to end in
  milliseconds without GPUs or a database.

## 4. Guided Tour

### Step 1: Project Overview and CLI

Start with the README, which documents only the legacy `cartridges
prepare-data` Prefect flow, then move to `cli.py`, the real map of the
platform: a `fire`-based CLI whose methods are the runbook for everything that
follows. `pyproject.toml` wires that module up as the `cartridges` console
script and lists the stack, while the Makefile wraps the common invocations
and the LibreChat lifecycle. Keep `docs/plans/cartridge-pipeline.md` open as
the narrative contract: self-study distillation of a Teacher into a KV-cache
cartridge, followed by async GRPO.

Open: `README.md`, `src/cartridges/cli.py`, `pyproject.toml`, `Makefile`,
`docs/plans/cartridge-pipeline.md`.

### Step 2: OCR Fine-Tuning and the OCR Service

`training/finetune.py` is a Modal app that LoRA-fine-tunes a vision-language
model with unsloth, logs to W&B, and runs k-fold and loop-rate evals, driven
by the Pydantic configs in `training/config.py`. `serve/recipe.py` freezes
how a page becomes text into an immutable `OcrRecipe` whose hash is the served
model id; `serve/registry.py` promotes verified SHAs on the HF Hub to a prod
tag, and `serve/vllm_server.py` packages vLLM behind a FastAPI `/ocr` endpoint
on Modal.

Open: `src/cartridges/training/finetune.py`, `src/cartridges/training/config.py`,
`src/cartridges/serve/recipe.py`, `src/cartridges/serve/registry.py`,
`src/cartridges/serve/vllm_server.py`.

### Step 3: Journal PDFs to Timeline

`batch/pipeline.py` turns whole journal PDFs into text: it renders pages in a
process pool into a bounded queue, fans them out to N async OCR workers,
detects repetition loops, retries, and stores one `PageRow` per page.
`storage/cache.py` is the Postgres layer under it. `batch/timeline.py` and
`batch/outputs.py` date the pages, heap-merge all journals into one
chronological `timeline.md`, and collate them into `teacher_context.md` with
stable page anchors, the single document the Teacher will hold in context.

Open: `src/cartridges/batch/pipeline.py`, `src/cartridges/storage/cache.py`,
`src/cartridges/batch/timeline.py`, `src/cartridges/batch/outputs.py`,
`src/cartridges/processors/ocr.py`.

### Step 4: The Task Store

`docs/plans/task-db.md` is the design doc; `tasks/types.py` defines the frozen
Pydantic records (Corpus, Page, GenRun, Task, Label, Sample) and their Literal
kinds, origins and splits, and `tasks/store.py` is the only module allowed to
talk to Postgres. Three ingestion paths populate it: `seed.py` turns the
user's own questions into origin `human` tasks, `chats.py` mines ChatGPT
history for real prompts, and `psych.py` loads psychology instruments as
`framework_lens` tasks.

Open: `docs/plans/task-db.md`, `src/cartridges/cartridge/tasks/types.py`,
`src/cartridges/cartridge/tasks/store.py`, `src/cartridges/cartridge/tasks/seed.py`,
`src/cartridges/cartridge/tasks/chats.py`.

### Step 5: Synthesizing and Sampling Tasks

`tasks/synthesize.py` plans recipes (kind, style, focus page chunks,
exemplars), prompts the journal-context model, parses JSON-schema drafts and
validates them before inserting new tasks; `tasks/validate.py` holds the
shared checks. `tasks/sampler.py` is the deliberately pure counterpart: a
deterministic, stratified selection by split, kind mix and framework.
`tasks/psych.py` rounds out the origins with held-out psychology instruments.

Open: `src/cartridges/cartridge/tasks/synthesize.py`,
`src/cartridges/cartridge/tasks/validate.py`,
`src/cartridges/cartridge/tasks/sampler.py`,
`src/cartridges/cartridge/tasks/psych.py`.

### Step 6: Teacher Labeling and Targets

The Teacher is a vLLM-served Qwen model on Modal that answers with the entire
`teacher_context.md` in its prompt; `cartridge/teacher.py` defines it, an
HF-based Scorer for top-k logprobs, and the shared `render()`. `tasks/label.py`
streams sampled tasks through the Teacher, validates completions (citations,
refusals, item echo), and records labels, while `tasks/targets.py` writes the
top-k logprob distributions to Parquet shards on a Modal Volume.
`docs/plans/label-queue.md` explains the throughput problem this stage faces.

Open: `src/cartridges/cartridge/teacher.py`,
`src/cartridges/cartridge/tasks/label.py`,
`src/cartridges/cartridge/tasks/targets.py`, `docs/plans/label-queue.md`.

### Step 7: Training the Cartridge

This is the core idea of the project. `cartridge/cache.py` defines a
Cartridge: a trainable KV prefix for the attention layers plus initial
conv/recurrent states for the Gated DeltaNet layers of a frozen qwen3_5 model.
`train_distill.py` is Stage 1: self-study context distillation against the
Teacher's top-k distribution, with sweeps, early stopping and a final eval.
`grpo.py` is Stage 2: asynchronous GRPO; `evals.py` supplies the grounded
reward and the base/teacher-ICL/cartridge comparisons, and `tracking.py`
enforces W&B naming and GPU-cost budgets.

Open: `src/cartridges/cartridge/cache.py`,
`src/cartridges/cartridge/train_distill.py`, `src/cartridges/cartridge/grpo.py`,
`src/cartridges/cartridge/evals.py`, `src/cartridges/cartridge/tracking.py`.

### Step 8: Serving Through LibreChat

`cartridge/serve_cartridge.py` is a deployed Modal class that keeps the base
model plus one cartridge loaded per container; `serve/backends.py` wraps it
and the Teacher with spawn/get timeouts and maps transport failures to
`TransientBackendError`. `serve/gateway.py` is an OpenAI-compatible FastAPI
app exposing `/v1/models` and `/v1/chat/completions`, with
`serve/chat_models.py` mapping each exposed model id to a backend.
`librechat/librechat.yaml` points LibreChat's single custom `Journal` endpoint
at that gateway.

Open: `src/cartridges/cartridge/serve_cartridge.py`,
`src/cartridges/serve/backends.py`, `src/cartridges/serve/gateway.py`,
`src/cartridges/serve/chat_models.py`, `librechat/librechat.yaml`.

### Step 9: End-to-End Chat Test and CI

`docs/plans/librechat.md` is the contract for the stack (ports, keys,
component ownership, done criteria). `e2e/librechat.spec.ts` proves it works:
a Playwright test logs into LibreChat, selects the Journal endpoint, and sends
the three fixture questions from `e2e/fixtures/tasks.json`. The GitHub
Actions workflow runs that suite on every push and PR and uploads the HTML
report, and `tests/test_gateway.py` covers the gateway without a browser.

Open: `docs/plans/librechat.md`, `e2e/librechat.spec.ts`,
`e2e/fixtures/tasks.json`, `.github/workflows/playwright.yml`,
`tests/test_gateway.py`.

### Step 10: Test Suite and Reports

The pipeline is unit-tested without GPUs or a database:
`tests/test_cartridge_train.py` checks the distillation loss against teacher
top-k, cartridge-only gradients and checkpointing parity on a tiny model,
while `tests/test_cartridge_tasks_cli.py` exercises the whole tasks CLI
against a FakeStore and FakeTeacher. Finish with the written record:
`docs/report/summary.md` is the team-lead review answering ten questions
about the pipeline, and `docs/report/recipe.md` frames it as an ML system
design with success metrics, infrastructure and risks.

Open: `tests/test_cartridge_train.py`, `tests/test_cartridge_tasks_cli.py`,
`docs/report/summary.md`, `docs/report/recipe.md`.

## 5. File Map

### CLI Entry & Logging

- `src/cartridges/cli.py`: Fire CLI for data prep, OCR, model eval/promote/merge, Modal serving, gateway, notes, cartridge train/sweep/ask, and the Task Store lifecycle.
- `src/cartridges/logging/config.py`: configures stdlib logging with a timestamped level/name format.

### OCR Fine-Tuning

- `src/cartridges/training/finetune.py`: Modal LoRA fine-tuning app (unsloth image, cache volume, W&B, k-fold and loop-rate evals).
- `src/cartridges/training/config.py`: Pydantic download/train/eval/decoding configs with `str` paths for Modal pickling.
- `src/cartridges/training/collator.py`: DeepSeek-OCR vision collator: chat conversion, image tiling, label padding.
- `src/cartridges/training/dataset.py`: parses `page_NNNN.md` transcriptions into an ImageFolder dataset with a frozen split.
- `src/cartridges/training/encoder_mask.py`: patches the DeepSeek-OCR-2 vision encoder mask to match vLLM on newer transformers.
- `src/cartridges/training/fast_decode.py`: incremental n-gram blocker, sync-free MoE path, CUDA-graph decoder with static KV cache.
- `src/cartridges/training/loops.py`: `ends_in_loop` and `loop_rate` for the repetition-loop failure.
- `src/cartridges/training/metrics.py`: corpus CER and mean normalized edit distance.
- `src/cartridges/training/perf_metrics.py`: pure-Python profiling math (timing, nvidia-smi parsing, trace stats).
- `src/cartridges/training/profiling.py`: Modal profiler for single-page OCR inference with torch.profiler and a GPU sampler.
- `src/cartridges/training/profiling_train.py`: profiles one LoRA training step.
- `src/cartridges/training/profile_page_report.py`: renders a single-page profile run into an HTML report.
- `src/cartridges/training/review.py`: HTML review page of word-level prediction-vs-label diffs.
- `configs/finetune/example.yaml`: template fine-tune config to copy before a run.
- `configs/dataset/split.json`: frozen train/test membership (302 train, 76 test pages).
- `notebooks/001_build_ocr_dataset.py`, `notebooks/002_finetune_ocr_model.py`: marimo notebooks for dataset build and fine-tune workflow.

### OCR Serving & Data Pipeline

- `src/cartridges/serve/vllm_server.py`: vLLM behind FastAPI on Modal with `/ocr`, `/info`, `/health` and API-key auth.
- `src/cartridges/serve/recipe.py`: immutable `OcrRecipe`; its hash is the served model id.
- `src/cartridges/serve/registry.py`: HF Hub SHA resolution, promotion thresholds, prod tag, deployed recipe in a Modal Dict.
- `src/cartridges/serve/merge.py`: merges a LoRA adapter into base safetensors and pushes to the HF Hub.
- `src/cartridges/serve/inference.py`: Modal inference app for the OCR model using the training image.
- `src/cartridges/serve/evaluate.py`: evaluates the served OCR model through `/ocr` into an `EvalReport`.
- `src/cartridges/serve/schema.py`: OpenAI-style multimodal request/response schema and `ServeConfig`.
- `src/cartridges/batch/pipeline.py`: batch OCR pipeline (process pool, bounded queue, async workers, loop detection, retries).
- `src/cartridges/batch/backends.py`: `OcrBackend` protocol and async httpx client for the OCR service.
- `src/cartridges/batch/keys.py`: BLAKE2b `file_key` and `page_key`.
- `src/cartridges/batch/dates.py`: rule-based reader of handwritten entry dates with a DP pass over the timeline.
- `src/cartridges/batch/timeline.py`: groups pages into dated entries and heap-merges journals into `timeline.md`.
- `src/cartridges/batch/outputs.py`: transcript, PageIndex, manifest writers and `collate()` into `teacher_context.md`.
- `src/cartridges/storage/cache.py`: Postgres page cache keyed on (page_hash, model_id) plus `source_files` and `ocr_pages`.
- `src/cartridges/processors/ocr.py`: PaddleOCR and Modal VLM transcribers behind an `OcrEngine` protocol with a cache.
- `src/cartridges/processors/extraction.py`: Modal app serving Gemma-3 via vLLM to extract page dates.
- `src/cartridges/datagen/flows.py`: Prefect prepare-data flow converting a journal PDF to markdown page by page.
- `src/cartridges/datagen/loaders.py`: per-format document-to-markdown loaders (`LOADERS` registry).
- `src/cartridges/datagen/qa.py`: script-checkable QA pairs from journal passages with citation verification.
- `src/cartridges/datagen/qa_dataset.py`: torch IterableDataset that streams verified QA pairs.
- `src/cartridges/datagen/agent/pipeline.py`, `src/cartridges/datagen/agent/tools.py`: Claude Agent SDK prepare-data workflow and its in-process MCP page tools.

### Task Store

- `src/cartridges/cartridge/tasks/types.py`: frozen Pydantic records and Literal kinds, styles, origins, splits, statuses.
- `src/cartridges/cartridge/tasks/store.py`: idempotent `cartridge` schema and all insert/fetch/update functions.
- `src/cartridges/cartridge/tasks/taxonomy.py`: data-only kind/style/origin mixes and Synthesizer instructions.
- `src/cartridges/cartridge/tasks/seed.py`: human seed tasks stratified into eval/exemplar splits.
- `src/cartridges/cartridge/tasks/chats.py`: imports ChatGPT history, classifies first prompts, promotes them to origin `chat` tasks.
- `src/cartridges/cartridge/tasks/psych.py`: psychology test items as `psych_test` tasks with held-out instruments.
- `src/cartridges/cartridge/tasks/synthesize.py`: the Task Synthesizer with a revalidate path.
- `src/cartridges/cartridge/tasks/validate.py`: normalization, hashing, deterministic ids, Jaccard overlap, `validate_draft`.
- `src/cartridges/cartridge/tasks/sampler.py`: pure deterministic stratified sampling.
- `src/cartridges/cartridge/tasks/label.py`: Teacher labeling with completion validation and Parquet target shards.
- `src/cartridges/cartridge/tasks/targets.py`: Target Store reading and writing Parquet shards locally or on a Modal Volume.
- `src/cartridges/cartridge/tasks/qa.py`: joins validated labels, tasks and targets into frozen `QAPair` records.
- `src/cartridges/cartridge/tasks/qgen.py`: sequential question-generator eval driver that always stops the Teacher.
- `src/cartridges/cartridge/tasks/pairs.py`: with/without-journal answer pairs, blinded A/B, Weave annotation queue.
- `src/cartridges/cartridge/tasks/ksweep.py`, `src/cartridges/cartridge/tasks/neighbors.py`: exemplar-count sweep and embedding-neighbour metrics.
- `src/cartridges/cartridge/tasks/judge.py`: order-swapped side-by-side LLM judge with versioned rubrics.
- `src/cartridges/cartridge/tasks/cgc.py`: context gap closed metric with bootstrap CIs.
- `src/cartridges/cartridge/tasks/export.py`: renders tasks as reviewable Markdown grouped by kind.
- `src/cartridges/cartridge/tasks/rubrics/judge-v0.1.md`: current judge rubric with citation checks.

### Cartridge Training & Teacher

- `src/cartridges/cartridge/teacher.py`: vLLM Teacher on Modal, HF Scorer, shared `render()`, container lifecycle helpers.
- `src/cartridges/cartridge/cache.py`: the `Cartridge` with prefill init, HF `DynamicCache` integration, safetensors save/load.
- `src/cartridges/cartridge/train_distill.py`: Stage 1 distillation pipeline with sweeps, early stopping, final eval.
- `src/cartridges/cartridge/grpo.py`: Stage 2 async GRPO with staleness-aware buffer.
- `src/cartridges/cartridge/evals.py`: eval set construction, scorers, grounded reward, Modal eval runs.
- `src/cartridges/cartridge/data.py`: self-study conversations and grounded QA pairs for GRPO.
- `src/cartridges/cartridge/chunks.py`: token-bounded page chunks from the timeline with citations.
- `src/cartridges/cartridge/selfstudy.py`: four self-study seed families in first-person voice.
- `src/cartridges/cartridge/tracking.py`: W&B naming, metadata, GPU cost accounting, artifacts.
- `src/cartridges/cartridge/ask.py`: Modal function answering a question with base model plus cartridge.
- `src/cartridges/cartridge/serve_cartridge.py`: deployed Modal class serving base model plus one cartridge.
- `scripts/cartridge_drift.py`: measures layer-by-layer drift of a trained cartridge from its init prefill.

### Chat Gateway & LibreChat

- `src/cartridges/serve/gateway.py`: OpenAI-compatible FastAPI gateway with SSE chunking and retries.
- `src/cartridges/serve/backends.py`: Teacher and CartridgeServer adapters with spawn/get timeouts.
- `src/cartridges/serve/chat_models.py`: chat model id registry (teacher, cartridge, mock backends).
- `librechat/librechat.yaml`: single custom `Journal` endpoint at `http://127.0.0.1:8008/v1`.
- `librechat/.env.example`: template LibreChat env; `start.sh` fills in secrets.
- `librechat/start.sh`, `librechat/stop.sh`: start and stop gateway, Mongo, LibreChat, and idle Modal containers.
- `librechat/mongo.js`: persistent local mongod via mongodb-memory-server.
- `e2e/librechat.spec.ts`, `e2e/fixtures/tasks.json`: Playwright three-turn chat test and its questions.
- `playwright.config.ts`: chromium-only config with `webServer` entries for gateway and LibreChat.
- `.github/workflows/playwright.yml`: runs the e2e suite on push and PR, uploads the HTML report.
- `Makefile`: `setup`, `notebook`, `prepare-data`, and the `chat-*` lifecycle targets.

### Documentation & Config

- `docs/plans/cartridge-pipeline.md`: cartridge pipeline contract and definition of done.
- `docs/plans/task-db.md`: Task Store design proposal.
- `docs/plans/batch-ocr-pipeline.md`: batch OCR pipeline design.
- `docs/plans/label-queue.md`: label queue design to keep the Teacher GPU fed.
- `docs/plans/librechat.md`: LibreChat stack contract.
- `docs/plans/loop-elimination.md`, `docs/plans/inference-speed.md`, `docs/plans/profiling.md`: OCR investigations and plans.
- `docs/hf-vllm-parity.md`: closed investigation into HF vs vLLM OCR output differences.
- `docs/report/summary.md`, `docs/report/recipe.md`, `docs/report/sections/*.md`: the cartridge pipeline review report.
- `docs/cartridge-report.md`: in-progress results report for the cartridge experiment.
- `pyproject.toml`: project metadata, dependencies, `cartridges` console script.
- `environments/docs_to_markdown/docs_to_markdown/taskset.py`: verifiers.v1 taskset over the datagen loaders.
- `configs/eval/`, `configs/gepa/`, `configs/rl/`: Prime Intellect experiment configs per model family.

## 6. Complexity Hotspots

Files the graph marks `complex`, with the reason to tread carefully.

- `src/cartridges/cartridge/tasks/types.py`: highest fan-in file in the repo (15 importers). Changing a record or Literal vocabulary ripples through store, sampler, label, synthesize and the CLI.
- `src/cartridges/cartridge/tasks/store.py`: the only Postgres boundary; schema is idempotent DDL, so migrations are by hand.
- `src/cartridges/cartridge/tasks/synthesize.py`: recipe planning, prompt rendering, JSON-schema parsing and validation in one module.
- `src/cartridges/cartridge/tasks/label.py`: streams tasks through the Teacher, validates citations/refusals, and writes Parquet shards; many failure modes.
- `src/cartridges/cartridge/tasks/chats.py`: ChatGPT export parsing plus a classifier model; input format is external.
- `src/cartridges/cartridge/tasks/pairs.py`: blinded A/B pairs plus a Weave annotation queue; ordering and seeding matter.
- `src/cartridges/cartridge/tasks/judge.py` and `tasks/cgc.py`: statistical outputs (CIs, order-swapped verdicts) that are easy to bias.
- `src/cartridges/cartridge/teacher.py`: Modal lifecycle, vLLM top-k parsing, HF Scorer and `render()` shared by three systems (7 importers).
- `src/cartridges/cartridge/cache.py`: trainable KV prefix plus DeltaNet states tied to qwen3_5 internals and HF `DynamicCache` (6 importers).
- `src/cartridges/cartridge/train_distill.py`: full Stage 1 pipeline with sweeps and early stopping on Modal (6 importers).
- `src/cartridges/cartridge/grpo.py`: threads, a staleness-aware buffer and a clipped importance-ratio loss.
- `src/cartridges/cartridge/evals.py` and `cartridge/data.py`: scorers and dataset builders that define rewards and targets.
- `src/cartridges/cartridge/tracking.py`: W&B naming and GPU cost budgets that every run depends on.
- `src/cartridges/cli.py`: about 1400 lines and 38 subcommands; lazy imports hide dependencies.
- `src/cartridges/training/finetune.py`, `collator.py`, `fast_decode.py`, `profiling.py`, `perf_metrics.py`, `review.py`: GPU and CUDA-graph code tied to the DeepSeek-OCR model internals.
- `src/cartridges/serve/vllm_server.py`: owns the served recipe and the vLLM subprocess on Modal.
- `src/cartridges/batch/pipeline.py` and `batch/dates.py`: process pool plus async workers, and a DP date resolver over noisy OCR text.
- `src/cartridges/processors/ocr.py`: multiple OCR engines behind one protocol with a content-addressed cache.
- `src/cartridges/storage/cache.py`: shared by the per-page flow and the batch pipeline (6 importers).

## 7. How to Run Things

Install dependencies (also `make setup`):

```bash
uv sync
```

The Fire CLI in `src/cartridges/cli.py` is installed as `cartridges`. Fire
accepts both `prepare_data` and `prepare-data` spellings. Examples documented
in the repo:

```bash
uv run cartridges prepare-data data/journals/inputs/example.pdf data/journals/output/example.md
uv run cartridges prepare-data data/journals/inputs/example.pdf data/journals/output/example.md --use-agent
uv run cartridges tasks_sample --split eval --n 3 --seed 1
uv run cartridges gateway_serve --port 8008
```

Secrets (`DATABASE_URL`, `HF_TOKEN`, `WANDB_API_KEY`) live in `.env.local`.
The Makefile and `docs/plans/librechat.md` load them with
`set -a && . ./.env.local && set +a` before running the CLI.

Other CLI subcommands (run `uv run cartridges` to list them): `serve_deploy`,
`serve_status`, `notes_process`, `notes_timeline`, `notes_collate`,
`cartridge_train`, `cartridge_sweep`, `cartridge_ask`, `tasks_init`,
`tasks_seed`, `tasks_synthesize`, `tasks_label`, `tasks_stats`, `eval_cgc`,
`eval_judge`.

LibreChat stack (see `Makefile` and `docs/plans/librechat.md`):

```bash
make chat-up        # gateway (8008) + Mongo + LibreChat (http://localhost:3080)
make chat-test      # 3-turn Playwright chat; starts the stack if needed. MODEL=journal-teacher-27b
make chat-status    # what is listening locally and running on Modal
make chat-down      # stop all three and this app's idle Modal GPU containers
make chat-clean     # also delete Mongo data (users, chats), logs and Playwright output
```

`make chat-test` defaults to `MODEL=mock-echo`, which needs no GPU. Pass
`MODEL=journal-teacher-27b` to exercise the real Teacher (cold start takes
minutes and bills a GPU).

Tests:

```bash
uv run pytest
```

Postgres round-trip tests run only when `CARTRIDGES_TEST_DATABASE_URL` is set.
`tests/cartridge_cache_modal.py` and `tests/encoder_mask_modal.py` are the GPU
halves of their tests and run on Modal, not under plain pytest.

Notebooks:

```bash
make notebook       # uv run marimo edit notebooks/
```
