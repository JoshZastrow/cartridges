# Cartridges: a whole journal in 0.3 GB

## The idea in one picture

A language model can answer questions about your journal if it reads the journal first.
To read it, the model fills a KV cache with one key and one value per token, in every attention layer.
Our journal is 714 handwritten pages, about 205,000 tokens.
On Qwen3.8-27B, that cache is about **13.4 GB**. It takes about 33 s to build on an H200, and the model must hold it for every question.

A **cartridge** replaces that cache with a small one that we train.
It has 2,048 positions and uses about **0.3 GB**: about **45× less memory**.
You load it like a file. No prefill is necessary.

Think of two students:
- **The Teacher** reads the full journal again before each answer.
- **The cartridge** is the memory of having read it. It is small, fast and always ready.

## What we measured

We train the cartridge so that the frozen base model, with the cartridge in place of the journal, gives the same next-token probabilities as the Teacher.

| Measure (198 held-out questions) | Base model (no journal) | **Cartridge (0.3 GB)** | Teacher (13.4 GB journal) |
|---|---|---|---|
| Perplexity on the Teacher's answers | 3.61 | **2.09** | 1.34 |
| Top-1 token agrees with the Teacher | 68.7% | **81.3%** | 100% |
| Validation KL to the Teacher (top-20) | 0.82 at init | **0.38** | 0 |

**Context gap closed (CGC) = 0.55** (95% CI 0.53–0.58).
CGC = (NLL_base − NLL_cartridge) / (NLL_base − NLL_teacher).
A value of 1 means "as good as the full journal". A value of 0 means "no better than no journal".
So the cartridge closes **55% of the gap**, with **2% of the memory**.

The gap closed is a little larger on names (0.65) and dates/numbers (0.65) than on ordinary text (0.54).
Point recall of single events is the weakest area (0.25, 3 questions).
A side-by-side LLM-judge comparison (cartridge vs Teacher vs base) is in progress.

## How it works

1. **OCR.** A LoRA-tuned vision-language model reads the handwritten PDFs. A batch pipeline dates the pages and merges them into one timeline (`teacher_context.md`).
2. **Task Store.** Postgres holds the questions: the author's own questions, real prompts mined from ChatGPT history, psychology-test items, and questions that a generator writes from journal pages.
3. **Teacher.** Qwen3.8-27B on vLLM (Modal, H200) answers each question with the full journal in context. It cites pages as `[Journal:p.N]`. We keep its top-20 token probabilities as targets (Parquet).
4. **Distillation.** The base model is frozen. Only the cartridge trains: a KV prefix for the attention layers and initial states for the Gated DeltaNet layers. The loss is KL(Teacher top-20 ‖ student). We use Adam with a per-tensor relative learning rate (5e-2) and early stopping.
5. **Evaluation.** CGC on held-out questions, an order-swapped LLM judge, and citation validity.
6. **Serving.** An OpenAI-compatible gateway serves the Teacher or the cartridge to LibreChat.

![ML system design](docs/report/diagrams/ml-system.svg)

Interactive version: [`docs/report/diagrams/ml-system.html`](docs/report/diagrams/ml-system.html).

## Training results

![27B training curves](docs/report/figures/train-27b.png)

- **Sweep:** relative learning rates 1e-2, 2e-2, 5e-2 and 1e-1. The best is 5e-2. Five warmup steps remove the early gradient spike.
- **Scale-up:** 190 steps of 16 examples (about 2.9M tokens). Early stopping selected step 160 (validation KL 0.380).
- **Drift:** training turns the cache vectors 13–32° into new directions. The deep layers move most. See [`docs/report/figures/cartridge-drift.html`](docs/report/figures/cartridge-drift.html).
- **Cost:** about $26 of GPU time for the full project (labeling, sweep, training and evaluation).

## Project layout

| Path | What it does |
|---|---|
| `src/cartridges/cli.py` | One Fire CLI for all stages |
| `src/cartridges/training/`, `src/cartridges/serve/` | OCR fine-tuning and the OCR service |
| `src/cartridges/batch/` | PDF → pages → dated timeline |
| `src/cartridges/cartridge/tasks/` | Task Store, sampler, generator, labeling, CGC, judge |
| `src/cartridges/cartridge/teacher.py` | The vLLM Teacher and the shared prompt `render()` |
| `src/cartridges/cartridge/cache.py` | The `Cartridge` module |
| `src/cartridges/cartridge/train_distill.py` | Distillation, sweeps, early stopping |
| `src/cartridges/serve/gateway.py` | Chat gateway for LibreChat |
| `docs/ONBOARDING.md` | Guided tour of the code |
| `docs/plans/` | Design documents |

## Quick start

```bash
uv sync
set -a; source .env.local; set +a          # DATABASE_URL, W&B and Modal keys

# Ask the trained cartridge a question (base model + cartridge, no journal in context)
uv run cartridges cartridge_ask "What do I write about most?" cartridge-fedz6mv3:latest

# Measure context gap closed on the eval split
uv run cartridges eval_cgc --cartridge cartridge-fedz6mv3:latest --label_runs <eval label runs>
```

The journal data is private and is not in this repository.

## Prepare data

The `prepare-data` command runs a Prefect flow by default:

```bash
uv run cartridges prepare-data data/journals/inputs/example.pdf data/journals/output/example.md
```

Use the Anthropic agent workflow explicitly when you want the page-by-page agent loop:

```bash
uv run cartridges prepare-data data/journals/inputs/example.pdf data/journals/output/example.md --use-agent
```

The Prefect entrypoint is `src/cartridges/datagen/flows.py:prepare_data_flow`.
For local development, call it through the CLI or from Python. Once the repo is
published to GitHub, deploy it with Prefect Cloud using that entrypoint and the
GitHub source:

```bash
uvx prefect-cloud deploy src/cartridges/datagen/flows.py:prepare_data_flow \
  --from <github-account>/<repo-name> \
  --name prepare-data
```
