# Cartridge pipeline: self-study distillation → async GRPO

Team contract. Each agent owns the modules listed for it and builds against the
interfaces here. Change an interface only after messaging every agent that uses it,
and update this file in the same change.

## Goal

Train a cartridge on the journal timeline, then show that
**base model + cartridge (no timeline in context)** answers questions about the
journals about as well as **the teacher (base model + full timeline in context)**,
and far better than the base model alone. Ship a CLI command that loads the model
with the cartridge and answers a question, with citations.

References: Cartridges paper/code (https://github.com/HazyResearch/cartridges,
arXiv 2506.06266) and async GRPO with LoRA
(https://huggingface.co/blog/asyncgrpo-lora-hfjobs).

## Decisions (made with the user, 2026-10-13)

- **Model**: `Qwen/Qwen3.8-27B` serves as base, teacher and student (self-distillation, as in the paper).
  Its architecture is `qwen3_5`: 64 layers, of which 16 are `full_attention` (4 KV heads, head_dim 256)
  and 48 are `linear_attention` (Gated DeltaNet). It has a 262k-token context window,
  and its weights take about 54 GB in bf16. It needs `transformers==5.17.0` (pinned in every Modal image; vLLM 0.29.0).
  **Develop and smoke-test on `Qwen/Qwen3.5-4B`**, which has the same architecture family; switch to 27B only for real runs.
- **The cartridge is a trainable cache, not LoRA.** For each `full_attention` layer it is a trainable
  KV prefix of `p` tokens. For each `linear_attention` layer it is a trainable initial recurrent
  state, plus the conv state. Initialize it the way the paper does: prefill the base model on the first `p` tokens of the
  corpus and take the resulting cache. The model weights stay frozen.
- **Training**:
  - Stage 1 is self-study context distillation: minimize the KL between the teacher's
    next-token distribution (full timeline in context) and the student's (cartridge only),
    over self-study conversations.
  - Stage 2 is async GRPO on the same cartridge parameters, following the blog's structure:
    a generation worker and a trainer, an off-policy buffer, and importance-ratio clipping.
    The blog's "LoRA weights" are replaced by the cartridge tensors.
  - The reward is a grounded-answer score against the teacher's reference answer and the cited evidence.
- **Compute**: Modal. Use A100-80GB, or H100 if 27B prefill/backprop requires it. Reuse the
  existing secrets `wandb-secret` and `HF_TOKEN`. Use a new volume, `cartridges-cartridge-cache`.
- **Budget: about $50 of GPU in total.** Smoke runs use 4B. Before any 27B job, post
  an estimate (GPU type × hours × rate) to the team lead. Track spend in the run summaries.

## Context sizes

**The corpus is `data/journals/output/2026_10_13/teacher_context.md`**: 204,512 Qwen tokens. It is the same
timeline as `timeline.md` (194,461 tokens), in the same order, but with a `[journal:p.N]` anchor on every page,
and the citation metrics depend on those anchors. Use it for the teacher's ICL prefill, the cartridge init text,
and self-study sampling. At ICL, the full-attention KV cache is about 64 KiB per token, roughly 13.4 GB.

**Held-out split:** `data/cartridge/eval/heldout_pages.json` (owned by eval) is the only source of truth.
Self-study covers the whole corpus, as in the paper. Only QA/GRPO training pairs exclude held-out pages.
The test is therefore "unseen questions about corpus facts".

## Stages and owners

| # | Stage | Owner | Module (new code under `src/cartridges/cartridge/`) |
|---|-------|-------|------|
| 1 | Chunk sampling + self-study seed prompts | **datagen** | `chunks.py`, `selfstudy.py` |
| 2 | Teacher: load model, prefill timeline once, serve generation + top-k logprobs | **teacher** | `teacher.py` (Modal) |
| 3 | Training data: self-study conversations + held-out QA pairs, with teacher logprobs | **trainer** | `data.py`, `train_distill.py` |
| 4 | Cartridge module + async GRPO | **cartridge** | `cache.py`, `grpo.py` |
| 5 | Evals, eval datasets, reports per stage | **eval** | `evals.py`, `docs/cartridge-report.md` |
| 6 | W&B logging, naming, grouping, profiling | **monitoring** | `tracking.py` |
| — | Inference CLI | **cartridge** (with trainer) | `cli.py: cartridge_ask` |

The trainer owns the end-to-end orchestration: `cartridges cartridge-train` runs
stage 1 and then stage 2 using the other agents' modules.

## Interfaces (initial; owners refine)

- `chunks.sample_chunks(timeline_path, n, min_tokens=512, max_tokens=4096, seed=0, exclude_pages=frozenset(), count_tokens=approx_tokens) -> list[Chunk]`.
  A `Chunk` has `id`, `text`, `citations` (`journal:p.N`), `date_range`, `char_span`. Reuse
  `datagen.qa.passages` and the `[journal:p.N]` anchors where they fit.
- `selfstudy.seed_prompts(chunk, kinds=("structuring","summarization","question","use_case","creative"))`
  returns the seed prompts used for the paper's self-study.
- `teacher.Teacher` (Modal class):
  - `generate(messages, max_tokens, n) -> list[str]` answers with the full timeline in context.
  - `logprobs(messages, completion, top_k=20) -> TopK` scores a completion.
  - The timeline is prefilled once per container and reused through prefix caching.
- `cache.Cartridge` (an `nn.Module`):
  - `init_from_text(model, tokenizer, text, p)`, `save(path)`, `load(path)`.
  - `as_past_key_values(batch_size)` returns a hybrid cache that the HF forward accepts, and gradients flow to the cartridge tensors.
  - Artifact: `cartridge.safetensors` plus `cartridge.json` (base model id and revision, `p`, which layers are which type, training run ids).
- Eval dataset: `data/cartridge/eval/qa_heldout.jsonl` with fields `question`, `answer`, `answer_type`,
  `citations`, `evidence`. It is built from chunks held out from training sampling, and all three
  systems below are scored on it.
- The artifact is saved to the Modal volume, and the W&B artifact `cartridge-<run_id>` is stored alongside it.

## Proof of equivalence (definition of done)

All three systems are evaluated on `qa_heldout.jsonl` with the same judge, and the results go in
`docs/cartridge-report.md`:

1. **base**: the base model, no context.
2. **teacher (ICL)**: the base model with the full timeline in context.
3. **cartridge**: the base model with the trained cartridge and no timeline.

For each system, record accuracy (judge), token-F1 against the reference, citation
validity, and the memory and time-to-first-token of the cache. The pipeline succeeds when
cartridge ≥ 90% of teacher accuracy and cartridge ≫ base, with the KV/state memory
reported against ICL.

Done means all of the following:
- a trained cartridge is saved,
- `uv run cartridges cartridge-ask "<question>" --cartridge <path|wandb-artifact>` prints a grounded, cited answer,
- the report exists,
- a W&B project has every stage grouped under one experiment,
- tests pass (`uv run pytest tests/test_cartridge_*.py`).

## W&B conventions (monitoring owns; everyone follows)

- Project: `cartridges-selfstudy`.
- Group: `exp-<YYYYMMDD>-<short-name>` for each end-to-end experiment.
- `job_type` is one of `datagen | teacher | distill | grpo | eval | profile`.
- Run name: `<intent>-<job_type>[-<role>]-<model-short>-<p>tok-<hash>`; `<intent>` is the group's short name (`cartridge-train --name`), `<role>` the eval system or GRPO gen/train.
- All runs go through `tracking.init_run(...)`. Do not call `wandb.init` directly.

## Team rules

- Do not commit. The lead reviews and commits at the end.
- Stay in your own files. If you need a change in someone else's module, message the owner.
- Add tests in `tests/test_cartridge_<module>.py` that run on CPU with a tiny model or fakes.
- Before any 27B Modal run, message the lead with a cost estimate and wait for the go-ahead.
- Report blockers immediately. Do not work around another agent's interface silently.
