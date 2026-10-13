# Plan: eliminate end-of-page looping in the fine-tuned OCR model

**Status:** in progress (started 2026-10-08)
**W&B:** project `conle-ai/cartridges`, group **`eos-loop-debug`**, tag `looping`

## Problem

The LoRA-fine-tuned DeepSeek-OCR-2 adapter (`joshzastrow/deepseek-ocr2-lora@run-liw4qzol`,
trained with `configs/finetune/lr_decay_100steps.yaml`) repeats its last few words on
roughly 10% of eval pages instead of stopping. Median test CER is 19%, but mean test CER
is 698%: a few runaway pages dominate it, which is why CER alone hid the problem.

## Metric (single value)

**`loop_rate`**: share of pages whose transcription *ends* in a loop that the
hand-written reference doesn't. Lower is better; the target is **0**.

A transcription ends in a loop when its final words are a phrase of 1–12 words repeated
back-to-back ≥3 times (lowercased, punctuation stripped; a trailing partial copy cut off
by the token cap still counts). Repetition in the middle of a page doesn't count. Pages
whose reference itself ends that way ("Repair, repair, repair.") are forgiven.

- Code: `src/cartridges/training/loops.py`, tests: `tests/test_loops.py`
- On the references alone it scores 0.000 (train and test), so it has no false positives on real handwriting.
- **Corpus CER** (`training/metrics.py`: all character edits ÷ all reference characters; blank
  references excluded and counted) is logged next to it as a guardrail, not optimized. The
  mean of per-page CER was dropped: one blank page (test/57, empty reference) made it 1000%+.

## EOS audit (done before any experiment)

`finetune.py::eos_audit` loads the tokenizer through Unsloth (as training does) and collates a
padded batch:

| Check | Result |
|---|---|
| EOS id: tokenizer / model config / generation config | 1 / 1 / 1 (`<｜end▁of▁sentence｜>`), all match pre-training |
| pad == EOS? | No (pad id 2), so value-based label masking does **not** erase EOS |
| EOS supervised in labels | Yes |
| Tail of a training target | `'Ġramp', '.', 'Ġ', '<｜end▁of▁sentence｜>'` |

**Finding:** `collator.py` builds the target as `f"{text} {eos}"`, which puts a standalone
space token (`Ġ`, id 223) before EOS. Pre-training (`conversation.py`, `sep2`) appends EOS
directly after the text. The fine-tuned model therefore learns "lone space, then EOS" as
its stop sequence. Under greedy decoding a lone space loses to a space-prefixed word token
(`Ġword`), so the model keeps writing instead of stopping.

## Eval set

1. Run the base model and `run-liw4qzol` over all 387 labeled pages (309 train + 78 test),
   with infer()'s default decoding (`baseline-base`, `baseline-ft-liw4qzol`).
2. **Loop set:** every page where either model loops. **Control set:** the same number of
   pages where neither does (seeded random), so a fix that just truncates output can't
   look good.
3. Stored locally at `data/train/loop_eval/manifest.jsonl` (`{"split", "index", "set"}`).
   Images stay in `data/train/hf_dataset`. W&B gets the page ids as a `loop-eval-items`
   dataset artifact. No journal images or text go to W&B.

## Experiments (one run each, in order, check-in after each)

| # | Hypothesis | Change | Model |
|---|---|---|---|
| E1 | Decoding alone fixes it | `no_repeat_ngram_size` 35 → 10 | existing adapter |
| E2 | The space-before-EOS target breaks stopping | Collator appends EOS with no space; retrain with the identical config | new adapter |
| E3 | The fixed model is still undertrained | E2 fix + `max_steps` 200 | new adapter |
| E4 | Confirm the winner | Best of E1–E3 on loop set + control set and full test split | best |

(The original E2, "override inference EOS id", was dropped: the audit showed the ids
already match everywhere.)

## W&B naming

Every run in this investigation goes in group **`eos-loop-debug`** with tag `looping`
plus the experiment's tag (`e0`…`e4`). Run names are `<experiment>-<job>-<change>`:

| Experiment | Training run (`job_type=finetune`) | Eval run (`job_type=loop-eval`) |
|---|---|---|
| E0 baselines | (existing adapter `run-liw4qzol`) | `e0-eval-base`, `e0-eval-ft-liw4qzol` |
| E1 decoding | none | `e1-eval-ngram10` |
| E1b decoding + repetition penalty | none | `e1b-eval-ngram10-rep105` |
| E2 EOS fix | `e2-train-eos-fix` | `e2-eval-eos-fix` |
| E3 EOS fix + 200 steps | `e3-train-eos-fix-200steps` | `e3-eval-eos-fix-200steps` |
| E4 confirm | none | `e4-eval-<winner>-test` |

Smoke tests and other plumbing go in group `infra-smoke-tests`, never in `eos-loop-debug`.
The two E0 baselines were launched before this scheme as `baseline-base` and
`baseline-ft-liw4qzol`; they get renamed when they finish.

## Running

```bash
# eval (spawns, prints CALL_ID; results come back as a dict with predictions)
uv run modal run --detach -m cartridges.training.finetune::run_loop_eval \
    --run-name e1-eval-ngram10 --tags e1 --items data/train/loop_eval/manifest.jsonl \
    --model-source artifact --hf-revision run-liw4qzol --no-repeat-ngram-size 10

# retrain
uv run modal run --detach -m cartridges.training.finetune::run_finetune \
    --config-file configs/finetune/eos_fix_100steps.yaml
```

## Done when

- `loop_rate == 0` on the loop set and on the full test split
- Test median CER no worse than the current adapter (19%) by more than 1 point

## Results log

| Run | Model | Change | loop_rate | n_looped / n | corpus CER | W&B |
|---|---|---|---|---|---|---|
| E0 `e0-eval-base` | base DeepSeek-OCR-2 | none (infer() defaults) | **0.240** (test 0.321) | 93 / 387 | 94.4% | [xz1yd8c3](https://wandb.ai/conle-ai/cartridges/runs/xz1yd8c3) |
| E0 `e0-eval-ft-liw4qzol` | adapter `run-liw4qzol` | none (infer() defaults) | **0.062** (test 0.077) | 24 / 387 | 42.1% | [yrbjfidv](https://wandb.ai/conle-ai/cartridges/runs/yrbjfidv) |
| E0b `e0b-eval-ft-liw4qzol-fast` | adapter `run-liw4qzol` | re-baseline: fast decode, 2,560 cap, 212-page eval set | **0.071** (loop set 10/107, control 5/105) | 15 / 212 | 36.6% (mean NED 0.234) | [4bvd0h3m](https://wandb.ai/conle-ai/cartridges/runs/4bvd0h3m) |
| E1 `e1-eval-ngram10` | adapter `run-liw4qzol` | `no_repeat_ngram_size` 35 → 10 | **0.024** (loop set 4/107, control 1/105) | 5 / 212 | 35.4% (mean NED 0.224) | [qqrnb27e](https://wandb.ai/conle-ai/cartridges/runs/qqrnb27e) |
| E1b `e1b-eval-ngram10-rep105` | adapter `run-liw4qzol` | n-gram 10 + `repetition_penalty` 1.05 | **0.019** (loop set 3/107, control 1/105) | 4 / 212 | 34.2% (mean NED 0.219) | [quibo1kg](https://wandb.ai/conle-ai/cartridges/runs/quibo1kg) |
| A `e5-eval-ft-liw4qzol-fixprompt-ng35` | adapter `run-liw4qzol` | **collator prompt** (15e1308), server decoding 35 / 1.0 | **0.066** (loop 14/107, control 0/105) | 14 / 212 | 36.7% (mean NED 0.225) | [09ebxlfv](https://wandb.ai/conle-ai/cartridges/runs/09ebxlfv) |
| B `e5-eval-ft-liw4qzol-fixprompt-ng10-rep105` | adapter `run-liw4qzol` | collator prompt, n-gram 10 + rep 1.05 | **0.019** (loop 3/107, control 1/105) | 4 / 212 | 32.1% (mean NED 0.212) | [ili9fmb3](https://wandb.ai/conle-ai/cartridges/runs/ili9fmb3) |
| E2 `e2-eval-eos-fix` | **adapter `run-8eqhat68`** (E2: no space before EOS) | collator prompt, n-gram 10 + rep 1.05 | **0.009** (loop 2/107, control 0/105) | 2 / 212 | **30.1%** (mean NED 0.194) | [tba7ky3b](https://wandb.ai/conle-ai/cartridges/runs/tba7ky3b) |

E0 notes: 9 pages loop under both models, 84 only under the base model, 15 only under the
fine-tuned one. Eval set built from these: `data/train/loop_eval/manifest.jsonl`, 108 loop
pages (79 train, 29 test) + 108 control pages (94 train, 14 test). The fine-tuned model
decodes at ~9 tok/s median vs ~19 tok/s for the base model.

Corpus CER notes (E0 fine-tuned, 42.1%): 12 pages have references longer than the 1,024-token
output cap; they hold 20% of reference characters but 38% of all character errors (71% of them
deletions, i.e. truncation). train/244 alone has a 46,397-character (10,880-token) reference,
likely a labelling error (several pages in one label), and accounts for ~18% of all edits.
Pages with references within the cap: corpus CER 32.7%.

**Dataset change, 2026-10-12.** PDF pages 87 (a typed, not handwritten, 53k-px page) and
91–93 (scribbles) were removed from `There He Is.pdf`; later pages were renumbered (old 88–90
→ 87–89, 94–102 → 90–98) and their labels/images moved to `_removed/`. The split is now frozen
in `configs/dataset/split.json`: the test split is identical (same 78 pages, order and text);
train is 305 rows (old rows 26, 116, 132, 244 removed). The eval set
(`data/train/loop_eval/manifest.jsonl`) was remapped to the new row indices: 108 loop + 106
control (2 control pages removed). E0 result files keep the old indices
(`data/train/hf_dataset.bak-2026-10-12`).

**Dataset change, 2026-10-13.** Morning_Pages 21 (blank), 35, 214, 250 (drawings, no text) and
87 (typed) were dropped: labels moved to `Morning_Pages/_removed/`, split now 302 train / 76
test, eval set 212 pages (107 loop + 105 control; 21 and 35 were in it). Backups:
`hf_dataset.bak-2026-10-13`, `manifest.bak-2026-10-13.jsonl`.

E0b notes: all 15 looping pages stopped on their own (EOS) after repeating a phrase several
times; none ran to the 2,560-token cap (longest 31 s). So the model does emit EOS; the
failure is a stutter of 3–15 repeats before it, and on some pages (e.g. test/47: 717 output vs
2,962 reference chars) the loop replaces the rest of the page. 14 of 15 are Morning_Pages.

E1 notes: 10 of E0b's 15 loops are gone, no new loops; 134/212 outputs are byte-identical to
E0b. Per page (NED change > 0.05): 29 better, 17 worse (worst: There_He_Is 83, 93, 73, long
pages where the 10-gram block forces a wrong word on a legitimately repeated phrase). The 5
remaining loops repeat 1–3-token phrases ("hunger,", "no,", "- Test", "I am scared.") that a
10-token block lets through ~3+ times, and they mutate around the block ("hungry", "Tools").

E1b notes: 4 of E1's 5 loops end, but 3 new ones appear (MP 260, 135; There_He_Is 57), so
loops move rather than vanish. Only 21/212 outputs match E1 exactly: the penalty touches
nearly every page. Per page vs E1 (NED change > 0.05): 40 better, 30 worse. Aggregates improve
on both sets (mean NED loop 0.273 → 0.265, control 0.175 → 0.171), but the worse pages show
penalty artefacts: invented words ("crushingishness", MP 8) and a Chinese token ("创业",
There_He_Is 87) where the right English word had already been used.

**Eval prompt fix, 2026-10-13.** Runs above A were measured with `model.infer()`, whose
`format_messages()` strips the trailing space of the training prompt `"<image>\nFree OCR. "`,
so the model saw a prompt one token shorter than in training. From `15e1308` the eval builds
inputs with the training collator. A, B and E2 use the fixed path; compare them only with each other.

## Outcome (2026-10-13)
- Prompt fix alone (A vs E0b, different prompts): loop_rate 0.071 → 0.066 — little change.
- Decoding (B vs A): n-gram 10 + rep 1.05 cuts loops 14 → 4 and corpus CER 36.7% → 32.1%;
  control pages unchanged (NED 0.172 → 0.171).
- EOS fix (E2 vs B, same decoding): loops 4 → 2, corpus CER 32.1% → 30.1%, control NED
  0.171 → 0.148; 0 loops on the 42 test-split pages in the set. Per page: 52 better, 33 worse
  (NED change > 0.05). Remaining loops: Morning_Pages 62, 63.
- **Decision:** adapter `run-8eqhat68` (E2) + decoding n-gram 10 / repetition penalty 1.05.
  E3 (200 steps) and E4 dropped. Before production: merge E2, serve on vLLM with matching
  decoding (check how the server's n-gram processor maps to HF's whole-sequence rule) and
  re-run the 212 pages through the server. The HF-vs-vLLM CER gap is tracked separately.
