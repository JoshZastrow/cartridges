# Next experiments: closing the fidelity gap

Status: plan. Nothing here has run yet. Budget left: about $15.9 of the $50 cap
(tracked GPU spend, 2026-10-24).

## 1. Where we are

The stage 2 cartridge (`cartridge-v7txld2a:v0`, best step 20) was scored by
the Teacher on 40 held-out tasks (eval run `c3tfzh2d`, rubric `score-v0`):

| Dimension | Teacher | Cartridge | Delta | 95% CI |
|---|---|---|---|---|
| Fidelity | 4.63 | 2.87 | −1.76 | −2.26 to −1.24 |
| Coverage | 4.45 | 3.05 | −1.39 | −1.79 to −1.03 |
| Responsiveness | 4.84 | 3.97 | −0.87 | −1.18 to −0.58 |

In the pairwise judge, the cartridge loses 87.5% of the time, and 85% of its
answers have a factual error flagged in both orders (Teacher: 2.5%).

The main failure is **style without facts**. The cartridge writes the Teacher's
kind of answer: direct quotes with `[Journal:p.N]` citations. But it invents
the quotes or puts them on the wrong page. Its themes are usually right.
Citation validity (does the cited page exist) is 97.5%, so that metric does not
see this failure.

Two facts about the training data (train labels of the stage 1 runs):

- The Teacher's answers cite 670 of the 714 pages, median 15 citations per page.
  But 102 pages are cited at most twice, and 126 pages are cited 50 times or more.
- 2,146 of the 2,785 labels (77%) are psychology-test items. Chat prompts: 275.
  Synthetic questions: 364. The judge scores are lowest on chat and on the
  author's own questions.

## 2. What the paper says

Fu et al., "Rethinking On-Policy Distillation of LLMs II: One Training Example"
(arXiv 2609.04172). The setting is full-weight OPD on a skill gap (math, code,
tools), not a cartridge. The findings that apply to us:

1. **Long horizons.** OPD keeps improving for hundreds of steps. Each update
   removes a smaller share of the remaining gap, at the same rate for 1 or 17k
   questions. The learning rate only rescales the step axis.
2. **Diversity, not volume.** What predicts gains is *state coverage*: how many
   distinct regions of the Teacher's state space the student visits.
   16 diverse questions match the full dataset.
3. **Reused samples still help.** A fixed set of 64 off-policy answers keeps
   improving the student for about 200 steps.
4. **Content-light prompts work.** Empty or unrelated prompts train as well
   as real ones. This suggests OPD mostly transfers behavior.

The difference that matters: our gap is **knowledge** (the Teacher has the
journal in context), not skill. So read finding 4 as a split:

- Stage 2 (on-policy) can teach **calibration**: do not write a verbatim quote
  or page number unless it is known. Reverse KL penalizes confident tokens the
  Teacher finds unlikely, which is what an invented quote is. Few, diverse
  questions may be enough for this.
- Stage 1 (teacher-forced) is where **facts** enter. Each label hands the
  cartridge the right quotes and pages. Its coverage of the journal limits
  what the cartridge can know.

## 3. Hypotheses

- **H1 (calibration).** Stage 2 lowers invented quotes, and more steps on
  diverse questions lower them further.
- **H2 (knowledge).** Fidelity is limited by what stage 1 taught. More labels on
  thinly covered pages and on chat-style questions raise fidelity and coverage.
- **H3 (capacity).** A 2,048-position cartridge cannot hold enough detail.
  A larger one closes more of the gap at the same training cost.

The experiments below are ordered to test the cheapest hypothesis first and
to let each result choose the next step.

## 4. Experiments

### E0. Make evaluation cheap (prerequisite)

`answer_tasks` decodes one task at a time on an A100: about 10 tok/s, about a
minute per answer. Every later experiment needs evals, so fix this first.

- Change: decode in chunks of 16 with `Cartridge.generate_batch`; run on an H200.
- Check: greedy outputs match the one-at-a-time path on 5 tasks (same tokens,
  or a documented, explained difference from padding).
- Expected: a 40-task eval in minutes, not 40 minutes.
- Cost: about $0.5.

### E1. Quote validity: a free, exact fidelity metric

The judge finds invented quotes, but it is slow and shares the Teacher's bias.
Most fabrications have a mechanical test: the quoted text is not on the cited page.

- Change: for each `"…" [Journal:p.N]` in an answer, fuzzy-match the quote
  against page N (and its neighbors, for page-boundary text). Report
  `quote_validity` = valid quotes / all quotes, and `quotes_per_answer`.
- Run it on the saved answers of `c3tfzh2d`: Teacher, cartridge and base. No GPU.
- Use: a cheap metric for every checkpoint; validates the judge's fidelity scores.
- Cost: $0.

### E2. What did stage 2 add? (decision gate)

- Run: the same 40 tasks, scorer and judge on the stage 1 checkpoint
  (`cartridge-fedz6mv3`), with E0's decoder.
- Compare stage 1 vs stage 2 on fidelity, coverage and quote validity.
- Decision:
  - fidelity or quote validity rose clearly (CI of the paired delta above 0):
    H1 holds; run E3 first.
  - no clear change: the gap is knowledge; run E4 first.
- Cost: about $1.

### E3. Longer stage 2 on coverage-chosen questions (paper findings 1–3)

- Questions: 64, chosen for coverage, not drawn in label order:
  - the pages the Teacher cites for each question define what it covers;
  - choose greedily to cover the most pages not yet covered, across all periods;
  - at least half chat and author-style questions.
- Steps: 90 (3× `v7txld2a`), from the stage 2 checkpoint. Each rollout batch
  is used for 2 updates (finding 3), which halves generation per update.
- Validation: 64 on-policy questions, not 32, so the plateau is not noise.
  Log the share of the on-policy KL removed per 10 steps (the paper's
  absorption rate) to see whether the run is still learning.
- Checkpoints at steps 30, 60, 90: score with quote validity (E1) on all;
  the judge scorer on the best one.
- Success: cartridge fidelity ≥ 3.5, quote validity up, coverage not down.
- Cost: about $6 (90 steps at about 80 s, minus the reused generation).

### E4. Fill the stage 1 coverage gaps (H2)

- Labels: about 500 new Teacher answers:
  - questions generated from the 102 pages cited at most twice, plus their
    neighbors;
  - chat-style and synthetic questions, not psychology-test items.
- Train: resume stage 1 from `cartridge-fedz6mv3` on the new labels mixed with
  a sample of the old, then repeat a short stage 2 (30 steps) from that.
- Success: coverage and fidelity up on the 40 tasks; validation KL not worse.
- Cost: about $4 (labels about $2, training about $2).

### E5. Capacity curve on the 4B model (H3)

- Train stage 1 at p = 512, 2,048 and 8,192 on Qwen3.5-4B, same labels and steps.
- Report the context gap closed (CGC) and quote validity against p.
- Read: if CGC keeps rising at 8,192, capacity binds and a larger 27B cartridge
  is worth its memory. If it levels off by 2,048, put the budget into data (E4).
- Cost: about $3.

### E6. Attention Matching init (issue #2)

Stage 1 starts from the first 2,048 tokens of the context block, so it only
knows the first pages. Attention Matching (Zweiger et al., arXiv 2602.16284)
builds the start from the whole block in closed form. For each attention layer
and KV head, it keeps the 2,048 keys that the Teacher's own queries attend to
most, then fits their values by least squares, so attention over the kept keys
matches attention over all ~205k keys.

- No per-key bias β. Kept keys are moved to positions 0..2047. Every consumer
  is unchanged.
- The DeltaNet states are the Teacher's own after the whole block.
- Step 0: val KL for the prefix init, for the prefix with the block's DeltaNet
  states, and for the matched init.
- Then 60 stage 1 steps from the matched init, with the 6h8b2z9m config and
  data order, compared with its val KL curve (0.632 at step 10, 0.455 at
  step 60).
- Cost: about $4.

## 5. Budget and order

| Step | Cost | Runs if |
|---|---|---|
| E0 batched eval decode | $0.5 | always |
| E1 quote validity | $0 | always |
| E2 stage 1 vs stage 2 scores | $1 | always |
| E3 long stage 2, coverage questions | $6 | E2 shows stage 2 helps |
| E4 stage 1 coverage labels | $4 | E2 shows no help, or after E3 |
| E5 4B capacity curve | $3 | money is left after E3/E4 |
| **Total** | **$14.5** | of $15.9 left |

E3 and E4 together, plus E0–E2, fit the budget. E5 fits only if E3 or E4 runs
under its estimate.

## 6. How results are judged

- **Primary:** judge fidelity (rubric `score-v0`) and quote validity on the
  same 40 held-out tasks (stratified subset, seed 0). Report paired deltas
  with bootstrap CIs, not only means.
- **Secondary:** coverage and responsiveness scores; on-policy and
  teacher-forced validation KL; CGC on the 198 held-out questions.
- **Guard:** responsiveness must not fall. A cartridge that stops quoting can
  raise fidelity by saying less. Coverage and responsiveness catch that.
- Read 20 judge reasons per run by hand before trusting a score change.

## 7. Not in this plan

- Layer-local losses and per-layer attention matching: interesting, but they
  target optimization speed, not the fidelity failure. Revisit after E5.
- vLLM serving with cartridge injection: about 2 engineer-weeks; not needed
  for these experiments once E0 lands.
- A reward model or RL on judge scores: a per-token Teacher signal already
  exists and is cheaper and denser.
