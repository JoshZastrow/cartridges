# Stage 2b: mixed on-policy distillation of the cartridge

Status: built and piloted (run `v7txld2a`; results in the README, "Stage 2: on-policy distillation").

## 1. Why

Stage 1 is teacher-forced. The cartridge only ever reads the teacher's answers
(`train_distill.student_loss`). At inference it reads its own answers. Once it
writes one token the teacher would not have written, it is in a state it never
trained on (exposure bias). `eval.md` names this blind spot: a cartridge can
score the teacher's tokens well and still decode badly.

On-policy distillation (GKD, Agarwal et al. 2023, arXiv 2306.13649) trains on
the cartridge's own answers. The teacher grades every token of them.

## 2. The idea in one picture

```
step k (cartridge θ_k)
 ├─ half A: stored teacher answers  ──► forward KL(p_T ‖ p_S) on top-20     (stage 1 loss, unchanged)
 └─ half B: questions ─► cartridge θ_k SAMPLES full answers y   (no grad)
                         ├─ teacher pass:  [journal cache | q | y]  → p_T(·|y<t) at every t   (no grad)
                         └─ student pass:  [cartridge θ_k | q | y]  → p_S(·|y<t) at every t
                         reverse KL(p_S ‖ p_T) summed over t, full vocabulary
 one backward over A + B ─► one Adam step ─► θ_{k+1} ─► next step samples with θ_{k+1}
```

- **The update is once per step, not per token.** The cartridge writes the
  whole answer first, with gradients off. Then one forward pass over the full
  answer gives all T per-token KLs at once, because the whole sequence is known.
  There is one backward and one update per batch of 16. A per-token update
  would need about 400 backward passes per answer. It would also change the
  sampler in the middle of its own answer.
- **The loop is synchronous at the step level:**
  generate (batch) → score → backward → update → generate with the new cartridge.
  Every sample comes from the current θ_k. So there is no importance ratio and
  no clipping. `grpo.py`'s async queue is not needed.
- **The teacher never generates.** It only reads the cartridge's tokens
  (teacher forcing on the student's answer).
- **It is "mixed" or "semi" on-policy:** a fraction λ of each batch is
  on-policy (B), and the rest are stored teacher answers (A). Half A anchors
  the cartridge to good answers while its own samples are still weak.

It is not GRPO, PPO or DPO. There is no answer-level reward, no value network
and no preference pair. It is a supervised loss on the student's own samples.
In RL terms, it is a per-token reward r_t = log p_T(y_t) − log p_S(y_t) with
discount 0. We use the exact full-vocabulary KL instead of that one-sample
estimate.

## 3. Why one GPU is enough

Teacher and student are the same frozen Qwen3.8-27B. Only the prefix differs:
the journal (205k tokens) versus the cartridge (2,048 positions). One process
holds both prefixes and one copy of the weights:

| Item | Memory |
|---|---|
| Weights, bf16 | ~54 GB |
| Journal cache (`teacher.Scorer.prefill`, once per container) | 13.4 GB |
| Cartridge fp32, plus Adam moments | ~1.3 GB |
| Stage-1 training activations (measured peak 61 GB minus the weights) | ~7 GB |
| Teacher pass over q + y on a fork of the journal cache (`fork_cache` clones DeltaNet states, ~150 MB) | ~2 GB |
| **Total** | **~78 GB** |

That does not fit an A100-80GB. Stage 1 peaked at 61 GB, and the journal cache
adds 13.4 GB. **Use an H200 (141 GB).**

TITO comes for free: the sampled token ids go straight from `Cartridge.generate`
into both forward passes. Nothing is decoded and re-encoded. Text is decoded
for logging only.

## 4. Loss

For an on-policy sequence with prompt q and sampled answer y (T tokens):

    L_B = (1/T) Σ_t Σ_v p_S(v | y<t) · [log p_S(v | y<t) − log p_T(v | y<t)]

- **Full vocabulary, exact.** Teacher logits are not stored as [T, 248k] fp32,
  which would be 3 GB per answer. The teacher pass keeps its **last hidden
  states** [T, 5120] instead (bf16, ~8 MB at T = 768). `lm_head` is applied per
  1024-token chunk inside the loss, like `student_loss`.
- Step loss: `(1 − λ) · mean(L_A) + λ · mean(L_B)`. L_A is the stage-1
  `student_loss`, unchanged.
- Gradients flow into the cartridge only through p_S's logits. Sampling is
  treated as fixed data (stop-gradient through the sample), as in GKD.
- **Truncated samples are fine.** At `max_new_tokens` with no EOS, each
  position's KL is still valid. Unlike GRPO, no answer-level score is lost.
- **Early stop is penalized.** If the cartridge emits `<|im_end|>` where the
  teacher would continue, the teacher gives it a low probability, and that is
  costly.

## 5. Sampling

- **Questions:** split='train' tasks from the Task Store, with the same origin
  mix as stage 1 (chat 0.40 / synthetic 0.35 / psych_test 0.25). Half B needs
  no teacher labels, so unlabeled train tasks are usable too. The pilot keeps to
  labeled ones, so A and B draw from the same pool.
- **Prompt:** `label.answer_messages` rendered by `teacher.render`. These are
  the same `prompt_ids` as stage 1 and the teacher (`check_prompts`).
- **Grouping:** `Cartridge.generate_batch` decodes all of a step's questions
  together (up to `max_gen_rows` = 16 rows). Each prompt is prefilled alone and
  unpadded, because padding in a prefill would enter the DeltaNet recurrent
  state. The caches are then merged (`cache.merge_caches`):
  - attention keys and values are zero-padded behind each prompt and masked out;
  - DeltaNet conv and recurrent states are fixed-size and stack as they are;
  - each row keeps its own position ids.
  A single-token decode step passes no padding mask to the linear layers.
  Tests check that the batched logits match each prompt alone, and that
  unmasked padding is caught. Run 1 (before batching) measured ~31 s per
  question for 512 tokens, linear in the number of questions.
- **Settings:** temperature 1.0, top_p 1.0. The samples must come from the
  cartridge's own distribution, or the loss is not on-policy.
  `max_new_tokens` = 768.
- **λ schedule:** 0.25 for 10 steps, then 0.5. Go to 0.75 only if validation
  metrics (§7) are still improving.

## 6. Code changes

| Where | Change |
|---|---|
| `cartridge/onpolicy.py` (new, ~150 lines) | `sample(model, cart, prompt_ids, n, max_new_tokens)` → token ids, via `Cartridge.generate` and `grpo.trim_eos`. `teacher_hidden(model, journal_cache, prompt_ids, completion_ids)` → [T, d], no grad, on `fork_cache`. `reverse_kl_loss(model, cart, input_ids, teacher_h, chunk)` → (loss, kl), chunked and checkpointed like `student_loss`. `train_mixed(...)`, modelled on `train_distill.train`: same Adam, relative LR, warmup, NaN guard and `on_step` early stopping. |
| `train_distill.py` | `OnPolicyConfig` (λ schedule, questions per step, samples per question, `max_new_tokens`, temperature). A `PipelineConfig.onpolicy` stage `_onpolicy()` beside `_grpo()`: load the journal cache with `teacher.Scorer.prefill`, init from `init_cartridge` (`distill-best`), save `onpolicy-best`, log a W&B artifact. GPU H200 through `with_options`. |
| `cli.py` | `cartridge_train --onpolicy '{...}' --init_cartridge /vol/.../distill-best --gpu H200`. `dry_run` validates the config and the question pool on CPU. |
| `grpo.py` | No change. Its async loop and text reward stay for the answer-level reward (`grounded_reward`). |

### Tests (CPU, tiny model, before any GPU)
1. **Zero loss:** if the "journal" prefix equals the cartridge's own prefix,
   the reverse KL is 0 and its gradient is 0.
2. **Matches the brute-force value:** the chunked `reverse_kl_loss` equals a
   brute-force [T, V] computation (within 1e-5).
3. **Gradients reach only the cartridge:** base weights get no gradient, and
   the journal cache is unchanged after a step (fork, not in place).
4. **TITO:** with `tokenizer.encode` monkeypatched to raise during
   `train_mixed`, the sampled ids reach both forward passes.
5. **Mixing:** with λ = 0, `train_mixed` is bit-identical to `train` on the
   same order and seed.

## 7. How we judge it

The same eval sets as stage 1, so the comparison holds:

| Metric | Expectation |
|---|---|
| **On-policy val KL (new):** cartridge samples 32 split='eval' questions, the teacher grades them, reverse KL per token | The target metric. It should fall. |
| Teacher-forced `val/kl` (stage 1's curve, 0.380 at `distill-best`) | Must not rise by more than ~0.02. Half A is there for this. |
| CGC on the eval set (0.55 [0.53–0.58]) | Unchanged or better |
| Judge side-by-side (`eval_judge`, same task subset) | Fewer unsupported claims and citations |
| Answer length, abstention rate, repetition | **Watch for collapse.** Reverse KL is mode-seeking and can drift toward short or hedged answers. |

Early stopping is on the on-policy val KL, with patience 3 and evaluation every
5 steps. The guard on teacher-forced KL still applies.

## 8. Cost and rollout

Step time is an estimate, to be replaced by the probe's numbers:

| Part (batch 16, λ = 0.5) | Estimate |
|---|---|
| Generation: 4 groups × ≤768 tokens at an assumed 30–60 tok/s (HF decode, unmeasured) | 50–100 s |
| Teacher passes: 8 × T ≤ 768 on the journal fork | ~10 s |
| Student forward + backward: 16 sequences (stage 1 measured 17–21 s on A100) | ~15 s on H200 |
| **Per step** | **~75–125 s** |

At an assumed ~$4.5/h for an H200 (to confirm against Modal's current price):

1. **Probe** (needs approval): about 15 min, about $1.50.
   - Load the model, prefill the journal and record its time.
   - 3 steps at λ = 0.5.
   - Measure tok/s, peak memory and s/step.
   - Stop if peak memory is above 120 GB or a step takes more than 4 min.
2. **Pilot** (needs approval after the probe): 30 steps from `distill-best`
   with the λ schedule. About 1 h, about $5.
   - Report all four §7 curves.
3. **Decide.** Scale up (≤100 steps, early stopping, ~$10–15) only if the
   on-policy val KL falls and the teacher-forced KL holds. Otherwise stop and
   write up why.

## 9. Open questions

- **Teacher numerics.** Half A's targets came from vLLM, and half B's teacher
  runs in HF. bf16 differences are small, but the probe should check it: the HF
  teacher's top-20 on 5 stored labels against the stored top-20, as a KL.
- **After a wrong token, the teacher plays along.** It grades the rest of the
  answer as if the error were true, so the error gets the penalty only where it
  first appears. That is the right place to fix it, but the per-sequence KL
  understates how wrong the answer is. The judge covers this.
- **Unlabeled questions.** Half B could use the whole train question pool,
  labeled or not. That is cheaper than labeling more, but it is untested.
