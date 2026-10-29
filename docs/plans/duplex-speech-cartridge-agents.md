# Multi-agent instructions: implementing the duplex speech cartridge

Companion to `docs/plans/duplex-speech-cartridge.md` (the plan: survey,
requirements, spec). Tracking: epic #7, issues #8–#18 on the Cartridges
board. This file says how several Claude Code agents build it without
tripping over each other. The rules in `CLAUDE.md` (claim an issue before
working, worktree per issue, never commit to `main`, Status on the board is
the truth) apply unchanged; this file only adds what is specific to this epic.

## 1. What we are building, in one paragraph

A trained KV prefix ("cartridge") for NVIDIA PersonaPlex 7B that replaces its
voice-prompt + text-prompt with learned keys and values per temporal layer.
The cartridge is initialised from a prefill of the author's voice clip, a
persona text and a journal digest, then trained by self-study: a PersonaPlex
teacher with per-question evidence in its prompt supplies per-frame
distributions on the agent text stream and the 8 agent audio codebooks, the
student with the cartridge matches them (stage 1), then matches them on its
own rollouts (stage 2). The base model stays frozen. Serving needs a pinned
prefix region in the ring KV cache so the cartridge is never evicted.

## 2. Roles

- **Coordinator (one agent, no worktree).** Owns the board. Re-checks blockers
  when an issue moves to Done, flips the next issues to Ready, reads every
  summary comment, and keeps section 5 of this file (interface contracts)
  current. Does not write feature code.
- **Implementer (one agent per issue, own worktree).** Claims one Ready issue,
  builds exactly its acceptance criteria, opens a PR with `Closes #N`, posts
  the summary comment. Never edits files listed under another In Progress
  issue without commenting there first.
- **Reviewer (optional, one agent per PR).** Reviews against the issue's
  acceptance list and the contracts in section 5, nothing else.

Run at most four implementers at once (the four Ready issues of wave 1).
More than that only creates merge conflicts in `src/cartridges/duplex/`.

## 3. Waves (dependency order)

| Wave | Issues | Can run in parallel | Unblocks |
|---|---|---|---|
| 1 | #8 cartridge module + parity · #9 eval harness + baseline · #10 data · #18 hertz-dev pilot (optional) | all four | #11, #12 |
| 2 | #11 teacher study (needs #9, #10) · #12 pinned cache + server flag (needs #8) | both | #13, #15 |
| 3 | #13 teacher rollouts (needs #11, #10) | alone | #14 |
| 4 | #14 stage 1 distillation (needs #8, #13) | alone | #15, #16 |
| 5 | #15 stage 2 on-policy (needs #12, #14) · #16 voice term (needs #10, #14) | both | #17 |
| 6 | #17 voice LoRA, gated on #16's result | alone | done |

The coordinator flips an issue to Ready only when every "Blocked by" issue is
Done (merged). Do not start a Blocked issue "to get ahead": its contracts are
set by the issues before it.

## 4. Shared conventions for this epic

- **Code lives in `src/cartridges/duplex/`.** One module per concern as laid
  out in plan §4.1. Reuse `cartridges.cartridge.{teacher,tracking,tasks}`;
  do not copy them. No changes to `cartridges/cartridge/` except the
  `stream` column in `tasks/targets.py` (#13 owns that change).
- **PersonaPlex is vendored and pinned.** The `moshi` fork from
  `NVIDIA/personaplex` at commit `3428dfd`, installed into the Modal image
  from `train.py` (#8 sets this up; everyone else reuses the image). Patches
  to the fork (#12) live in `src/cartridges/duplex/patches/` and are applied
  at import, not committed into the vendored tree.
- **GPU work runs on Modal, `gpu="H200"`.** Every GPU entry point has a
  `--dry-run` that runs on CPU with a tiny fixture before spending money.
  Spend cap per issue: $40 unless the issue says otherwise; post the W&B
  run and cost in the summary comment. Phase 1 total is estimated at
  $160–200 and exceeds the cap recorded in `docs/plans/next-experiments.md`,
  so the coordinator confirms budget with the author before wave 3.
- **The author's voice.** `data/voice/josh-thoughts-1.m4a` (about 9.8 min,
  git-ignored). #10 converts it to 24 kHz mono WAV at −24 LUFS, splits it
  80/10/10, and uploads it to the Modal Volume `cartridges-voice` with a
  manifest. Nothing under `data/` is ever committed, uploaded to W&B as a
  public artifact, or pasted into an issue. Treat it as personal data.
- **Tests.** Every module has a CPU test with synthetic tensors or audio
  under `tests/test_duplex_*.py`. The parity fixture from #8 (stored logits
  of the stock prompt path) is the one GPU-derived fixture and is committed
  once, small (10 dialogues, 50 frames each, top-20 only).
- **Tracking.** W&B project as the text pipeline, run group
  `duplex-<issue>-<short>`, artifact type `duplex-cartridge`.
- **Style.** Match `cartridges/cartridge/cache.py`: pure functions where
  possible, IO injected, Pydantic configs, docstring at the top of each
  module stating the shapes.

## 5. Interface contracts (owned by the first issue that ships them)

Later issues code against these. Changing one means a comment on every issue
that consumes it, before the change.

**Cartridge (from #8)**
```
DuplexCartridge(keys: list[Tensor[1,32,p,128]], values: ..., num_frozen_tokens=1, meta: dict)
DuplexCartridge.from_prefill(lm_model, frames: Tensor[1,17,p]) -> DuplexCartridge
DuplexCartridge.init_from_prompt(lm_model, mimi, spm, voice_wav, persona_text, digest_text, p) -> DuplexCartridge
DuplexCartridge.save(dir) / .load(dir)          # cartridge.safetensors + cartridge.json
with cartridge.patched(lm_model, batch_size):   # training forward: cartridge K/V prepended, RoPE offset p
    out = lm_model.forward(codes)               # codes [B,17,T]; out.text_logits [B,T,32000], out.audio_logits [B,8,T,2048]
```
`cartridge.json`: `{base_model, fork_commit, p, num_frozen_tokens, voice_sha,
persona, digest_sha, stage, step}`.

**Dialogue and frames (from #10)**
```
dialogue.jsonl: {id, task_id, split, user_wav, brief, digest_ref, teacher_rollout_ref?}
frames/<id>.safetensors: user_audio int16 [T,8], agent_text int32 [T], agent_audio int16 [T,8]
```
Frames are 12.5 Hz, Mimi 8 codebooks, vocab 2048; text is SentencePiece 32k
with PersonaPlex's pad id for silence.

**Targets (from #13)**
```
parquet shard: id, frame, stream (0=text, 1..8=audio), topk_ids int32[20], topk_logprobs float32[20], sampled int32
```
Written with `tasks/targets.py` plus the `stream` column; read by #14, #15.

**Eval CLI (from #9)**
```
uv run cartridges duplex_eval --cartridge <dir|none> --tasks <jsonl> --out <dir>
```
Writes `report.json` with the rows of plan §3.5 (each `{value, ci_low,
ci_high, n}`) and logs to W&B. Baseline run name `duplex-baseline-stock`.

**Server flag (from #12)**
```
python -m moshi.server --cartridge <dir>    # fills the pinned region, skips step_system_prompts
python -m moshi.offline --cartridge <dir> --input user.wav --output agent.wav
```

**Trainer (from #14, extended by #15, #16)**
```
uv run cartridges duplex_train --config <toml>     # stage: distill | onpolicy | voice
```
Config is a Pydantic model in `train.py`; every stage reads the same
`DuplexCartridge` and writes the same artifact.

## 6. Per-issue guidance (what the issue body does not say)

- **#8.** Start from the shipped `.pt` voice path: `LMGen.step_embeddings`
  shows exactly which frames a prompt occupies and how `end_offset` moves.
  Build the prefill with the stock `step_system_prompts`, then read the
  `RingKVCache` out of each layer; that is the init. The parity test compares
  the non-streaming patched forward against the streaming stock path, so
  expect bf16 noise, not zero. Watch the delay pattern
  `[0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]`: the prefix must end on a frame
  where all streams are consistent.
- **#9.** Run the baseline before anything else exists; it is the only row
  every later issue compares to. The spurious-onset test is 20 runs of 5 min
  silence on the user channel; budget the GPU time. Keep the Qwen judge call
  identical to `train_distill.task_judge` so numbers are comparable to the
  text cartridge.
- **#10.** Voice first: convert, split, measure the author's TTS similarity
  to the eval split and record that number in the plan doc (it is the voice
  ceiling, plan §3.2 D2). Spoken questions use other voices only. Briefs come
  from the existing Qwen Teacher with the full journal, `label.py` style;
  keep page citations in the brief text so the PersonaPlex teacher can say
  them.
- **#11.** This is a measurement, not code. Four brief lengths × 40
  questions, three prompt lengths × duplex metrics. Post the two curves and
  the chosen sizes as a comment; the coordinator copies them into plan §3.2.
- **#12.** Pin, do not enlarge. Capacity stays 3000; the ring covers
  `p..2999`. The 6-minute survival test is the acceptance test. Document in
  the patch whether it is upstreamable.
- **#13.** Capture logits on the 9 agent streams only; user streams are
  inputs. Top-20 per stream per frame is ~3 GB for 3,000 dialogues; shard by
  200 dialogues. Transcribe and judge every rollout; keep ≥ 2,000 with
  fidelity ≥ 4. Reuse the `label.py` refusal and citation validators.
- **#14.** Loss weights are in plan §3.3 M3; do not tune them in this issue.
  Sweep the relative LR over four values with the same seed, pick on held-out
  teacher-forced KL, stop early. UTMOS every validation step on 20 rollouts
  is the guard against acoustic drift. p = 1024 first; p = 2048 only as a
  follow-up issue.
- **#15.** Mirror `onpolicy.py`: anchor mix 0.25 for 10 steps then 0.5,
  reverse KL over the full vocab, teacher re-scores the student's own agent
  streams with the brief in context. Serve rollouts through the #12 pinned
  cache so training matches inference.
- **#16.** The hard part is aligning the TTS render to the teacher's text
  frames. Document the alignment method (word timestamps from Whisper, as
  `moshi-finetune/annotate.py` does) before training. Report similarity with
  and without the term; the decision for #17 is in this issue's comment.
- **#17.** Only if #16 reports < 0.65. Write explicit LoRA wrappers for
  `in_proj_weight`, gating `F.linear` and the depformer stacked weights
  before anything else; module-swap LoRA silently skips them. Merge, then
  rerun #14 and #15 on the merged weights as a follow-up issue.
- **#18.** Isolated under `src/cartridges/duplex/hertz/`; shares no code.
  First deliverable is the one-line answer to whether hertz-ar's head is
  discrete or continuous, because it decides the loss. CUDA only.

## 7. Protocol, in order

1. Coordinator lists Ready and unassigned issues; implementers claim one
   each (assign, Status In Progress, claim comment, re-check for a race).
2. Implementer reads the plan section the issue cites, section 5 above, and
   the summary comments of the issues it is blocked by. Then works in
   `issue-<N>-<slug>` in its own worktree.
3. Any contract change or any file outside the issue's "Files/areas" list:
   comment on the owning issue first, wait for the coordinator's reply.
4. Done = acceptance list checked, tests green, dry run passes, GPU run cost
   posted, PR open with `Closes #N`, Status Review, summary comment
   (what changed, numbers, what the next issue needs to know).
5. Coordinator reviews the summary against section 5, merges or requests
   changes, flips Status Done, then flips newly unblocked issues to Ready.

## 8. Kick-off prompt for an implementer

```
Read CLAUDE.md, docs/plans/duplex-speech-cartridge.md and
docs/plans/duplex-speech-cartridge-agents.md. Claim issue #<N> on the
Cartridges board per CLAUDE.md, work in a worktree on branch
issue-<N>-<slug>, and build only that issue's acceptance criteria using the
interface contracts in the agents file. Run the CPU tests and the --dry-run
before any Modal GPU run; stay under the $40 cap. When done, open a PR with
"Closes #<N>", set Status to Review, and post a summary comment with the
numbers and anything the next issue needs. Do not edit files owned by
another In Progress issue without commenting there first.
```

Kick-off prompt for the coordinator:

```
Read CLAUDE.md and docs/plans/duplex-speech-cartridge-agents.md. You are
the coordinator for epic #7. Keep the board truthful: flip issues to Ready
when their blockers are Done, review summary comments against the interface
contracts in section 5, keep that section current, and confirm GPU budget
with the author before wave 3. Do not write feature code.
```
