# A cartridge for a full-duplex speech model

Status: research and technical requirements. Nothing here has run yet.
Date: 2026-10-29. The implementation contracts, waves and team protocol
live in `docs/plans/duplex-speech-cartridge-agents.md` (the brief); this
document holds the research, the requirements and the decisions.

## 0. Top-line answer

**Feasible, on two open models, trained in parallel behind one interface.**

The goal is a trained KV-cache prefix (a cartridge) that makes a full-duplex
speech model **talk as the author**: in the author's voice and prosody, in
the author's diction, in the first person, retelling the author's own
journal sentences near-verbatim. Asked "how have you been feeling lately?",
it tells the story from the most recent entry in the author's words; asked
"how have you changed over the years?", it contrasts what the author wrote
in 2024 with what the author wrote recently. It never speaks about
sexuality or anger. This is not the text cartridge's job (answer facts
*about* the author); it is a conversational self-portrait, distilled from
the journal by self-study with the base model frozen.

Of the three models first named, only **PersonaPlex 7B** is viable; of the
textless models closest to the Dessa profile, only **hertz-dev** has usable
weights. Both are targets:

| Model | Verdict | Why |
|---|---|---|
| **PersonaPlex 7B** (nvidia/personaplex-7b-v1) | **Go, primary** | Moshi architecture: a 7B pure-attention temporal transformer with a causal ring KV cache; voice sample and persona text already enter as a stream prefix, which is exactly the slot a cartridge fills. Real Fisher-trained backchannels, a text stream to evaluate through, an MLX port for Apple Silicon, open weights. No training code, so we write the trainer. |
| **hertz-dev** (si-pbc/hertz-dev) | **Go, parallel** | The closest open match to Dessa: audio in, audio out, no text anywhere, 8.4B decoder-only transformer with a plain KV cache, trained on real conversational audio, Apache-2.0. Its LM predicts discrete 15-bit audio tokens per channel, so the same distillation applies. It is a base model with no instruction following, CUDA-only, with a 256 s context: harder, and one wave behind PersonaPlex. |
| NemotronLabs VoiceChat 11B | No | Mamba-heavy hybrid backbone (4 of 56 layers have a KV cache), voice-locked TTS, ~2 min context, NeMo-only stack. |
| tsinghua-ee/ELLSA2 | No | The duplex speech backbone was never released; only a gesture generator shipped. |

What to expect, from the evidence in section 1 and the prior art:

- **Diction and content: the same method as the text cartridge.** The
  cartridge is initialised from a prefill of the author's voice clip, a
  persona and a journal digest, then trained on synthetic conversations
  whose responses are composed from the author's own passages (stage 1),
  then refined on its own rollouts (stage 2). The text cartridge closes
  about 60% of the context gap on facts; here the target is closer to the
  source text, so verbatim coverage is the measure, not recall.
- **Voice: prompt-level similarity for free, more with training.**
  PersonaPlex's audio prefix reaches 0.57–0.65 speaker similarity from a
  short clip. Stage 1 trains on the author's voice rendered by Kyutai
  Pocket TTS, so the acoustic target *is* the author; the ablation in #16
  measures how much that adds over the prefix, and a LoRA phase (#17) is
  gated on that number.
- **Evaluation is the product.** A deployed stock model is scored against a
  versioned eval set before any cartridge code exists, and every later
  change adds a point to the same W&B record (brief §3, §4, §8).
- **Nobody has done this.** No published prefix-tuning or trained-KV result
  exists on any codec-token model. Section 3.6 lists the unknowns.

## 1. Model survey

### 1.1 PersonaPlex 7B

Sources: model card, repo at commit 3428dfd, arXiv 2602.06053.

- **Architecture.** Moshi, initialised from `kyutai/moshiko-pytorch-bf16`.
  Temporal transformer: dim 4096, 32 layers, 32 heads, RoPE, attention
  context 3000 frames. Depth transformer: dim 1024, 6 layers, context 8
  *within a frame*, no cross-frame state. Mimi codec at 24 kHz, 12.5 Hz
  frames, 8 codebooks per stream, vocab 2048. 17 streams per frame: 1 agent
  text (SentencePiece, 32k) + 8 agent audio + 8 user audio. Weights 16.7 GB
  bf16. Trained at max 2048 frames (164 s); inference ring cache 3000 frames
  (240 s).
- **Conditioning = a stream prefix.** `LMGen.step_system_prompts` runs
  `_step_voice_prompt → _step_audio_silence → _step_text_prompt →
  _step_audio_silence`. Each is an ordinary `LMGen.step()`. Voice frames carry
  Mimi tokens of the clip on the agent audio streams, pad on agent text, and a
  fixed 440 Hz sine constant on the user streams. Text frames force one
  SentencePiece token per frame on agent text (`<system> … <system>`), silence
  tokens on agent audio. The prompt is masked from the training loss. Nothing
  else conditions the model: no speaker embedding, no cross-attention.
- **Shipped voice files are already prompt-tuning-shaped.** `voices.tgz` holds
  18 `.pt` files, each `{"embeddings": [N, 1, 1, 4096], "cache": [1, 17, CT]}`:
  the *pre-transformer input embeddings* of the voice frames and the token
  ring buffer. On load they are replayed through `LMGen.step_embeddings`,
  which rebuilds the KV cache. A `.pt` is not a KV cache, but
  `step_embeddings` is a public hook that accepts arbitrary 4096-d frame
  vectors.
- **Cache internals.** `RingKVCache` tensor `(2, B, heads, capacity, dh)` plus
  `end_offset` per `StreamingMultiheadAttention`. No API to inject a cache;
  streaming state is plain attributes. After 3000 frames the ring wraps, so a
  prefix of `p` frames is **evicted after (3000 − p)/12.5 seconds of dialogue**.
- **Training.** Full fine-tune of Moshi, 24.6k steps, batch 32, 6 h on
  8×A100. Losses: next-token CE on all streams; non-semantic audio codebooks
  ×0.02, padded text ×0.3. Training code not released (issues #5, #52
  unanswered). Reported: speaker similarity 0.57 (0.65 released checkpoint)
  vs Moshi 0.10; role adherence 4.48 vs 1.75. Training data includes 7,303
  real Fisher conversations (1,217 h) plus synthetic TTS dialogues.
- **Fine-tuning ecosystem.** Kyutai `moshi-finetune` (LoRA r=128 on every
  `nn.Linear`, stereo WAV + word timestamps, 1×H100 ≈ 40 GB peak at 100 s
  sequences and batch 16). Caveat verified in code: PersonaPlex's vendored
  transformer stores attention `in_proj_weight` as a raw Parameter and calls
  `F.linear` on gating weights directly, so module-swap LoRA silently misses
  those paths. Community LoRA ports exist (CharacterPlex, Hinglish) but their
  trainers are private or unverified.
- **Inference.** PyTorch, vendored `moshi` fork, CUDA graphs, WebSocket server
  and `moshi.offline` (WAV in, WAV out). A100 80 GB tested; ~17 GB steady,
  32–34 GB load spike. Turn-taking latency 0.17 s. Kyutai's upstream `moshi`
  has an MLX port that runs on Apple Silicon; PersonaPlex's fork does not
  ship one.

### 1.2 NemotronLabs VoiceChat 11B

Sources: model card (`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`, Aug 2026),
arXiv 2609.21967, `NVIDIA-NeMo/Speech` branch `nemotron-labs-voicechat`.

- Streaming FastConformer encoder (600M, 80 ms frames) → Nemotron-Nano-9B-v2
  backbone (56 layers: 27 Mamba2, 4 attention, 25 MLP; hidden 4480) with
  three heads (agent text, function channel, RNN-T user transcript) →
  separate 977M "VoiceChat-TTS" (Gemma-3-based codec LM, 31 RVQ codebooks,
  12.5 Hz).
- Truly full-duplex (steps every 80 ms, pad tokens model silence) and has
  live tool calling. Training code is released (NeMo, FSDP2), no PEFT.
- **Why not:** only 4 of 56 layers have a KV cache; the rest carry Mamba
  recurrent state (fp32), which is not a prefix we can train the way we train
  keys and values (our Gated-DeltaNet initial states are the closest analogue,
  but here they would be the whole cartridge). The TTS is voice-locked by
  design: named speaker latents are pre-baked and the audio-prompt projection
  is randomised so other reference clips produce incoherent conditioning.
  Context beyond ~2 min "may not be retained reliably" and long sessions
  degrade. The system prompt is applied only to the first inference call.
  Weights are a NeMo checkpoint, served by a custom vLLM fork inside a NIM.

### 1.3 tsinghua-ee/ELLSA2

Sources: HF repo (16 Oct 2026), arXiv 2610.21452, `bytedance/SALMONN` branch
`ELLSA2`.

- Paper: "Full-Duplex Interaction with Speech, Expressions and Gestures".
  10.78B total, 776M trainable (LoRA on a Llama-3-8B-class backbone inherited
  from SALMONN-omni, Mamba streaming encoder, CosyVoice2-0.5B synthesiser).
  80 ms time-multiplexed stream; voice fixed by the fine-tuned synthesiser.
- **Why not:** the released checkpoint is the standalone gesture/expression
  generator (Qwen2.5-0.5B, 2.6 GB) plus an ASR initialiser and RVQ motion
  tokenizers. The speech backbone and synthesiser weights are "upon
  acceptance" and SALMONN-omni has never shipped. No speech training code.

### 1.4 The Dessa profile: textless, audio-native, local

Dessa (unreleased) is described as audio-in audio-out with no text stream,
full-duplex, trained on real conversation (so it backchannels and interrupts
naturally), carrying paralinguistics, faster than real time, and small
enough for a MacBook Pro M4 Max.

Dessa is itself row 1 of the Awesome-Full-Duplex-SDM list (2026, end-to-end,
no weights). The list has 58 model entries; 29 are end-to-end and 10 of those
ship code and weights. Only two textless models on the list have weights
(dGSLM, and RTTL-DG unverified), and the closest open match to Dessa is not on
the list at all.

| Rank | Model | Textless | Backbone | Real-conversation data | Weights / license | Local on Mac | Main gap vs Dessa |
|---|---|---|---|---|---|---|---|
| 1 | **hertz-dev** (Standard Intelligence, Nov 2024; not on the list) | Yes. Audio in, audio out; no text at inference. | 40-layer 8.4B decoder-only transformer, 2048-step context over an 8 Hz VAE codec (32-dim latents, 1 kbps) ≈ 4 min. Standard causal KV cache. Duplex = two projection heads, one per channel. | Claimed 20M h of "real-world conversational audio"; composition undisclosed. | Yes, Apache-2.0. | No. CUDA 12.1 only; no MPS or MLX port. | A **base model**: no instruction tuning, no question answering. One checkpoint was initialised from a 2T-token text LM. No backchannel or paralinguistic evaluation published. |
| 2 | **dGSLM** (Meta, 2022; on the list) | Yes. | Dual-tower transformer with cross-attention, 6 layers, d=512 (tens of millions of params). HuBERT 50 Hz units + duration. | Yes, only: 2,000 h of two-channel Fisher calls. Laughter, backchannels and overlap emerge. | Yes (fairseq, archived Mar 2026), MIT. | Trivially small, but no streaming inference exists. | 2022 quality: little semantic coherence. Not a single decoder-only stack, so a KV prefix is two caches plus cross-attention. |
| 3 | **Moshi** (Kyutai, 2024; on the list) | No. Text "inner monologue" stream next to audio. | Helium 7B temporal + depth transformer, Mimi 12.5 Hz. | Partly: Fisher 2,000 h in the multi-stream stage, then 20K h synthetic dialogue. | Yes, CC-BY-4.0. | Yes. MLX q4 / int8 / bf16, tested on M3. | Text-trained backbone; laughter not modelled. |
| 4 | **PersonaPlex** (NVIDIA, 2026; on the list) | No. Same streams as Moshi. | Moshi 7B fine-tune. | Yes: 1,217 h real Fisher plus 2,250 h synthetic. Released checkpoint shows "significantly increased backchannel frequency". | Yes, NVIDIA Open Model License. | Yes. Community MLX 4-bit port (5.3 GB, RTF 0.87 on M2 Max). | Text stream; restrictive license; laughter not modelled. |
| 5 | **RTTL-DG** (2025; on the list) | Yes. | Encoder-decoder, 8+8 layers, d=768; 160 ms decisions; HuBERT units. | 372 h Switchboard + 5.8K h synthetic. Backchannels, fillers and laughter explicit. | Unverified; repo marked deprecated, successor RelayS2S is text-based. | Small, but no checkpoint confirmed. | Encoder-decoder, not a causal decoder; weights unconfirmed. |

Everything else with weights on the list is text-assisted (a Qwen, GLM,
Step-Audio or LLaMA backbone with speech tokens interleaved), and four of the
2026 entries that might qualify (DuplexSLA, SteerDuplex, Diffuplex, Seeduplex)
have no weights yet. hertz-dev is the only open model that matches Dessa's
defining property of learning language from speech alone at a usable scale.

**What a cartridge on hertz-dev would look like, and why it runs one wave
behind PersonaPlex.** Structurally it is the easiest of all: one decoder-only transformer,
one causal KV cache, no depth transformer, no text stream. A cartridge is K/V
for `p` of the 2048 steps per layer (40 layers × 2 × p × d). Initialisation is
a prefill of your own voice on the agent channel, which is the only way the
model takes a voice anyway. The problems are what the model *is*:

1. **No instruction following.** hertz-dev continues conversations; it does
   not answer questions. That suits the retelling target (continue the
   conversation as the author, about the author's life) better than a Q&A
   target would, but the question must be in the audio and the model must
   take the turn on its own; the timed test in the brief's §8 measures
   exactly that.
2. **Textless self-study is unproven.** The teacher is hertz-dev with the
   passages *read aloud by Pocket TTS* inside its 4-minute window, followed
   by the question; the student is the cartridge. The code inspection
   settled the loss: hertz-ar's LM emits a discrete 15-bit FSQ token per
   channel (so KL distillation applies) and its resynthesizer a Gaussian
   mixture over 32-d latents (so the SFT loss adds a mixture NLL).
3. **No evaluation path without text.** Every memory metric we have goes
   through transcripts. We can still transcribe the output with Phonon-2, but the model's own
   knowledge is not readable the way a text stream is.
4. **No Mac build.** Dessa's local-run promise is met today by Moshi and
   PersonaPlex through MLX, not by hertz-dev.

Decision: both. PersonaPlex is the primary target because it is the most
Dessa-like open model that also reads text, carries real Fisher-trained
backchannels and runs on Apple Silicon; hertz-dev is the parallel target
because it is the only open model that learned language from speech alone,
and the brief's backend interface (§4b) lets one pipeline train and
evaluate both. Where PersonaPlex reads persona and digest as text, hertz-dev
hears them spoken by Pocket TTS in the author's voice; where PersonaPlex is
distilled on its text stream, hertz-dev is distilled on its discrete
audio-token stream. hertz-dev's cartridge work runs one wave behind
PersonaPlex's so the first backend's lessons carry over.

## 2. Mapping the cartridge onto the two backends

| Our text cartridge (Qwen3.8-27B) | PersonaPlex | hertz-dev |
|---|---|---|
| 16 full-attention layers: K/V for `p` prefix tokens, post-RoPE at 0..p−1 | 32 temporal layers: K/V for `p` prefix *frames* (12.5 Hz), one cache group `temporal` | 32 LM layers plus the 8 + 4 split resynthesizer layers: two cache groups, `lm` and `resyn`, `p` frames at 8 Hz |
| 48 Gated-DeltaNet initial states | None; the depth transformer has no cross-frame state | None; the resynthesizer's state is its own KV cache, held in the `resyn` group |
| Positions are text tokens | `p = 1024` ≈ 82 s of prompt time | `p = 512` ≈ 64 s; context is only 2048 frames (256 s) |
| Init: prefill the first `p` corpus tokens | Prefill of the hybrid prompt: voice clip on agent audio, sine on user, `<system>` persona and digest forced one token per frame | Prefill of the voice clip followed by Pocket TTS speech of persona and digest on the agent channel, user silent, ≤ 720 frames |
| Teacher: same model, full journal in context | Same model with the source passages in its `<system>` prompt (the journal does not fit: 3000 frames) | Same model with the passages read aloud on the agent channel before the question |
| Targets: top-20 logprobs on answer tokens | Primary: the author's-voice render (Mimi tokens) and its aligned text; regulariser: top-20 on the text stream | Primary: the author's-voice render (FSQ tokens + latents); regulariser: top-20 on the agent FSQ stream |
| Loss: KL per token | SFT: text CE (pad 0.3), semantic codebook 1, acoustic 0.02; plus λ·KL text stream | SFT: FSQ CE 1, GMM NLL on latents 1, user channel 0; plus λ·KL FSQ stream |
| Student forward: HF `DynamicCache` with read-only layers | Patched `StreamingMultiheadAttention`: cartridge K/V prepended, RoPE offset `p`, mask extended | Strip `@torch.no_grad` on the two forwards; prepend both groups; pass an explicit causal mask (upstream drops it for multi-token chunks) |
| Serving: vLLM prefix-cache swap | Pinned region in `RingKVCache`; the fork's own WebSocket server | Cloned prefix caches; `inference_server.py`; explicit window re-prefill at `context_frames − p` |
| Size: 0.3 GB at p = 2048 | 0.54 GB at p = 1024 | ≈ 0.5 GB at p = 512 across both groups |

Easier than the text case: no linear-attention states, and both base models
already take their only conditioning from a prefix. Harder: the teachers are
weaker and must read passages from the prompt; PersonaPlex's ring cache
evicts the prefix unless pinned; hertz-dev has no instruction following and
a 256 s context; and voice and content share one prefix, so they are
trained jointly (independently trained cartridges collapse when mixed,
arXiv 2606.04557).

## 3. Technical requirements

### 3.1 Functional

- **F1. Drop-in prefix.** A cartridge is a directory (`cartridge.safetensors`
  + `cartridge.json`) loadable by the backend's own server in place of its
  prompt. The step-0 (prefill-only) cartridge reproduces the stock prompt's
  logits on a fixed test dialogue to bf16 round-off.
- **F2. Talks as the author.** First person; responses retold from the
  journal with ≥ 0.6 verbatim 4-gram coverage of the transcript on story
  prompts; chronology in written order on change prompts; no spoken
  citations.
- **F3. Voice.** Speaker similarity to held-out clips ≥ the stock prompt
  baseline (≈ 0.6) after stage 1; ≥ 0.7 after the gated LoRA phase.
- **F4. Duplex behaviour preserved.** Turn-taking latency, interruption
  latency and spurious-onset rate within 10% of the stock baseline.
- **F5. Persistence.** The cartridge stays resident for the whole session,
  independent of dialogue length.
- **F6. Topic exclusion.** Zero responses about sexuality or anger on the
  baited prompts; an in-character decline instead.
- **F7. Evaluable as deployed.** Every number above is measured by
  `voice eval` against a served endpoint, logged to W&B next to the
  backend's stock baseline (brief §4, §8).

### 3.2 Data

- **D1. The author's voice.** `data/voice/josh-thoughts-1.m4a` (~9.8 min,
  git-ignored) → 24 kHz mono WAV at −24 LUFS, split 80/10/10: TTS prompt
  and cartridge init / Pocket TTS fine-tune if needed / speaker-similarity
  eval. A 30 s prompt segment goes into the eval-set artifact so every
  backend's stock prompt is the same voice.
- **D2. Author-voice TTS.** Kyutai Pocket TTS cloned from the prompt split
  renders every agent-side answer. Its similarity to the eval split is the
  voice ceiling; if < 0.7, fine-tune it on the training split first.
- **D3. Passages.** The journal segmented into dated passages with pages.
  The Qwen Teacher labels each `sexuality` / `anger` / `other` plus a
  `borderline` flag; only `other` and not borderline survive, in every
  artifact. Held-out passages span every journal year and are disjoint
  from training.
- **D4. Questions.** Conversational, generated from the passages: story
  (one passage), change (passages ≥ 1 year apart), idea (an opinion), plus
  the author's own questions. Psychology-test items are dropped.
- **D5. Responses.** Composed by the Qwen Teacher under hard constraints
  and validated mechanically: ≥ 80% of tokens verbatim from the source
  passages (4-gram coverage), first person, spoken register of 2–6
  sentences, dates in words, no citations in the spoken text, topic
  classifier pass. ~3,000 train, ~200 held-out.
- **D6. Eval set first.** A versioned W&B artifact `voice-evalset` with the
  held-out prompts and reference responses, 20 baited prompts, 60 scripted
  2-minute user tracks for the timed test, Full-Duplex-Bench audio, and the
  voice eval split, in `smoke` (10 prompts) and `full` sizes. Built before
  any training data (#22).
- **D7. Digest.** A ≤ 1,500-token journal digest from `other` passages for
  the cartridge init (PersonaPlex as text; hertz-dev as ≤ 60 s of speech).
- **D8. Transcription.** Phonon-2 (`FermionResearch/Phonon-2`) for every
  transcript and every word timestamp, through one wrapper, with
  Parakeet-TDT per-word confidence.

### 3.3 Model and code

- **M1. Backend protocol.** One `DuplexBackend` protocol and registry
  (brief §4b): `encode_dialogue`, `decode_agent`, `build_prefix`,
  `cartridge_from_prefill`, `forward`, `sft_loss`, `distill_loss`,
  `rollout`, `serve`; shared `Frames`, `Cartridge`, `Outputs` types; a
  generic trainer, eval and report that only ever call `backends.get`.
- **M2. Cartridge.** Named cache groups of per-layer K/V parameters, fp32
  master and bf16 in forward, first frame frozen as the sink; `save`/`load`
  with `cartridge.json` metadata (backend, spec hash, fork commit, p,
  voice and digest hashes, stage, step).
- **M3. Training forward.** Full-sequence, differentiable with respect to
  the cartridge only, model frozen, gradient checkpointing; the
  backend-specific patches of section 2.
- **M4. Loss.** `backend.sft_loss` on the agent streams of the
  author's-voice frames plus `λ · backend.distill_loss` to the
  passage-prompted teacher's stored top-20 on the discrete agent stream;
  λ swept over {0, 0.5}; prefix excluded.
- **M5. Stage 2.** On-policy: rollouts from the pinned prefix, reverse KL
  on the discrete agent stream to the passage-prompted teacher, audio
  anchored by Pocket TTS renders of the sampled text, anchor mix
  0.25 → 0.5 as in `onpolicy.py`.
- **M6. Pinned prefix and serving.** PersonaPlex: a pinned region in
  `RingKVCache` (section 4.4). hertz-dev: cloned prefix caches with an
  explicit window re-prefill. `voice deploy --system <cartridge>` serves
  either on Modal.
- **M7. Isolation.** Everything lives in the `voice/` and `shared/`
  workspace members; nothing in `src/cartridges/` changes; `shared/` holds
  adapted copies of the Qwen Teacher client, tracking, judge, target shards
  and optimizer (section 4.1).
- **M8. LoRA (gated).** Only if the ablation (#16) leaves speaker similarity
  < 0.65: explicit LoRA wrappers for the fork's `in_proj_weight`, gating and
  depformer stacked weights, merged, then the cartridge retrained.

### 3.4 Infrastructure

- **I1. GPU.** One H200 per job. PersonaPlex: 16.7 GB weights, ≈ 10–20 GB
  activations at micro-batch 2 with checkpointing, cartridge + Adam ≈ 2 GB
  at p = 1024. hertz-dev: ≈ 17 GB weights bf16 for LM + resynthesizer,
  `torch==2.5.1`, CUDA only. Modal apps `voice-train` and `voice-serve`.
- **I2. Storage.** Volumes `voice-data` (passages, dialogues, per-backend
  frames, targets) and `voice-voice` (the author's clips). Targets ≈ 0.4 GB
  per backend for 3,000 dialogues.
- **I3. Tracking.** W&B project `voice` under the brief's §4 conventions:
  one group per experiment with a baseline pointer, fixed run names and
  summary metric names, artifacts `voice-evalset`, `voice-dialogues`,
  `voice-targets`, `voice-cartridge`, `voice-report`.
- **I4. Audio tooling.** Mimi and hertz-codec (shipped with the models);
  **Phonon-2** (English ASR derived from Parakeet-TDT-0.6B-v3, CC-BY-4.0,
  164 MB; `phonon transcribe --json` gives per-word start and end times;
  MLX locally, CUDA Docker on Modal); Kyutai Pocket TTS; WavLM-TDNN (or
  `speechbrain` ECAPA) for speaker similarity; Full-Duplex-Bench v1 and
  v1.5 harnesses; UTMOS.

### 3.5 Evaluation

The protocol is the brief's §8, modelled on Thinking Machines' interaction
model evaluation: five axes, a frontier plot of fidelity against
interaction quality and against latency, five side-by-sides, and every
checkpoint reported next to its backend's stock baseline. Headline measure
per axis:

| Axis | Headline measure | Stock baseline | Phase 1 target |
|---|---|---|---|
| Responsiveness | FDB v1 turn-taking latency (s); spurious onsets | PersonaPlex 0.17 s; 27.5% | within 10%; ≤ baseline |
| Interaction quality | FDB v1.5 scenario average; CueSpeak-style timed macro-accuracy | measured | within 10%; reported |
| Fidelity to the author | verbatim 4-gram coverage; `voice-v0` judge; first-person rate; topic violations | ≈ 0; measured; measured; 0 | ≥ 0.6 (story); reported with CI; ≥ 0.95; 0 |
| Voice | WavLM-TDNN speaker similarity; drift minute 1 vs 4; UTMOS | ≈ 0.6 (prompt) | ≥ baseline; no drift; UTMOS not down > 0.2 |
| Naturalness | paired human preference, 30 prompts × 3 raters, plus the author's 1–5 | n/a | reported with binomial CI |

Transcripts come from Phonon-2; the judge is the existing Qwen Teacher
with a new `voice-v0` rubric that sees the matching passages. Two eval
sizes: `smoke` (10 prompts, under 10 minutes, every PR and every training
validation step) and `full`.

### 3.6 Risks and unknowns

1. **Long passage prompts are out of distribution** for both teachers.
   #11 measures answer quality against passage length before any targets
   are generated; if a backend cannot use passages at any length, its KL
   regulariser drops to 0 and its objective is pure SFT.
2. **Prefix attention share.** A long trained prefix takes a growing share
   of softmax mass (PrefixMemory-Tuning, arXiv 2506.13674); the symptom on
   speech is degraded user-audio tracking. F4 catches it; p = 1024 / 512
   are the safer first sizes.
3. **Acoustic drift.** A drifted prefix could push acoustic codebooks or
   latents into junk. Low acoustic weight (PersonaPlex 0.02) and UTMOS at
   every validation step guard it.
4. **Voice via prefix saturates.** Expected; the ablation (#16) measures
   what training on the author's voice adds, and LoRA (#17) is the gated
   fallback, with its own quality-loss risk on noisy clips.
5. **hertz-dev is a base model.** It continues conversations rather than
   answering; the conversational-retelling target suits that, but there is
   no text stream to read, CUDA-only code with hard-coded autocast, a
   broken streaming codec path, and a 2048-frame cache that raises on
   overflow. It is one wave behind for that reason.
6. **No upstream trainer** for either model; we own both training forwards
   on pinned commits (PersonaPlex fork `3428dfd`, hertz-dev `076c275`).
7. **Two backends double the GPU bill** (section 4.6); the coordinator
   confirms budget before wave 4.
8. **Personal data.** The cartridge encodes the author's voice and journal.
   Nothing under `data/` is committed or uploaded publicly; the eval split
   lives in a restricted W&B artifact. Licenses: NVIDIA Open Model License
   (PersonaPlex), Apache-2.0 (hertz-dev), CC-BY-4.0 (Phonon-2).

## 4. Implementation spec

The authoritative contracts, conventions and per-issue guidance are in the
brief (§4, §4b, §5, §6); this section records the shape.

### 4.1 Package layout

Two uv workspace members, isolated from the cartridge package:

```
voice/                      # member "voice": the whole speech project
  src/voice/
    backends/               # base.py, registry.py, personaplex.py, hertz.py
    passages.py data.py tts.py evalset.py teacher.py
    train.py evals.py report.py client.py serve.py deploy.py cli.py
  tests/
shared/                     # member "shared": copied-out components
  src/shared/
    teacher.py tracking.py judge.py targets.py optim.py
  tests/
src/cartridges/             # untouched
```

### 4.2 Data formats

```
passages.jsonl       {passage_id, entry_date (YYYY-MM-DD), pages [int], text, topic, borderline}
dialogue.jsonl       {id, family, split, question, response_text, passage_ids [str], coverage,
                      user_wav, agent_wav, words [{w, start_s, end_s, conf}]}
frames/<backend>/<id>.safetensors   PersonaPlex: user_audio [T,8], agent_text [T], agent_audio [T,8]
                                    hertz: user/agent FSQ tokens [T], user/agent mu latents [T,32]
targets (parquet)    id, frame, stream (name), topk_ids int32[20], topk_logprobs float32[20]
cartridge.json       {backend, spec_hash, base_model, fork_commit, p, num_frozen_tokens,
                      voice_sha, persona, digest_sha, stage, step}
cartridge.safetensors  <group>.keys.<layer>, <group>.values.<layer>, fp32
```

### 4.3 Training forward per backend

PersonaPlex, per temporal layer: rotate `q` and `k` at positions `p..p+T`,
concatenate the cartridge's `K`/`V` in front, extend the causal mask by `p`
fully visible columns, run SDPA; input embedding and depth transformer
untouched. hertz-dev: remove the `@torch.no_grad` decorators on
`HertzDevModel.forward` and `TransformerVAE.forward`, prepend both cache
groups, and pass an explicit causal mask because `GQA._attend` sets
`is_causal=False` whenever `q_len != k_len`; encode latents with
`get_parameters=True` and use `mu`. Both: micro-batch 2, accumulation to
16, bf16, Adam with relative LR 5e-2 (sweep 1e-2 … 1e-1), warmup 5, early
stopping on held-out SFT loss with the smoke eval as the promotion
criterion.

### 4.4 Pinned prefix and serving

PersonaPlex: `RingKVCache(pinned=p)` keeps slots `0..p-1` fixed and rings
over `p..capacity-1`; attention always sees prefix + window; capacity
stays 3000, leaving 158 s of rolling dialogue at p = 1024. hertz-dev: the
cache is a fixed tensor that raises past its length, so `serve` clones the
prefix caches and, at `context_frames − p`, drops the oldest dialogue
frames and re-prefills behind the pinned prefix. `voice deploy --backend X
--system stock|<artifact>` serves either as Modal app `voice-serve`.

### 4.5 Waves and gates

Evaluation-driven, inference first (brief §3 has the full table):

| Wave | Issues | Exit criterion |
|---|---|---|
| 0 | #19 workspace, protocol, registry, generic trainer | root tests unchanged; dry run on a fake backend |
| 1a | #20 PersonaPlex stock inference + deploy + client · #21 hertz-dev stock inference + deploy · #22 eval set | two stock endpoints; `voice-evalset` v1 |
| 1b | #9 harness and report against the endpoints | **two baseline rows in W&B**; smoke eval < 10 min |
| 2 | #8 PersonaPlex cartridge half · #10 training data · #11 teacher study | step-0 parity; dialogues on the volume |
| 3 | #18 hertz-dev cartridge half · #12 pinned cache + cartridge deploy · #13 targets | a cartridge served and re-evaluated |
| 4 | #14 stage 1, deploy best, full eval | cartridge rows next to baselines; frontier plot |
| 5 | #15 stage 2 · #16 voice ablation | same |
| 6 | #17 LoRA, gated on #16 | same |

### 4.6 Cost

Modal H200 at about $4.54/h. Per backend: labels and targets ≈ 2 GPU-h,
stage 1 with a four-point LR sweep ≈ 15 GPU-h, stage 2 ≈ 4 GPU-h, evals
(baseline, smoke at each validation step, full per checkpoint) ≈ 6 GPU-h,
≈ $120–150. Both backends plus the ablation ≈ $300. Above the cap in
`docs/plans/next-experiments.md`; the coordinator confirms budget before
wave 4. LoRA, if opened, ≈ 6 h on one H100 per backend.

### 4.7 Acceptance criteria for "done" (phase 1)

- [ ] Parity tests for both backends pass on stored fixtures.
- [ ] `voice-evalset` v1 and two stock baseline rows exist in W&B.
- [ ] A stage 2 `voice-cartridge` artifact per backend, each with a
      full-eval run in its experiment group and every §3.5 row filled with
      CIs next to the baseline.
- [ ] The report: table, two frontier plots, five side-by-sides per
      backend, played to the author.
- [ ] A 3-minute recorded conversation per backend where the model, with no
      text prompt and no voice file, retells three journal stories in the
      author's voice and survives one interruption.
- [ ] README section "Speech cartridge" with the recipe tables in the same
      format as the text recipe.

## 5. Decisions taken in this document

- Target behaviour: the author's conversational self-portrait (first
  person, near-verbatim journal retelling, the author's voice), not
  question answering about the author. Sexuality and anger excluded from
  every artifact.
- Backends: PersonaPlex 7B (primary, fork commit 3428dfd) and hertz-dev
  (parallel, one wave behind, commit 076c275), behind one `DuplexBackend`
  protocol. Not VoiceChat 11B, not ELLSA2.
- Cartridge form: KV-level, named cache groups per backend; one cartridge
  for voice and content, trained jointly; never compose two.
- Teacher: two-tier. The Qwen Teacher with the full journal writes passages
  and responses as text; each backend's own model with the passages in
  context supplies the distributions used for distillation.
- Primary audio target: the author's voice via Kyutai Pocket TTS, from
  stage 1. LoRA only if the ablation says the prefix is not enough.
- Transcription: Phonon-2 everywhere.
- Order: eval set and stock inference first, baselines as the wave-1 exit,
  then cartridge, training, redeploy, re-eval; W&B as the record.
- Layout: `voice/` and `shared/` workspace members; `src/cartridges/`
  untouched.
- First sizes: p = 1024 (PersonaPlex), p = 512 (hertz-dev).

## 6. Open questions, each owned by an issue

1. How long a passage prompt can each backend read before quality or
   duplex behaviour drops? (#11)
2. Does Pocket TTS reach ≥ 0.7 similarity to the author on its own? If not,
   fine-tune it before rendering. (#22)
3. Does the packaged Phonon-2 CLI expose per-word confidence, or must it
   come from NeMo decoding of the same lineage? (#9)
4. Does a trained prefix add speaker similarity over the raw prefix on
   either backend? (#16)
5. Does hertz-dev's resynthesizer tolerate a trained `resyn` prefix, or
   should only the `lm` group train? (#18)

## 7. Sources

- PersonaPlex: https://huggingface.co/nvidia/personaplex-7b-v1 ·
  https://github.com/NVIDIA/personaplex · https://arxiv.org/abs/2602.06053
- Moshi: https://arxiv.org/abs/2410.00037 ·
  https://github.com/kyutai-labs/moshi-finetune
- VoiceChat 11B: https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B ·
  https://arxiv.org/abs/2609.21967 ·
  https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat
- ELLSA 2: https://huggingface.co/tsinghua-ee/ELLSA2 ·
  https://arxiv.org/abs/2610.21452 · SALMONN-omni https://arxiv.org/abs/2505.17060
- Cartridges: https://arxiv.org/abs/2506.06266 · Cartridges at Scale
  https://arxiv.org/abs/2606.04557 · Learned Structure in Cartridges
  https://arxiv.org/abs/2508.17032
- Voice prompting and LoRA evidence: VALL-E 2 https://arxiv.org/abs/2406.05370 ·
  Long-Prompt TTS https://arxiv.org/abs/2610.19774 · LoRA on NeuTTS
  https://arxiv.org/abs/2603.10904 · F-Actor https://arxiv.org/abs/2601.11329
- Prefix failure modes: PrefixMemory-Tuning https://arxiv.org/abs/2506.13674 ·
  spurious onsets https://arxiv.org/abs/2609.13445
- Memory in speech models: MoshiRAG https://arxiv.org/abs/2604.12928 ·
  VoiceMem https://arxiv.org/abs/2608.26005 · "Where Should a Document Live"
  https://arxiv.org/abs/2609.17346
- Transcription: Phonon-2 https://huggingface.co/FermionResearch/Phonon-2
- Full-duplex model list: https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM
- Dessa-profile candidates: hertz-dev https://github.com/Standard-Intelligence/hertz-dev ·
  https://huggingface.co/si-pbc/hertz-dev · dGSLM https://arxiv.org/abs/2203.16502 ·
  RTTL-DG https://arxiv.org/abs/2501.04877 · PersonaPlex MLX port
  https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit
