# A cartridge for a full-duplex speech model

Status: research and technical requirements. Nothing here has run yet.
Date: 2026-10-29.

## 0. Top-line answer

**Feasible, on one of the three named models.** NVIDIA PersonaPlex (7B, Moshi
architecture) conditions on a *voice sample* and a *persona text* by placing
both as ordinary frames at the start of its single token stream. They sit in
the same causal KV cache as the dialogue. That prefix is exactly the slot our
cartridge fills: we can replace it with a trained KV prefix, initialised from
a prefill of your own voice clip plus a persona text, and train it with
self-study the way we train the journal cartridge today.

The other two named models are not viable targets:

| Model | Verdict | Reason |
|---|---|---|
| **PersonaPlex 7B** (nvidia/personaplex-7b-v1) | **Go** | Pure-attention 7B temporal transformer, standard causal ring KV cache, voice + persona already supplied as a stream prefix, open weights (NVIDIA Open Model License), PyTorch inference code, ~17 GB VRAM steady. No training code, so we write the trainer. |
| NemotronLabs VoiceChat 11B | No | Backbone is Nemotron-Nano-9B (NemotronH hybrid: 27 Mamba2 / 4 attention layers), so a "cache" is mostly SSM state, not KV. The TTS head is *voice-locked* (the audio-prompt projection was randomised on release) and the model keeps only ~2 min of context. NeMo-only stack, no LoRA/PEFT. |
| tsinghua-ee/ELLSA2 | No | The duplex speech backbone (SALMONN-omni lineage) is **not released**. The repo ships only a 0.5B gesture/expression generator, an ASR initialiser and motion tokenizers. Nothing to train a cache on. |

**On the Dessa profile** (textless, audio-only, trained on real conversation,
runs on a MacBook): no open model on the Awesome-Full-Duplex-SDM list matches
it with released weights at usable quality. The closest open equivalent is
**hertz-dev** (Standard Intelligence, 8.4B, audio-only, Apache-2.0), which is
not on the list. A KV cache is structurally trainable on it, but it is a base
model with no instruction following, no text stream to evaluate through, and
no Mac build. Section 1.4 ranks the candidates and section 5 records the
decision: PersonaPlex for the journal cartridge, hertz-dev as an optional
voice-only pilot.

What the cartridge can and cannot do, given what we know today:

- **Journal memory: same method, same expected ceiling.** The self-study loop
  (Teacher with evidence in context, student with cartridge, per-token KL,
  then on-policy distillation) transfers directly. Expect the same class of
  result as the text cartridge: strong style and theme transfer, partial fact
  recall (we close ~60% of the context gap on text; speech will be at or below
  that because the 7B Helium backbone is weaker than Qwen3.8-27B).
- **Voice: a prefix gets you to PersonaPlex's own level, not beyond.**
  PersonaPlex's audio prefix reaches 0.57–0.65 speaker similarity (WavLM-TDNN)
  from a short clip. Across the TTS literature, longer reference prompts
  saturate speaker similarity and improve only prosody; LoRA fine-tuning adds
  roughly +0.1 similarity but can degrade audio quality. So: the cartridge
  carries your voice *at prompt quality for free*; a closer match needs a small
  LoRA on PersonaPlex trained on your voice clips, with the cartridge trained
  on top of the merged weights. We treat that as a gated phase 2.
- **Nobody has done this.** There is no published prefix-tuning or trained-KV
  result on any codec-token LM. We would be first, and we must budget for
  unknowns (section 3.6).

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

**What a cartridge on hertz-dev would look like, and why it is not the main
line.** Structurally it is the easiest of all: one decoder-only transformer,
one causal KV cache, no depth transformer, no text stream. A cartridge is K/V
for `p` of the 2048 steps per layer (40 layers × 2 × p × d). Initialisation is
a prefill of your own voice on the agent channel, which is the only way the
model takes a voice anyway. The problems are what the model *is*:

1. **No instruction following.** hertz-dev continues conversations; it does
   not answer questions. "Compress my journal" becomes "continue
   conversations as me, about my life". That is a legitimate goal, and
   arguably closer to Dessa, but it is not the question-answering target our
   evals and task DB are built for.
2. **Textless self-study is possible but unproven.** The teacher is hertz-dev
   with a journal chunk *read aloud by your TTS* inside its 4-minute window,
   followed by the other speaker's question; the student is the cartridge.
   Targets are the teacher's per-step predictions on the agent channel. We
   would need to confirm whether hertz-ar's head is a discrete distribution or
   a continuous-latent regression; the loss changes accordingly (KL vs
   likelihood on latents), and our pipeline has no continuous-target path.
3. **No evaluation path without text.** Every memory metric we have goes
   through transcripts. We can still Whisper the output, but the model's own
   knowledge is not readable the way a text stream is.
4. **No Mac build.** Dessa's local-run promise is met today by Moshi and
   PersonaPlex through MLX, not by hertz-dev.

Decision: PersonaPlex stays the target for the journal cartridge, because it
is the most Dessa-like open model that *also* answers questions, carries real
Fisher-trained backchannels, and runs on Apple Silicon. hertz-dev is the right
target for a *voice-and-style* cartridge in the Dessa sense (sound like you,
interrupt and backchannel like you, with no text anywhere), and is proposed
as an optional side pilot (phase 0b in section 4.5) whose only question is:
does a trained KV prefix hold a speaker identity and style across the full
4-minute window better than the raw audio prefix does? If that pilot works,
the textless route reopens for memory with a textless teacher.

## 2. Mapping our cartridge onto PersonaPlex

| Our text cartridge (Qwen3.8-27B) | PersonaPlex cartridge |
|---|---|
| 16 full-attention layers: K/V for `p` prefix tokens, post-RoPE at 0..p−1 | 32 temporal-transformer layers: K/V for `p` prefix *frames*, post-RoPE at 0..p−1 |
| 48 Gated-DeltaNet layers: conv + recurrent initial state | None. The depth transformer has no cross-frame state, so the cartridge is temporal-only. Simpler than today. |
| Positions are text tokens | Positions are 80 ms frames. `p = 1024` ≈ 82 s of "prompt time", `p = 2048` ≈ 164 s. |
| `init_from_text`: prefill the first `p` tokens of the context block | Prefill the hybrid prompt: your voice clip (Mimi tokens on agent audio, sine on user) + `<system>` persona text + a journal digest forced at one token per frame. The result is byte-for-byte what PersonaPlex computes today for a voice + text prompt, so step 0 reproduces the stock model. |
| Teacher: same model, full journal in context (205k tokens) | PersonaPlex cannot hold the journal (3000 frames ≈ 2.9k text tokens including the voice clip). Teacher is the **same model with per-question evidence** in its text prompt, where the evidence is written by our existing Qwen3.8 Teacher that has the full journal. This is the paper's chunked self-study, with the chunk chosen by a stronger text model. |
| Targets: Teacher top-20 logprobs on answer tokens | Targets: PersonaPlex-teacher full logits on **agent text + 8 agent audio codebooks**, every frame of its rollout. User-audio streams are inputs, never targets. |
| Student forward: HF `DynamicCache` with `_ReadOnlyKVLayer` | Student forward: non-streaming `LMModel.forward` over the full dialogue with each `StreamingMultiheadAttention` patched to concatenate cartridge K/V before its own K/V and offset RoPE and the causal mask by `p`. |
| Stage 2: student samples, Teacher scores per token | Same. Student rolls out with `LMGen` from the cartridge; Teacher re-scores the rollout with evidence in context. |
| Serving: vLLM prefix cache swap | Serving: patched `RingKVCache` with a **pinned prefix** region (section 4.4). |
| Size: 0.3 GB at p=2048 | 2 × 32 layers × p × 4096 × 2 bytes: **0.54 GB at p=1024, 1.07 GB at p=2048**. |

Two things make the speech case *easier* than the text case: no linear
attention states, and the base model already uses a prefix as its only
conditioning channel, so a trained prefix is in-distribution in form.

Three things make it harder:

1. **The teacher is weaker and two-tiered.** PersonaPlex is a 7B Helium that
   must read evidence from a forced text prompt. Our Qwen Teacher has to
   pre-digest the journal per question.
2. **The ring cache evicts the prefix.** With `p=2048`, the cartridge starts
   being overwritten after 76 s of conversation. We must pin it.
3. **Two objectives in one prefix.** Voice (acoustic codebooks) and memory
   (text + semantic codebook) share the same `p` frames. Cartridges at Scale
   (arXiv 2606.04557) reports that *mixing independently trained cartridges
   collapses to chance*, so voice and journal are trained jointly in one
   cartridge, never composed.

## 3. Technical requirements

### 3.1 Functional

- **F1. Drop-in prefix.** A cartridge is a directory `cartridge.safetensors` +
  `cartridge.json`, loadable by a patched PersonaPlex server in place of
  `--voice-prompt` and `--text-prompt`. The server behaviour with the
  step-0 (prefill-only) cartridge must be indistinguishable from the stock
  voice + text prompt: identical logits on a fixed test dialogue (tolerance
  bf16 round-off).
- **F2. Memory.** The model with the cartridge answers spoken questions about
  the journal. Metric: our existing Teacher judge (rubric `score-v0`) on
  Whisper transcripts of the agent's audio, plus context-gap-closed on the
  agent *text* stream against the PersonaPlex teacher.
- **F3. Voice.** The agent speaks in your voice. Metric: speaker similarity
  (WavLM-TDNN or ECAPA cosine) between agent audio and held-out clips of you.
  Target: ≥ the stock voice-prompt baseline (≈0.6) for phase 1; ≥ 0.7 for
  phase 2 (LoRA).
- **F4. Duplex behaviour preserved.** Turn-taking latency, interruption
  latency, and *spurious onset rate* (speaking into silence; stock PersonaPlex
  does this in 27.5% of 5-min silences, arXiv 2609.13445) must not regress by
  more than 10% relative to the stock prompt.
- **F5. Persistence.** The cartridge stays resident for the whole session
  (pinned), independent of dialogue length.

### 3.2 Data

- **D1. Your voice.** ≥ 10 min of clean 24 kHz mono speech, −24 LUFS
  normalised, split 80/10/10 for prompt / voice-training / speaker-similarity
  eval. A 10 s to 30 s segment becomes the voice prompt in the cartridge
  init; similarity saturates beyond ~10 s (VALL-E 2, Voicebox), longer helps
  prosody.
- **D2. Your TTS.** The fine-tuned TTS on your voice clips (Chatterbox,
  CosyVoice2 or Kyutai Pocket TTS style) renders agent-side answers in your
  voice for the voice objective (phase 1 audio CE term and phase 2 LoRA). Its
  own speaker similarity to you is measured first; it is the ceiling of what
  we can teach.
- **D3. Questions.** Reuse the task DB: 2,785 train / 198 held-out questions
  (author's questions, chat prompts, psychology items, synthetic). Spoken
  versions are rendered with a *pool of other voices* (not yours) on the
  user channel, with noise and room-IR augmentation as PersonaPlex did.
- **D4. Evidence briefs.** For each train question, the Qwen Teacher (full
  journal) writes a ≤ 300-token brief: the facts and page citations the
  answer rests on. This goes into the PersonaPlex teacher's `<system>` prompt.
  Budget: 300 tokens = 300 frames = 24 s of prompt time; keeps the teacher's
  total prompt under ~700 frames so the dialogue still fits in 2048.
- **D5. Teacher rollouts.** PersonaPlex with [voice clip + persona + brief]
  answers each spoken question; store per frame the agent text token, 8 agent
  audio tokens, and full logits (text 32k, audio 8 × 2048) as top-k (k=20) +
  sampled token, matching our target store shape. ~3,000 dialogues of ~60 s.
- **D6. Journal digest for init.** A ≤ 1,500-token digest of the journal
  (timeline + recurring people/places, written by the Qwen Teacher) forced
  into the init prefill after the persona text, so the step-0 cartridge
  already holds a skeleton of the facts (the analogue of initialising from
  the first 2,048 journal tokens; Attention Matching does not apply because
  the teacher never holds the full journal).

### 3.3 Model and code

- **M1. A `DuplexCartridge` module** mirroring `cartridge/cache.py`:
  per-layer K/V parameters `[1, 32 heads, p, 128]` in fp32 master, bf16 in
  forward; first `num_frozen_tokens` frames frozen as the attention sink;
  `from_prefill(lm_model, frames)` and `init_from_prompt(voice_wav,
  persona_text, digest_text)`; `save`/`load`.
- **M2. Training forward.** A context manager that patches every temporal
  `StreamingMultiheadAttention.forward` to (a) concatenate cartridge K/V in
  front of the layer's K/V, (b) start RoPE positions at `p`, (c) extend the
  causal mask by `p` fully-visible columns. Gradient flows into the cartridge
  through the concat exactly as in `_ReadOnlyKVLayer`. The depth transformer
  is untouched. Gradient checkpointing per temporal layer; model weights
  frozen, `requires_grad=False`.
- **M3. Loss.** Per frame, over agent streams only:
  `KL(teacher_text ‖ student_text) + Σ_c w_c · KL(teacher_audio_c ‖
  student_audio_c)` with `w_0 = 1` (semantic codebook) and `w_1..7 = 0.02`
  (PersonaPlex's own non-semantic weight), plus an optional voice term
  `CE(student_audio_c, mimi(your_tts_render))` on dialogues where the agent
  audio was re-rendered by your TTS. Padded text frames weighted 0.3. The
  prefix itself has no loss.
- **M4. Stage 2 on-policy.** `LMGen` rollouts from the cartridge with the
  pinned-prefix cache (M5); the PersonaPlex teacher re-scores the rollout
  (user audio + the student's agent streams, teacher-forced) with the brief in
  context; reverse KL, full vocab, as in `onpolicy.py`. Anchor mix 0.25 → 0.5
  as today.
- **M5. Pinned-prefix ring cache.** `RingKVCache` gains a `pinned: int`
  region: slots `0..p−1` are written once at session start and excluded from
  the ring arithmetic; the ring covers `p..capacity−1`. Attention always sees
  prefix + ring window. This is the StreamingLLM / attention-sink layout.
  Also needed by the stock server to keep any long voice prompt alive, so it
  is a candidate upstream contribution.
- **M6. Server flag.** `moshi.server --cartridge <dir>` and `moshi.offline
  --cartridge <dir>` load M1, fill the pinned region, and skip
  `step_system_prompts`. The `.pt` voice path remains as-is.
- **M7. Tokenizer boundary.** No cross-tokenizer KL anywhere: the Qwen Teacher
  only writes *text* (briefs, digest, judge scores); every distribution we
  match is PersonaPlex's own.
- **M8. Phase 2 LoRA (gated).** moshi-finetune-style LoRA on the temporal
  transformer, with explicit wrappers for `in_proj_weight`, gating and
  depformer stacked weights (the paths module-swap misses). Data: dialogues
  whose agent audio is your real voice or your TTS. Merge into weights, then
  train the cartridge on the merged model. Only if phase 1 similarity < 0.65.

### 3.4 Infrastructure

- **I1. GPU.** One H100 80 GB or H200 for everything. Weights 16.7 GB bf16;
  student forward at 3000 frames with checkpointing ≈ 10–20 GB activations at
  micro-batch 2; cartridge + Adam state ≈ 4.3 GB at p=2048 (268M fp32 params
  × 4 states), 2.1 GB at p=1024. Fits comfortably. Modal: `gpu="H200"`,
  `timeout=6h`, same image lineage as `train_distill.py` plus the vendored
  `moshi` fork, `sentencepiece`, `whisper` for evals.
- **I2. Storage.** Teacher rollouts with top-20 logits: 3,000 dialogues ×
  750 frames × 9 streams × 20 × (4+4 bytes) ≈ 3.2 GB. Target store shards as
  today (`tasks/targets.py`) with a `stream` column.
- **I3. Tracking.** W&B runs and artifacts exactly as the text pipeline;
  cartridge artifact type `duplex-cartridge`.
- **I4. Audio tooling.** Mimi encode/decode (shipped), Whisper-large-v3 for
  transcripts, WavLM-TDNN (or `speechbrain` ECAPA) for speaker similarity,
  Full-Duplex-Bench scripts for turn-taking.

### 3.5 Evaluation

| Metric | Baseline (stock prompt) | Phase 1 target | How |
|---|---|---|---|
| Teacher-forced KL to PersonaPlex teacher, agent text stream | measured at init | −40% from init | held-out rollouts |
| Context gap closed, text stream (base prompt ↔ teacher-with-brief) | 0 | ≥ 0.4 | NLL on held-out teacher rollouts |
| Judge score (score-v0) on Whisper transcript | measured | ≥ text cartridge stage 2 (2.9 fidelity) | 40 held-out tasks, Qwen Teacher judge |
| Speaker similarity to you | ≈ 0.6 (prompt) | ≥ baseline; phase 2 ≥ 0.7 | WavLM-TDNN cosine, 100 utterances |
| Turn-taking / interruption latency | 0.17 s / 0.24 s | within 10% | Full-Duplex-Bench |
| Spurious onset rate | 27.5% | ≤ baseline | 20 × 5-min silence runs |
| Word error rate of agent speech vs its own text stream | measured | within 10% | Whisper |

### 3.6 Risks and unknowns

1. **Long forced text prompts are out of distribution.** PersonaPlex was
   trained with short role prompts. Forcing a 300-token brief plus a
   1,500-token digest may degrade the teacher. Mitigation: measure teacher
   answer quality against brief length (0 / 100 / 300 / 600 tokens) before
   generating labels; shrink the digest if the step-0 cartridge harms
   turn-taking.
2. **Prefix attention share.** A long trained prefix takes a growing share of
   softmax mass as the dialogue shortens (PrefixMemory-Tuning, arXiv
   2506.13674). We already train through this on text; on speech the symptom
   would be degraded user-audio tracking (late or missed turn-taking). F4
   catches it; `p=1024` is the safer first size.
3. **Depth transformer sensitivity.** The depformer conditions on the temporal
   output of the same frame; a drifted prefix could shift acoustic codebooks
   into junk. No evidence either way in the literature. Mitigation: the
   non-semantic weight 0.02 keeps acoustic codebooks close to the teacher, and
   we monitor UTMOS / DNS-MOS on rollouts every validation step.
4. **Voice via prefix saturates.** Expected. Phase 2 LoRA is the fallback and
   its own literature warns of quality loss (DNS-MOS 3.65 → 3.23 in one
   study) when the data has recording artefacts; your clips must be clean.
5. **Two-tier teacher quality.** The 7B teacher may misread briefs. We can
   grade teacher rollouts with the Qwen judge and keep only those scoring
   ≥ 4 fidelity, as `label.py` already validates labels.
6. **No upstream trainer.** We own the training forward of a vendored fork;
   NVIDIA has not answered requests for training code. Budget the engineering
   (section 4.6), and pin the fork commit (3428dfd).
7. **License.** NVIDIA Open Model License permits commercial use and
   derivatives with attribution; voice cloning of *your own* voice is fine, but
   the shipped donated voices and Kyutai's watermark policy do not apply to
   our prefix. Keep your clips private; the cartridge encodes your voice and
   must be treated as personal data.

## 4. Implementation spec

### 4.1 Package layout

```
src/cartridges/duplex/
  __init__.py
  model.py        load PersonaPlex (vendored moshi fork), Mimi, SentencePiece; freeze
  cache.py        DuplexCartridge (M1), prefix patch context manager (M2), pinned RingKVCache (M5)
  prompt.py       build init frames: voice clip → Mimi tokens, persona + digest → forced text frames, sine user frames
  data.py         Dialogue, Frame, TargetRow (stream, frame, token, topk); shard IO
  teacher.py      PersonaPlex teacher: rollout with brief in context, capture per-stream logits
  brief.py        Qwen Teacher writes evidence briefs and the journal digest (reuses cartridge/teacher.py)
  tts.py          render user-channel questions (other voices) and agent answers (your TTS); augment
  train.py        stage 1 distill (M3), stage 2 on-policy (M4); Modal app
  evals.py        Whisper transcripts, speaker sim, turn-taking, spurious onsets, judge glue
  serve.py        --cartridge flag for moshi.server / moshi.offline (M6)
```

Reuse, do not copy: `cartridge/tasks/*` for questions and the Qwen judge,
`cartridge/tracking.py`, `cartridge/tasks/targets.py` shard writer (add a
`stream` column), `train_distill.py` optimizer (`make_optimizer`, relative
LR, warmup) and `EarlyStopping`.

### 4.2 Data formats

```
dialogue.jsonl       {id, task_id, user_wav, brief, teacher_rollout_ref, split}
frames (safetensors) user_audio [T, 8] int16, agent_text [T] int32, agent_audio [T, 8] int16
targets (parquet)    id, frame, stream (0=text, 1..8=audio), topk_ids[20], topk_logprobs[20]
cartridge.json       {base_model, fork_commit, p, num_frozen_tokens, voice_sha, persona, digest_sha, stage, step}
cartridge.safetensors keys.{layer} [1,32,p,128] fp32, values.{layer} [1,32,p,128] fp32
```

### 4.3 Training forward (M2) in detail

For each temporal layer `l`:

```
q, k, v = proj(x)                      # [B, T, H, D]
q = rope(q, pos = p + arange(T))
k = rope(k, pos = p + arange(T))
k = cat([K_cart[l].expand(B), k], dim=T)   # [B, p+T, H, D]
v = cat([V_cart[l].expand(B), v], dim=T)
mask = causal(T) padded with p True columns on the left
out = sdpa(q, k, v, mask)
```

`K_cart`, `V_cart` are the cartridge's bf16 views of fp32 parameters; the
first `num_frozen_tokens` (default 1) frames are buffers. The model's own
17-stream input embedding and depformer run unchanged. Micro-batch 2 at
T ≤ 2048 frames, grad accumulation to batch 16. Mixed precision bf16, Adam,
relative LR 5e-2 (sweep 1e-2 … 1e-1 as before), warmup 5 steps, early
stopping on held-out teacher-forced KL.

### 4.4 Pinned-prefix ring cache (M5)

```
class RingKVCache:
    def __init__(self, ..., capacity, pinned=0): ...
    # slots [0, pinned) fixed; ring over [pinned, capacity)
    def complete(self, k, v):   # returns keys/values for attention
        return cat(cache[:, :pinned], ring_window(...))
    positions: pinned region reports 0..pinned-1, ring reports absolute frame + pinned
```

`set_attention_context(3000)` stays. With `p=1024` the dialogue window is
1976 frames (158 s) of rolling context after the pinned cartridge, versus
3000 frames today; raising `capacity` to 4096 is possible but RoPE beyond
3000 frames is untested on this checkpoint and must be measured (F4).

### 4.5 Phases and gates

| Phase | Deliverable | Gate to continue |
|---|---|---|
| **0. Parity** (1 week) | `DuplexCartridge.from_prefill` + M2 + M5 + `--cartridge` flag. Step-0 cartridge reproduces stock prompt logits on 10 test dialogues; pinned cache keeps a voice prompt alive past 240 s. | Max abs logit diff < 1e-2 bf16; turn-taking unchanged. |
| **0b. hertz-dev voice pilot** (optional, 1 week, CUDA only) | KV prefix on hertz-ar initialised from 60 s of your voice on the agent channel; trained on 2-channel continuations where the agent channel is your TTS. Measures speaker similarity and backchannel rate at minutes 1–4 vs the raw audio prefix. | Similarity ≥ raw prefix at minute 4; decides whether the textless route is worth a memory attempt. |
| **1a. Teacher study** (3 days) | Teacher answer quality vs brief length; digest length vs duplex metrics. | Pick brief/digest sizes with no F4 regression. |
| **1b. Labels** (3 days GPU) | 3,000 teacher rollouts with per-stream top-20 targets; judge-filtered. | ≥ 2,000 rollouts with fidelity ≥ 4. |
| **1c. Stage 1** (1 week) | Distilled cartridge, LR sweep, validation KL curve. | KL −40% from init; F4 within 10%. |
| **1d. Stage 2** (3 days) | On-policy refinement. | On-policy KL −25%; judge ≥ 2.9 fidelity. |
| **1e. Voice term** (3 days) | Add CE on your-TTS renders for 30% of batches. | Speaker sim ≥ baseline, UTMOS not down > 0.2. |
| **2. LoRA** (gated, 1 week) | Voice LoRA with explicit wrappers; cartridge retrained on merged model. | Speaker sim ≥ 0.7, F4 holds. |

### 4.6 Cost

Modal H200 at about $4.54/h. Phase 0 is engineering, ~2 GPU-h of tests.
Labels: 3,000 × 60 s dialogues; PersonaPlex at batch 8 runs near real-time
per batch, so ≈ 6 GPU-h ≈ $30. Stage 1: 200 steps × batch 16 × ~1,500 frames
on a frozen 7B ≈ 3–5 GPU-h ≈ $20 per run, ~4 runs for the sweep ≈ $80.
Stage 2: 30 steps with rollouts ≈ 4 GPU-h ≈ $20. Evals ≈ $10. **Phase 1 total
≈ $160–200 GPU**, above the $15.9 left on the current $50 cap. Phase 2 LoRA
≈ 6 h on one H100 per the moshi-finetune reference ≈ $25–30.

### 4.7 Acceptance criteria for "done" (phase 1)

- [ ] F1 parity test passes in CI on a stored logits fixture.
- [ ] A `duplex-cartridge` W&B artifact exists with `cartridge.json` stage 2.
- [ ] Held-out eval report with every row of the 3.5 table filled, baseline
      and cartridge side by side, 95% CIs by bootstrap over questions.
- [ ] A 3-minute recorded conversation where the model, with no text prompt
      and no voice file, answers three journal questions in your voice and
      survives one interruption.
- [ ] README section "Speech cartridge" with the recipe tables in the same
      format as the text recipe.

## 5. Decisions taken in this document

- Target model: PersonaPlex 7B at fork commit 3428dfd. Not VoiceChat, not
  ELLSA2. Not hertz-dev for memory (base model, textless teacher unproven);
  hertz-dev only as the optional voice pilot 0b.
- Cartridge form: KV-level (keys and values per temporal layer), not
  embedding-level prompt tuning, to match our code and capacity. The
  embedding-level form (a trained `.pt` fed through `step_embeddings`) is a
  zero-server-change fallback and a cheap ablation, not the main line.
- Teacher: two-tier. Qwen3.8 with the full journal writes text briefs; the
  PersonaPlex teacher with the brief in its prompt supplies every distribution
  we match.
- One cartridge for voice and memory, trained jointly; never compose two.
- Voice beyond prompt quality is a gated LoRA phase, not part of phase 1.
- First size `p = 1024` (0.54 GB, 158 s rolling window).

## 6. Open questions to settle before phase 1b

1. How long a forced text prompt can PersonaPlex read before answer quality
   or turn-taking drops? (Phase 1a.)
2. Does your fine-tuned TTS reach ≥ 0.7 similarity to you on its own? If not,
   the voice term and phase 2 have no ceiling worth chasing.
3. Does a prefix of 1024 vs 2048 frames change spurious onsets? Decides the
   capacity question in 4.4.

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
- Full-duplex model list: https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM
- Dessa-profile candidates: hertz-dev https://github.com/Standard-Intelligence/hertz-dev ·
  https://huggingface.co/si-pbc/hertz-dev · dGSLM https://arxiv.org/abs/2203.16502 ·
  RTTL-DG https://arxiv.org/abs/2501.04877 · PersonaPlex MLX port
  https://huggingface.co/aufklarer/PersonaPlex-7B-MLX-4bit
