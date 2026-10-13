# HF eval vs vLLM serving: parity investigation (2026-10-13)

Status: closed. Production numbers come from the vLLM service. HF evals are
for relative comparisons only.

## Result

The HF code path and the vLLM code path run different vision encoders.

DeepSeek-OCR-2 encodes each view with a Qwen2 decoder used as an encoder
(`deepencoderv2.py`, "causal flow query"). The design mask is:
- image tokens attend to all image tokens (non-causal);
- query tokens attend to all image tokens and causally to query tokens.

The model's own HF code sets this mask in an override of
`Qwen2Model._update_causal_mask`. In transformers 4.56.2 (the version that
training pins), `Qwen2Model.forward` does not call `_update_causal_mask`. It
makes a standard causal mask unless the caller gives a mask dict
(`modeling_qwen2.py`, lines 360-372). Thus the override is dead code, and
the HF encoder runs with a plain causal mask.

vLLM 0.29.0 (`deepencoder2.py`) makes the design mask and gives it to
`Qwen2Model.forward` as `{"full_attention": mask}`. Thus vLLM runs the
design encoder.

Consequence:
- Training (unsloth + HF, transformers 4.56.2) and the HF eval use the
  causal-mask encoder. The fine-tune, including its LoRA on the Qwen2
  encoder, was trained against that encoder.
- The vLLM service uses the design encoder. With the same weights, it scores
  better (corpus CER 0.185 vs 0.347 on the 76 test pages, adapter
  run-liw4qzol, n-gram 35, no penalty).

This is not a layout bug, so this investigation does not change code. See
"Recommendation".

## Evidence: hidden states at each stage boundary

Script: `scripts/parity/stage_parity.py`. Both sides load the same merged
bf16 weights (`/cache/models/a2f23f4...`, adapter run-liw4qzol). Inputs are
the same pixels (training collator vs vLLM processor, bit-identical, see
`docs/plans/batch-ocr-pipeline.md` 3.8). Pages 63, 52, 44 of the test split.
HF ran with autocast bf16 (the eval path) and with plain bf16; the two modes
give the same result below.

| Stage | cosine HF vs vLLM | rel. L2 | Result |
|---|---|---|---|
| SAM encoder output (global, tiles) | 0.99997-0.99999 | 0.0001-0.004 | match (bf16 noise) |
| Qwen2 encoder output, global view | 0.70-0.74 | 0.76-0.77 | **differs** |
| Qwen2 encoder output, tiles | 0.82-0.85 | 0.55-0.59 | **differs** |
| Projector output (image embeddings) | 0.73-0.86 | 0.53-0.75 | differs (follows Qwen2) |
| View separator parameter | 1.0 | 0 | match |
| Layout in LM input: each side's image slots = its own [tiles, global, separator] | exact on both sides | 0 | same layout |
| Text-token embeddings in LM input | 1.0 | 0 | match |
| Final-norm hidden state, last prompt position | 0.93-0.95 | 0.31-0.37 | differs (follows the image embeddings) |

Verification (`scripts/parity/mask_verify.py`): the HF encoder, changed to
give the design mask as a mask dict (as vLLM does), matches vLLM:

| Pages 63, 52, 44 | Qwen2 global cos | Qwen2 tiles cos | Projector cos | rel. L2 |
|---|---|---|---|---|
| HF as shipped | 0.70-0.74 | 0.82-0.85 | 0.73-0.86 | 0.53-0.77 |
| HF with the design mask | 0.99988-0.99990 | 0.99981-0.99988 | 0.99980-0.99991 | 0.012-0.016 |

The remaining 1-2 % is bf16 kernel noise through the encoder layers.

Note on the harness: offline `LLM.chat` in vLLM added a second BOS (prompt
starts `0, 0`). The served OpenAI endpoint does not (`add_special_tokens`
defaults to false; checked on the live server earlier). The extra BOS does
not touch the vision stages. The final-norm row includes its effect.

## How the investigation got here

| Step | Test | Result | Rules out / in |
|---|---|---|---|
| 1 | Base-model HF eval vs HF adapter eval | 0.879 vs 0.383 corpus CER | HF eval does apply the adapter |
| 2 | Our merge vs PEFT `get_delta_weight` | 2,364/2,364 layers; bit-exact `bf16(W + B@A)` | merge is correct; bf16 storage loses 18-66 % of small deltas |
| 3 | HF eval on merged weights | 0.381 vs 0.383 (adapter) | merge rounding is not the gap |
| 4 | Property test, eval inputs vs collator (70 images) | prompt differed in 70/70: `model.infer()` stripped the trailing space token 223 | fixed in `15e1308`; HF 0.383 -> 0.347 |
| 5 | Three-way generate test, below | prefill logits differ at token 1 | unsloth and fast decode ruled out; forward pass differs |
| 6 | Stage-boundary dump (this page) | first difference: Qwen2 encoder | mask bug in HF path |

Sanity checks: both paths load adapter commit `a2f23f4` (tag run-liw4qzol,
root `adapter_model.safetensors`, not `checkpoint-2/`). The base on the
volume is `unsloth/DeepSeek-OCR-2` (shard sha256 = Hub LFS sha256). Weights
are bf16 on both paths; on HF only the LoRA A/B matrices (86M parameters)
are fp32.

### Three-way generate test

10 test pages (5 where HF and vLLM differ most, 5 where they agree). Same
collator inputs. Decoding: n-gram 35, repetition penalty 1.0, cap 2560.

| Run | Corpus CER, divergent pages | Corpus CER, agreeing pages | Loops |
|---|---|---|---|
| vLLM | 0.418 | 0.041 | 0 |
| a: unsloth + fast decode | 0.776 | 0.043 | 0 |
| b: unsloth + HF generate | 0.977 | 0.042 | 1 |
| c: plain transformers + PEFT (no unsloth) | 1.025 | 0.042 | 1 |

Prefill top-5 log-probabilities of the first output token: a, b and c agree
to 0.13 nats on all pages. All three differ from vLLM by up to 1.1 nats.
Example, page 63: vLLM ranks token 27765 second (-1.16); a, b, c rank it third
(about -2.6). Many first tokens are close decisions (heading or no heading),
so a small logit change gives a different page.

## Decoding: how the service applies the n-gram rule

- Processor: `vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor`
  (wraps `NoRepeatNGramLogitsProcessor`), set per request in `vllm_xargs`.
- It bans each token that completes an n-gram that occurs in the last
  `window_size` generated tokens. HF `no_repeat_ngram_size` also counts the
  prompt; for these prompts the effect is the same.
- Production values: `ngram_size=10`, `window_size=8192`,
  `whitelist_token_ids=[]`, `repetition_penalty=1.05` (vLLM's built-in
  penalty; prompt and output tokens, as HF).

## vLLM run-to-run noise

The same recipe on the 212-page loop set, two runs: 5 vs 3 loops, corpus CER
0.246 vs 0.233. Continuous batching changes the order of floating-point
operations, so greedy output is not exactly repeatable. Run each served
config two times before you call a winner.

## Fix (2026-10-13)

`src/cartridges/training/encoder_mask.py` `patch_encoder_mask()` replaces the
forward of the encoder class at load time. It makes the mask with the model's
own `_update_causal_mask` and gives it as `{"full_attention": mask}`, as vLLM
does. `finetune._load_base_model` applies it after the MoE patch, so training,
the HF eval and the loop eval all use it.

- `tests/test_encoder_mask.py` (Modal): before the patch, the layer-0 mask is
  not the design mask; after the patch, it is the design mask exactly.
- Through the real loader (unsloth, autocast bf16) on the merged run-liw4qzol
  weights, the Qwen2 encoder output now matches vLLM: cosine 0.997-0.9998
  (was 0.70-0.85).

Consequences:
1. Training and HF-eval numbers before this fix and after this fix are not
   comparable. Re-run baselines before you compare.
2. Adapters trained before the fix (E2, run-liw4qzol) were trained against the
   causal-mask encoder. Retrain with the fix and compare with E2 on the vLLM
   service, two runs for each config.
3. Run `tests/test_encoder_mask.py` again when the transformers version
   changes. If the unpatched mask becomes the design mask, remove the patch.
