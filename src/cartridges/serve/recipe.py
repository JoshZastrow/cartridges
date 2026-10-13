"""The OCR recipe: everything that turns a page image into text.

One recipe = merged weights (model_sha) + preprocessing + prompt + decoding +
postprocessing. The server owns the recipe; a client sends an image only. The
model id is the hash of the recipe. Thus:
- the cache key, the eval report, and the deploy record use one id;
- a change to any setting is a new model id, never a silent change.

Defaults match the training eval path (finetune._generate_one) with the
default DecodingConfig (no_repeat_ngram_size=10, repetition_penalty=1.05,
the looping investigation's pick): the collator's preprocessing, the
training prompt, greedy decoding, and the TrainConfig.max_new_tokens cap.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict

from cartridges.training.config import TrainConfig

END_OF_SENTENCE = "<｜end▁of▁sentence｜>"


class OcrRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_sha: str  # "base", or the adapter commit SHA (merged weights)

    # Preprocessing. vLLM's DeepSeek-OCR-2 processor gets these as
    # --mm-processor-kwargs. Same values as training/collator.py.
    base_size: int = 1024
    image_size: int = 768
    crop_mode: bool = True

    # Prompt. vLLM's template for this model gives BOS + image tokens + "\n"
    # + prompt: the same token ids as collator.FREE_OCR_PROMPT.
    prompt: str = "Free OCR. "

    # Decoding (greedy). ngram_size/window_size/whitelist go to vLLM's
    # NGramPerReqLogitsProcessor; a window >= max_model_len (8192) bans
    # repeated n-grams over the full output, as HF no_repeat_ngram_size does.
    max_tokens: int = TrainConfig().max_new_tokens
    temperature: float = 0.0
    ngram_size: int = 10
    window_size: int = 8192
    whitelist_token_ids: tuple[int, ...] = ()
    # vLLM's built-in penalty: like HF RepetitionPenaltyLogitsProcessor, it
    # applies to prompt and output tokens. 1.0 = off.
    repetition_penalty: float = 1.05

    # Postprocessing: remove these tokens, then strip whitespace.
    strip_tokens: tuple[str, ...] = (END_OF_SENTENCE,)

    @property
    def model_id(self) -> str:
        return hashlib.blake2b(self.model_dump_json().encode(), digest_size=16).hexdigest()

    def mm_processor_kwargs(self) -> dict:
        return {"base_size": self.base_size, "image_size": self.image_size, "crop_mode": self.crop_mode}

    def chat_request(self, image_base64: str, top_logprobs: int = 0) -> dict:
        """The vLLM /v1/chat/completions body for one page. top_logprobs > 0
        also asks for the top log-probabilities per output token (raw, before
        the logits processors), with token ids, for parity checks."""
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": self.ngram_size,
                "window_size": self.window_size,
                "whitelist_token_ids": list(self.whitelist_token_ids),
            },
        }
        if top_logprobs:
            body.update(logprobs=True, top_logprobs=top_logprobs, return_tokens_as_token_ids=True)
        return body

    def postprocess(self, text: str) -> str:
        for token in self.strip_tokens:
            text = text.replace(token, "")
        return text.strip()
