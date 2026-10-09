"""OCR backends: one async call transcribes one page image.

VllmBackend sends an OpenAI-style chat-completions request to the vLLM server
in cartridges.serve.vllm_server. The server batches concurrent requests on
the GPU, so the client only has to keep enough requests in flight.
"""

from __future__ import annotations

import base64
from typing import Protocol

import httpx

from cartridges.batch.keys import DecodingSettings

# DeepSeek-OCR-2 emits this token at the end; vLLM returns it because the
# request sets skip_special_tokens=False (the official recipe asks for it).
_END_TOKENS = ("<｜end▁of▁sentence｜>",)


class OcrBackend(Protocol):
    async def served_model(self) -> str:
        """Return the served model SHA (part of the cache key)."""

    async def transcribe(self, png: bytes) -> str:
        """Return the text of one page image."""


class VllmBackend:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        decoding: DecodingSettings = DecodingSettings(),
        timeout_s: float = 600.0,
        max_connections: int = 256,
    ) -> None:
        self.decoding = decoding
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections),
            # Modal web functions have a 150 s HTTP limit. A longer request
            # (for example, during a cold start) gets a 303 to a result URL;
            # the client must follow it. httpx allows 20 redirects (~50 min).
            follow_redirects=True,
        )
        self._model: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def served_model(self) -> str:
        """Read the served model name from /v1/models. The server sets the
        name to the model SHA (vllm --served-model-name)."""
        if self._model is None:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            models = [m["id"] for m in response.json()["data"]]
            if len(models) != 1:
                raise RuntimeError(f"expected 1 served model, got {models}")
            self._model = models[0]
        return self._model

    async def transcribe(self, png: bytes) -> str:
        d = self.decoding
        image_url = "data:image/png;base64," + base64.b64encode(png).decode()
        payload = {
            "model": await self.served_model(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": d.prompt},
                    ],
                }
            ],
            "max_tokens": d.max_tokens,
            "temperature": d.temperature,
            "skip_special_tokens": False,
            # vllm_xargs -> SamplingParams.extra_args -> NGramPerReqLogitsProcessor
            "vllm_xargs": {
                "ngram_size": d.ngram_size,
                "window_size": d.window_size,
                "whitelist_token_ids": list(d.whitelist_token_ids),
            },
        }
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"] or ""
        for token in _END_TOKENS:
            text = text.replace(token, "")
        return text.strip()
