"""OCR backends: one async call transcribes one page image.

OcrServiceBackend calls the packaged OCR service (serve/vllm_server.py). The
service owns the recipe (prompt, decoding, postprocessing), so the client
sends an image only. The service batches concurrent requests on the GPU, so
the client only has to keep enough requests in flight.
"""

from __future__ import annotations

import base64
from typing import Protocol

import httpx


class OcrBackend(Protocol):
    async def model_id(self) -> str:
        """Return the served model id (the recipe hash; the cache key)."""

    async def transcribe(self, png: bytes) -> str:
        """Return the text of one page image."""


class OcrServiceBackend:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str,
        timeout_s: float = 600.0,
        max_connections: int = 256,
    ) -> None:
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
        self._info: dict | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def info(self) -> dict:
        """{"model_id": ..., "recipe": {...}} from the service."""
        if self._info is None:
            response = await self._client.get("/info")
            response.raise_for_status()
            self._info = response.json()
        return self._info

    async def model_id(self) -> str:
        return (await self.info())["model_id"]

    async def transcribe(self, png: bytes) -> str:
        response = await self._client.post(
            "/ocr", json={"image_base64": base64.b64encode(png).decode()}
        )
        response.raise_for_status()
        return response.json()["text"]
