"""Deterministic cache keys for the batch OCR pipeline.

All keys are BLAKE2b hex digests (hashlib, standard library).

- file_key: the hash of the PDF bytes. A change to the file gives a new key.
- page_key: the hash of file_key, page number, and DPI.
- model_key: the hash of the served model SHA and all decoding settings.
  A new model or a new decoding setting gives a new key.

The pipeline caches one row for each (page_key, model_key).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_CHUNK = 1 << 20


def file_key(path: Path) -> str:
    """Hash the file bytes. Read the file in 1 MiB chunks."""
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def page_key(file_key: str, page_number: int, dpi: int) -> str:
    payload = f"{file_key}|{page_number}|{dpi}".encode()
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


class DecodingSettings(BaseModel):
    """The request settings that change the model output.

    Defaults are the official DeepSeek-OCR-2 vLLM values (see the vLLM
    multimodal example `run_deepseek_ocr2`), with a lower max_tokens cap
    for the same reason as TrainConfig.max_new_tokens.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The training prompt (collator.FREE_OCR_PROMPT is "<image>\nFree OCR. ").
    # vLLM's default template for this model gives BOS + image tokens +
    # "\n" + this text: the same token ids as the training prompt.
    prompt: str = "Free OCR. "
    max_tokens: int = 1024
    temperature: float = 0.0
    ngram_size: int = 30
    window_size: int = 90
    # <td>, </td>
    whitelist_token_ids: tuple[int, ...] = (128821, 128822)


def model_key(model_sha: str, decoding: DecodingSettings) -> str:
    payload = json.dumps(
        {"model_sha": model_sha, "decoding": decoding.model_dump(mode="json")},
        sort_keys=True,
    ).encode()
    return hashlib.blake2b(payload, digest_size=32).hexdigest()
