"""Deterministic cache keys for the batch OCR pipeline.

All keys are BLAKE2b hex digests (hashlib, standard library).

- file_key: the hash of the PDF bytes. A change to the file gives a new key.
- page_key: the hash of file_key, page number, and DPI.

The model side of the cache key is the served model id: the hash of the OCR
recipe (serve/recipe.py). A new model or any new setting gives a new id.

The pipeline caches one row for each (page_key, model_id).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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
