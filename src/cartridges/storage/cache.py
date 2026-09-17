"""Content-addressed cache for page transcriptions, keyed on (page content, model).

Reprocessing the same source page with the same model is a cache hit.
Changing the source PDF (different page bytes) or swapping in a new/fine-tuned
model (different model_id) is a cache miss.

source_path/source_mtime are stored alongside each row for auditability/lookup
by file. They are not part of the cache key because page_hash already is
content-addressed, so a renamed/copied file with identical page bytes still
hits the cache.
"""

import hashlib
import os
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS page_transcriptions (
    page_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    transcription TEXT NOT NULL,
    source_path TEXT,
    source_mtime TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (page_hash, model_id)
);
ALTER TABLE page_transcriptions ADD COLUMN IF NOT EXISTS source_path TEXT;
ALTER TABLE page_transcriptions ADD COLUMN IF NOT EXISTS source_mtime TIMESTAMPTZ;
"""


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _connect():
    import psycopg

    database_url = os.environ["DATABASE_URL"]
    return psycopg.connect(database_url)


def init_cache() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def get_cached(page_hash: str, model_id: str) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT transcription FROM page_transcriptions WHERE page_hash = %s AND model_id = %s",
            (page_hash, model_id),
        ).fetchone()
    return row[0] if row else None


def put_cached(
    page_hash: str,
    model_id: str,
    transcription: str,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> None:
    mtime = (
        datetime.fromtimestamp(source_mtime, tz=timezone.utc)
        if source_mtime is not None
        else None
    )
    with _connect() as conn:
        conn.execute(
            "INSERT INTO page_transcriptions "
            "(page_hash, model_id, transcription, source_path, source_mtime) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (page_hash, model_id) DO NOTHING",
            (page_hash, model_id, transcription, source_path, mtime),
        )
