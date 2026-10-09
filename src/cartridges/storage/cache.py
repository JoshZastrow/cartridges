"""Content-addressed cache for page transcriptions, keyed on (page content, model).

Reprocessing the same source page with the same model is a cache hit.
Changing the source PDF (different page bytes) or swapping in a new/fine-tuned
model (different model_id) is a cache miss.

source_path/source_mtime are stored alongside each row for auditability/lookup
by file. They are not part of the cache key because page_hash already is
content-addressed, so a renamed/copied file with identical page bytes still
hits the cache.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

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


# --- v2: batch pipeline store (cartridges.batch) ------------------------------
#
# The batch pipeline uses its own two tables. The table above stays for the
# old per-page flow. Keys come from cartridges.batch.keys (BLAKE2b):
# page_key covers the file bytes, page number, and DPI; model_key covers the
# served model SHA and the decoding settings.

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS source_files (
    path TEXT PRIMARY KEY,
    size BIGINT NOT NULL,
    mtime_ns BIGINT NOT NULL,
    file_key TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ocr_pages (
    page_key TEXT NOT NULL,
    model_key TEXT NOT NULL,
    file_key TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    source_path TEXT,
    model_sha TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL,
    elapsed_s REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (page_key, model_key)
);
"""

PageStatus = Literal["ok", "looped"]


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: int
    mtime_ns: int
    file_key: str
    page_count: int


class PageRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_key: str
    model_key: str
    file_key: str
    page_number: int
    source_path: str | None
    model_sha: str
    text: str
    status: PageStatus
    elapsed_s: float | None = None


_PAGE_COLUMNS = tuple(PageRow.model_fields)
_FILE_COLUMNS = tuple(SourceFile.model_fields)


class PostgresPageStore:
    """Async store for the batch pipeline. Every write commits at once
    (autocommit), so a page that is complete stays saved if the run stops."""

    def __init__(self, conn) -> None:
        self._conn = conn

    @classmethod
    async def connect(cls, database_url: str | None = None) -> PostgresPageStore:
        import psycopg

        conn = await psycopg.AsyncConnection.connect(
            database_url or os.environ["DATABASE_URL"], autocommit=True
        )
        await conn.execute(_SCHEMA_V2)
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    async def get_source_file(self, path: str) -> SourceFile | None:
        cur = await self._conn.execute(
            f"SELECT {', '.join(_FILE_COLUMNS)} FROM source_files WHERE path = %s", (path,)
        )
        row = await cur.fetchone()
        return SourceFile(**dict(zip(_FILE_COLUMNS, row))) if row else None

    async def put_source_file(self, record: SourceFile) -> None:
        await self._conn.execute(
            "INSERT INTO source_files (path, size, mtime_ns, file_key, page_count) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (path) DO UPDATE SET "
            "size = EXCLUDED.size, mtime_ns = EXCLUDED.mtime_ns, "
            "file_key = EXCLUDED.file_key, page_count = EXCLUDED.page_count, updated_at = now()",
            (record.path, record.size, record.mtime_ns, record.file_key, record.page_count),
        )

    async def get_pages(self, page_keys: list[str], model_key: str) -> dict[str, PageRow]:
        if not page_keys:
            return {}
        cur = await self._conn.execute(
            f"SELECT {', '.join(_PAGE_COLUMNS)} FROM ocr_pages "
            "WHERE model_key = %s AND page_key = ANY(%s)",
            (model_key, page_keys),
        )
        rows = [PageRow(**dict(zip(_PAGE_COLUMNS, row))) for row in await cur.fetchall()]
        return {row.page_key: row for row in rows}

    async def put_page(self, row: PageRow) -> None:
        # DO UPDATE, not DO NOTHING: a retry of a flagged page replaces it.
        await self._conn.execute(
            f"INSERT INTO ocr_pages ({', '.join(_PAGE_COLUMNS)}) "
            f"VALUES ({', '.join(['%s'] * len(_PAGE_COLUMNS))}) "
            "ON CONFLICT (page_key, model_key) DO UPDATE SET "
            "text = EXCLUDED.text, status = EXCLUDED.status, "
            "elapsed_s = EXCLUDED.elapsed_s, source_path = EXCLUDED.source_path, "
            "created_at = now()",
            tuple(getattr(row, c) for c in _PAGE_COLUMNS),
        )
