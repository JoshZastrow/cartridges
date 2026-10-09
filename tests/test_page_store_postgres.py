"""Integration test for PostgresPageStore.

Runs only if CARTRIDGES_TEST_DATABASE_URL is set (not DATABASE_URL, so that
the test never writes to the main database by accident). Use a Neon branch.
"""

import asyncio
import os
import uuid

import pytest

from cartridges.storage.cache import PageRow, PostgresPageStore, SourceFile

DATABASE_URL = os.environ.get("CARTRIDGES_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="CARTRIDGES_TEST_DATABASE_URL is not set")


def test_round_trip_and_upsert():
    prefix = f"test-{uuid.uuid4().hex}"

    async def run():
        store = await PostgresPageStore.connect(DATABASE_URL)
        try:
            record = SourceFile(path=f"/{prefix}.pdf", size=1, mtime_ns=2, file_key="f", page_count=1)
            await store.put_source_file(record)
            assert await store.get_source_file(record.path) == record

            row = PageRow(
                page_key=f"{prefix}-p1", model_key=f"{prefix}-m", file_key="f", page_number=1,
                source_path=record.path, model_sha="sha", text="looped", status="looped",
            )
            await store.put_page(row)
            await store.put_page(row.model_copy(update={"text": "fixed", "status": "ok"}))
            pages = await store.get_pages([row.page_key, f"{prefix}-p2"], row.model_key)
            assert list(pages) == [row.page_key]
            assert (pages[row.page_key].text, pages[row.page_key].status) == ("fixed", "ok")
            assert await store.get_pages([row.page_key], "other-model") == {}
        finally:
            await store._conn.execute("DELETE FROM ocr_pages WHERE page_key LIKE %s", (f"{prefix}%",))
            await store._conn.execute("DELETE FROM source_files WHERE path LIKE %s", (f"/{prefix}%",))
            await store.close()

    asyncio.run(run())
