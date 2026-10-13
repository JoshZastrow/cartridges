"""Batch OCR pipeline: PDFs in, one cached row per page out.

Shape (see docs/plans/batch-ocr-pipeline.md, section 3.4):

    job queue (cache misses) -> R render tasks (process pool)
        -> bounded PNG queue -> N OCR workers (async HTTP) -> store

- N workers set the number of requests in flight.
- The bounded PNG queue stops the render tasks when the workers are slow
  (backpressure). Memory stays bounded for all PDF sizes.
- Each worker saves its row when its page is complete. A stopped run loses
  only the pages in flight. The next run skips all saved pages.
- The store does not keep errors. It keeps looped pages with status
  "looped"; retry_flagged=True sends them again.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from cartridges.batch import keys
from cartridges.batch.backends import OcrBackend
from cartridges.storage.cache import PageRow, SourceFile
from cartridges.training.loops import ends_in_loop

logger = logging.getLogger(__name__)


class PageStore(Protocol):
    async def get_source_file(self, path: str) -> SourceFile | None: ...
    async def put_source_file(self, record: SourceFile) -> None: ...
    async def get_pages(self, page_keys: list[str], model_key: str) -> dict[str, PageRow]: ...
    async def put_page(self, row: PageRow) -> None: ...


@dataclass(frozen=True)
class Doc:
    path: Path
    file_key: str
    page_keys: tuple[str, ...]  # index i is page i + 1


@dataclass(frozen=True)
class Job:
    doc: Doc
    page_number: int
    page_key: str


class PageFailure(BaseModel):
    source_path: str
    page_number: int
    error: str


class DocReport(BaseModel):
    source_path: str
    file_key: str
    pages: int
    cached: int = 0
    processed: int = 0
    looped: int = 0
    failed: int = 0


class RunReport(BaseModel):
    model_id: str
    model_sha: str
    dpi: int
    dry_run: bool
    docs: list[DocReport]
    failures: list[PageFailure] = []
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        """True if no page failed and no page is looped (cached or new)."""
        return not self.failures and self.looped_total == 0

    @property
    def looped_total(self) -> int:
        return sum(d.looped for d in self.docs)


def render_png(path: str, page_number: int, dpi: int) -> bytes:
    """Rasterize one page to PNG bytes. Same pixels as
    processors.ocr.pdf_page_to_image. Module level, so the process pool
    can pickle it."""
    import fitz  # pymupdf

    with fitz.open(path) as doc:
        zoom = dpi / 72
        return doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")


def _page_count(path: Path) -> int:
    import fitz  # pymupdf

    with fitz.open(path) as doc:
        return len(doc)


async def describe_doc(path: Path, store: PageStore, dpi: int) -> Doc:
    """Get the file_key. If size and mtime_ns agree with the saved row, use
    the saved key and do not read the file again."""
    path = path.resolve()
    stat = path.stat()
    record = await store.get_source_file(str(path))
    if record is None or (record.size, record.mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        file_key = await asyncio.to_thread(keys.file_key, path)
        page_count = await asyncio.to_thread(_page_count, path)
        record = SourceFile(
            path=str(path),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            file_key=file_key,
            page_count=page_count,
        )
        await store.put_source_file(record)
    page_keys = tuple(
        keys.page_key(record.file_key, n, dpi) for n in range(1, record.page_count + 1)
    )
    return Doc(path=path, file_key=record.file_key, page_keys=page_keys)


async def process(
    paths: list[Path],
    *,
    backend: OcrBackend,
    store: PageStore,
    model_id: str,
    model_sha: str,
    dpi: int = 250,
    concurrency: int = 64,
    queue_size: int | None = None,
    render_workers: int | None = None,
    retry_flagged: bool = False,
    dry_run: bool = False,
    max_attempts: int = 3,
    retry_base_s: float = 2.0,
    out_dir: Path | None = None,
) -> RunReport:
    start = time.monotonic()
    model_key = model_id  # the recipe hash: weights + all settings

    docs = [await describe_doc(p, store, dpi) for p in paths]
    reports = {
        d.path: DocReport(source_path=str(d.path), file_key=d.file_key, pages=len(d.page_keys))
        for d in docs
    }

    jobs: list[Job] = []
    for doc in docs:
        cached = await store.get_pages(list(doc.page_keys), model_key)
        for n, key in enumerate(doc.page_keys, start=1):
            row = cached.get(key)
            if row is None or (retry_flagged and row.status != "ok"):
                jobs.append(Job(doc=doc, page_number=n, page_key=key))
            else:
                reports[doc.path].cached += 1
                reports[doc.path].looped += row.status == "looped"

    report = RunReport(
        model_id=model_id, model_sha=model_sha, dpi=dpi, dry_run=dry_run,
        docs=list(reports.values()),
    )
    logger.info(
        "plan: docs=%d pages=%d cached=%d to_process=%d model_id=%s model_sha=%s",
        len(docs), sum(r.pages for r in report.docs), sum(r.cached for r in report.docs),
        len(jobs), model_id, model_sha,
    )

    if jobs and not dry_run:
        served = await backend.model_id()
        if served != model_id:
            raise RuntimeError(
                f"server serves model {served!r}, but the run expects {model_id!r}. "
                "Deploy the expected revision first (cartridges serve_deploy)."
            )
        await _run_jobs(
            jobs, backend=backend, store=store, model_sha=model_sha, model_key=model_key,
            dpi=dpi, concurrency=concurrency, queue_size=queue_size or 2 * concurrency,
            render_workers=render_workers or min(8, os.cpu_count() or 1),
            max_attempts=max_attempts, retry_base_s=retry_base_s,
            reports=reports, failures=report.failures,
        )

    if out_dir is not None and not dry_run:
        from cartridges.batch.outputs import write_document

        for doc in docs:
            rows = await store.get_pages(list(doc.page_keys), model_key)
            write_document(
                out_dir, doc.path, doc.file_key, doc.page_keys, rows,
                model_sha=model_sha, model_key=model_key, dpi=dpi,
            )

    report.elapsed_s = time.monotonic() - start
    return report


async def _run_jobs(
    jobs: list[Job],
    *,
    backend: OcrBackend,
    store: PageStore,
    model_sha: str,
    model_key: str,
    dpi: int,
    concurrency: int,
    queue_size: int,
    render_workers: int,
    max_attempts: int,
    retry_base_s: float,
    reports: dict[Path, DocReport],
    failures: list[PageFailure],
) -> None:
    job_queue: asyncio.Queue[Job] = asyncio.Queue()
    for job in jobs:
        job_queue.put_nowait(job)
    png_queue: asyncio.Queue[tuple[Job, bytes] | None] = asyncio.Queue(maxsize=queue_size)
    loop = asyncio.get_running_loop()
    done = 0
    run_start = time.monotonic()

    def fail(job: Job, error: BaseException) -> None:
        reports[job.doc.path].failed += 1
        failures.append(
            PageFailure(source_path=str(job.doc.path), page_number=job.page_number, error=repr(error))
        )
        logger.warning("page failed: %s p.%d: %r", job.doc.path.name, job.page_number, error)

    async def renderer(pool: ProcessPoolExecutor) -> None:
        while not job_queue.empty():
            job = job_queue.get_nowait()
            try:
                png = await loop.run_in_executor(
                    pool, render_png, str(job.doc.path), job.page_number, dpi
                )
            except Exception as error:
                fail(job, error)
                continue
            await png_queue.put((job, png))  # waits while the queue is full

    async def transcribe(job: Job, png: bytes) -> str | None:
        for attempt in range(1, max_attempts + 1):
            try:
                return await backend.transcribe(png)
            except Exception as error:
                if attempt == max_attempts:
                    fail(job, error)
                    return None
                await asyncio.sleep(retry_base_s * 2 ** (attempt - 1))
        return None

    async def worker() -> None:
        nonlocal done
        while (item := await png_queue.get()) is not None:
            job, png = item
            page_start = time.monotonic()
            text = await transcribe(job, png)
            if text is None:
                continue
            status = "looped" if ends_in_loop(text) else "ok"
            await store.put_page(
                PageRow(
                    page_key=job.page_key, model_key=model_key, file_key=job.doc.file_key,
                    page_number=job.page_number, source_path=str(job.doc.path),
                    model_sha=model_sha, text=text, status=status,
                    elapsed_s=time.monotonic() - page_start,
                )
            )
            report = reports[job.doc.path]
            report.processed += 1
            report.looped += status == "looped"
            done += 1
            if done % 10 == 0 or done == len(jobs):
                rate = done / (time.monotonic() - run_start)
                logger.info("progress: %d/%d pages (%.2f pages/s)", done, len(jobs), rate)

    with ProcessPoolExecutor(max_workers=render_workers) as pool:
        async with asyncio.TaskGroup() as group:
            for _ in range(concurrency):
                group.create_task(worker())
            renderers = [group.create_task(renderer(pool)) for _ in range(render_workers)]
            await asyncio.gather(*renderers)
            for _ in range(concurrency):
                await png_queue.put(None)  # one stop signal for each worker
