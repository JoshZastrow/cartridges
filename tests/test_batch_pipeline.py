import asyncio
import json
import os
from pathlib import Path

import pytest

from cartridges.batch import keys
from cartridges.batch.outputs import collate, pageindex
from cartridges.batch.pipeline import process
from cartridges.storage.cache import PageRow, SourceFile

MODEL_SHA = "abc123"
MODEL_ID = "recipe-hash-1"


class MemoryStore:
    def __init__(self):
        self.files: dict[str, SourceFile] = {}
        self.pages: dict[tuple[str, str], PageRow] = {}

    async def get_source_file(self, path):
        return self.files.get(path)

    async def put_source_file(self, record):
        self.files[record.path] = record

    async def get_pages(self, page_keys, model_key):
        return {k: self.pages[(k, model_key)] for k in page_keys if (k, model_key) in self.pages}

    async def put_page(self, row):
        self.pages[(row.page_key, row.model_key)] = row


class FakeBackend:
    """Returns 'text <n>' for the n-th call. Fails the first `fail_first`
    calls. Returns a looped text for calls listed in `loop_calls`."""

    def __init__(self, fail_first=0, loop_calls=(), always_fail=False):
        self.calls = 0
        self.fail_first = fail_first
        self.loop_calls = set(loop_calls)
        self.always_fail = always_fail
        self.in_flight = 0
        self.max_in_flight = 0

    async def model_id(self):
        return MODEL_ID

    async def transcribe(self, png):
        assert png.startswith(b"\x89PNG")
        self.calls += 1
        n = self.calls
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            if self.always_fail or n <= self.fail_first:
                raise RuntimeError("server error")
            if n in self.loop_calls:
                return "dear diary and then and then and then and then"
            return f"text {n}"
        finally:
            self.in_flight -= 1


def make_pdf(path: Path, pages: int) -> Path:
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 50), f"page {i + 1}")
    doc.save(path)
    doc.close()
    return path


def run(paths, backend, store, **kwargs):
    kwargs.setdefault("render_workers", 1)
    kwargs.setdefault("retry_base_s", 0.0)
    kwargs.setdefault("dpi", 36)
    return asyncio.run(process(paths, backend=backend, store=store, model_id=MODEL_ID, model_sha=MODEL_SHA, **kwargs))


# --- keys ---------------------------------------------------------------------


def test_page_key_changes_with_each_input():
    base = keys.page_key("f", 1, 250)
    assert base == keys.page_key("f", 1, 250)
    variants = {keys.page_key("g", 1, 250), keys.page_key("f", 2, 250), keys.page_key("f", 1, 300)}
    assert base not in variants and len(variants) == 3


def test_file_key_is_content_hash(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert keys.file_key(a) == keys.file_key(b)
    b.write_bytes(b"other")
    assert keys.file_key(a) != keys.file_key(b)


# --- pipeline -----------------------------------------------------------------


def test_second_run_sends_no_requests(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 5)
    store = MemoryStore()

    first = run([pdf], FakeBackend(), store)
    assert first.docs[0].processed == 5 and first.ok

    backend = FakeBackend()
    second = run([pdf], backend, store)
    assert backend.calls == 0
    assert second.docs[0].cached == 5 and second.docs[0].processed == 0


def test_changed_file_is_processed_again(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 2)
    store = MemoryStore()
    run([pdf], FakeBackend(), store)

    make_pdf(pdf, 3)
    os.utime(pdf, ns=(1, 1))  # force a metadata change
    backend = FakeBackend()
    report = run([pdf], backend, store)
    assert backend.calls == 3
    assert report.docs[0].pages == 3


def test_retry_after_failure_succeeds(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 1)
    backend = FakeBackend(fail_first=2)
    report = run([pdf], backend, MemoryStore(), max_attempts=3)
    assert report.ok and report.docs[0].processed == 1
    assert backend.calls == 3


def test_failures_are_not_cached(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 2)
    store = MemoryStore()
    report = run([pdf], FakeBackend(always_fail=True), store, max_attempts=2)
    assert not report.ok
    assert report.docs[0].failed == 2 and len(report.failures) == 2
    assert store.pages == {}

    report = run([pdf], FakeBackend(), store)
    assert report.ok and report.docs[0].processed == 2


def test_looped_pages_are_retried_only_on_request(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 2)
    store = MemoryStore()
    report = run([pdf], FakeBackend(loop_calls={1}), store, concurrency=1)
    assert report.docs[0].looped == 1 and not report.ok

    backend = FakeBackend()
    report = run([pdf], backend, store)
    assert backend.calls == 0 and report.docs[0].looped == 1

    report = run([pdf], backend, store, retry_flagged=True)
    assert backend.calls == 1 and report.ok


def test_concurrency_bounds_requests_in_flight(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 12)
    backend = FakeBackend()
    run([pdf], backend, MemoryStore(), concurrency=3, render_workers=2)
    assert backend.calls == 12
    assert 1 < backend.max_in_flight <= 3


def test_dry_run_sends_nothing_and_writes_nothing(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 3)
    backend = FakeBackend()
    store = MemoryStore()
    report = run([pdf], backend, store, dry_run=True, out_dir=tmp_path / "out")
    assert backend.calls == 0 and store.pages == {}
    assert report.docs[0].pages == 3 and report.docs[0].cached == 0
    assert not (tmp_path / "out").exists()


def test_wrong_served_model_stops_the_run(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 1)

    class OtherModel(FakeBackend):
        async def model_id(self):
            return "other"

    with pytest.raises(RuntimeError, match="serves model"):
        run([pdf], OtherModel(), MemoryStore())


# --- outputs ------------------------------------------------------------------


def test_outputs_are_written(tmp_path):
    pdf = make_pdf(tmp_path / "notes.pdf", 3)
    out = tmp_path / "out"
    run([pdf], FakeBackend(), MemoryStore(), out_dir=out, concurrency=1)

    doc_dir = out / "notes"
    transcript = (doc_dir / "transcript.md").read_text()
    assert transcript.count("## p. ") == 3 and transcript.count("\n---\n") == 2

    tree = json.loads((doc_dir / "pageindex_tree.json").read_text())
    root = tree["structure"][0]
    assert (root["start_index"], root["end_index"]) == (1, 3)
    assert [n["start_index"] for n in root["nodes"]] == [1, 2, 3]
    md_lines = (doc_dir / "pageindex.md").read_text().splitlines()
    for node in root["nodes"]:
        assert md_lines[node["line_num"] - 1] == f"## {node['title']}"

    manifest = json.loads((doc_dir / "manifest.json").read_text())
    assert [p["status"] for p in manifest["pages"]] == ["ok"] * 3

    summary = collate(out, budget_tokens=10)
    context = (out / "teacher_context.md").read_text()
    assert "## [notes:p.2] date: unknown" in context
    assert summary["docs"] == ["notes"] and summary["fits"] is False


def test_pageindex_escapes_headings_in_page_text():
    markdown, tree = pageindex("doc", [(1, "# Title in handwriting\nbody", "ok")])
    assert "\n\\# Title in handwriting" in markdown
    assert len(tree["structure"][0]["nodes"]) == 1
