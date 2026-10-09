"""Output writers for the batch OCR pipeline.

For each PDF, in <out>/<pdf_stem>/:
- transcript.md: golden format (datagen.loaders.format_pdf_page).
- pageindex.md: heading tree for PageIndex md mode (# doc, ## page).
- pageindex_tree.json: the same tree in the PageIndex output schema.
- manifest.json: keys and the status of each page.

For all PDFs, in <out>/:
- teacher_context.md: all transcripts in one file (collate()).

The functions read cache rows only. They do not call a model. Thus a second
write gives the same bytes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from cartridges.datagen.loaders import format_pdf_page, sanitize_page_text
from cartridges.storage.cache import PageRow

MISSING_TEXT = "[missing: this page failed OCR]"
UNKNOWN_DATE = "unknown"
# PageIndex md mode reads a line as a heading if it starts with 1-6 "#" and a
# space. OCR text can contain such lines. Escape them, so that only our own
# headings make tree nodes.
_HEADING_RE = re.compile(r"(?m)^(\s*)(#{1,6}\s)")
# ~4 characters per token for English text. An estimate, not a tokenizer.
CHARS_PER_TOKEN = 4


def _page_texts(page_keys: tuple[str, ...], rows: dict[str, PageRow]) -> list[tuple[int, str, str]]:
    """(page_number, text, status) for each page; status 'missing' if no row."""
    out = []
    for n, key in enumerate(page_keys, start=1):
        row = rows.get(key)
        out.append((n, row.text, row.status) if row else (n, MISSING_TEXT, "missing"))
    return out


def transcript_md(pages: list[tuple[int, str, str]]) -> str:
    sections = [format_pdf_page(n, text, UNKNOWN_DATE) for n, text, _ in pages]
    return "\n---\n".join(section.rstrip() for section in sections) + "\n"


def page_title(page_number: int, date: str = UNKNOWN_DATE) -> str:
    return f"p. {page_number} - date: {date}"


def pageindex(doc_name: str, pages: list[tuple[int, str, str]]) -> tuple[str, dict]:
    """Return (markdown, tree). The tree uses the PageIndex output schema:
    node_id is 4 digits, line_num is the 1-based line of the heading in the
    markdown, and start_index/end_index are page numbers."""
    lines = [f"# {doc_name}", ""]
    children = []
    for n, text, _ in pages:
        body = _HEADING_RE.sub(r"\1\\\2", sanitize_page_text(text))
        lines.append(f"## {page_title(n)}")
        children.append(
            {
                "title": page_title(n),
                "node_id": f"{len(children) + 1:04d}",
                "line_num": len(lines),
                "start_index": n,
                "end_index": n,
                "text": body,
            }
        )
        lines.extend(["", body, ""])
    markdown = "\n".join(lines).rstrip() + "\n"
    root = {
        "title": doc_name,
        "node_id": "0000",
        "line_num": 1,
        "start_index": 1 if pages else 0,
        "end_index": pages[-1][0] if pages else 0,
        "nodes": children,
    }
    tree = {"doc_name": doc_name, "line_count": markdown.count("\n"), "structure": [root]}
    return markdown, tree


def write_document(
    out_dir: Path,
    source_path: Path,
    file_key: str,
    page_keys: tuple[str, ...],
    rows: dict[str, PageRow],
    *,
    model_sha: str,
    model_key: str,
    dpi: int,
) -> Path:
    doc_name = source_path.stem
    doc_dir = out_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    pages = _page_texts(page_keys, rows)

    markdown, tree = pageindex(doc_name, pages)
    manifest = {
        "doc_name": doc_name,
        "source_path": str(source_path),
        "file_key": file_key,
        "dpi": dpi,
        "model_sha": model_sha,
        "model_key": model_key,
        "pages": [{"page_number": n, "status": status} for n, _, status in pages],
    }
    (doc_dir / "transcript.md").write_text(transcript_md(pages))
    (doc_dir / "pageindex.md").write_text(markdown)
    (doc_dir / "pageindex_tree.json").write_text(
        json.dumps(tree, indent=2, ensure_ascii=False) + "\n"
    )
    (doc_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return doc_dir


def collate(out_dir: Path, budget_tokens: int = 1_000_000) -> dict:
    """Join all <out>/*/transcript.md into <out>/teacher_context.md.

    Each page heading gets a stable anchor [doc:p.N]. The function does not
    cut text: if the estimate is over the budget, it still writes the file
    and returns fits=False, so the caller can decide.
    """
    parts = []
    doc_names = []
    for transcript in sorted(out_dir.glob("*/transcript.md")):
        doc_name = transcript.parent.name
        doc_names.append(doc_name)
        body = re.sub(
            r"(?m)^## p\. (\d+) - ",
            lambda m: f"## [{doc_name}:p.{m.group(1)}] ",
            transcript.read_text(),
        )
        parts.append(f"# {doc_name}\n\n{body.rstrip()}\n")
    context = "\n".join(parts)
    path = out_dir / "teacher_context.md"
    path.write_text(context)
    est_tokens = len(context) // CHARS_PER_TOKEN
    return {
        "path": str(path),
        "docs": doc_names,
        "chars": len(context),
        "est_tokens": est_tokens,
        "budget_tokens": budget_tokens,
        "fits": est_tokens <= budget_tokens,
    }
