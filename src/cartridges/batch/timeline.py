"""One timeline across all journals in an output directory.

Reads each <out>/<journal>/transcript.md (headers "## p. N - date: <label>",
see batch.dates), groups pages into entries (a dated page and the "(cont.)"
pages after it), sorts the entries by date across journals, and writes:

- <out>/timeline.md: "# Journals" > "## <year>" > "### <year-month>" >
  "#### <date> · <journal> p. <first>–<last>", then the entry's text. Page
  text headings are escaped, so it is ready for PageIndex md mode.
- <out>/timeline_pageindex_tree.json: the same tree in the PageIndex output
  schema. start_index/end_index count pages in timeline order (1..N); each
  entry title names its journal and pages.

Entries on the same date keep journal order (by first date) and page order.
Reads output files only: no model, no database.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from cartridges.batch.outputs import _HEADING_RE

_PAGE_HEADER = re.compile(r"(?m)^## p\. (\d+) - date: (.*)$")
_SEPARATOR = "---"
UNDATED = "undated"


@dataclass
class Entry:
    journal: str
    date: str  # YYYY-MM-DD, or "undated" if the journal has no date at all
    marker: str  # "", "(?)" or "(est.)"
    pages: list[tuple[int, str]] = field(default_factory=list)  # (page number, text)

    @property
    def title(self) -> str:
        first, last = self.pages[0][0], self.pages[-1][0]
        span = f"p. {first}" if first == last else f"p. {first}–{last}"
        return f"{self.date}{' ' + self.marker if self.marker else ''} · {self.journal} {span}"


def read_entries(transcript: Path) -> list[Entry]:
    """The entries of one journal transcript, in page order."""
    journal = transcript.parent.name
    text = transcript.read_text()
    headers = list(_PAGE_HEADER.finditer(text))
    entries: list[Entry] = []
    for k, m in enumerate(headers):
        end = headers[k + 1].start() if k + 1 < len(headers) else len(text)
        body = text[m.end():end].strip()
        if body.endswith(_SEPARATOR):  # the separator before the next page
            body = body[: -len(_SEPARATOR)].rstrip()
        label = m.group(2).strip()
        page = (int(m.group(1)), body)
        if label.endswith("(cont.)") and entries:
            entries[-1].pages.append(page)
            continue
        date = label[:10] if label[:4].isdigit() else UNDATED
        entries.append(Entry(journal, date, label[10:].strip() if date != UNDATED else "", [page]))
    return entries


def build_timeline(out_dir: Path) -> dict:
    """Write timeline.md and timeline_pageindex_tree.json; return a summary."""
    journals = [read_entries(t) for t in sorted(out_dir.glob("*/transcript.md"))]
    journals = [j for j in journals if j]
    if not journals:
        raise FileNotFoundError(f"no */transcript.md in {out_dir}")
    # Journal order: by first date, so same-day entries follow the journals' order.
    journals.sort(key=lambda j: min((e.date for e in j if e.date != UNDATED), default="9999"))
    rank = {j[0].journal: i for i, j in enumerate(journals)}
    entries = sorted(
        (e for j in journals for e in j),
        key=lambda e: (e.date == UNDATED, e.date, rank[e.journal], e.pages[0][0]),
    )

    lines = ["# Journals", ""]
    root = _node("Journals", 1)
    page_index = 0
    year_node = month_node = None
    for e in entries:
        year, month = (e.date[:4], e.date[:7]) if e.date != UNDATED else (UNDATED, UNDATED)
        if year_node is None or year_node["title"] != year:
            lines += [f"## {year}", ""]
            year_node = _node(year, len(lines) - 1)
            root["nodes"].append(year_node)
            month_node = None
        if month != year and (month_node is None or month_node["title"] != month):
            lines += [f"### {month}", ""]
            month_node = _node(month, len(lines) - 1)
            year_node["nodes"].append(month_node)
        parent = month_node if month != year else year_node
        lines.append(f"#### {e.title}")
        heading_line = len(lines)
        body = "\n\n".join(
            (escaped if k == 0 else f"*(p. {n})*\n\n{escaped}")
            for k, (n, text) in enumerate(e.pages)
            for escaped in [_HEADING_RE.sub(r"\1\\\2", text)]
        )
        lines += ["", *body.split("\n"), ""]  # one item per line, so line_num stays exact
        parent["nodes"].append(
            {
                "title": e.title,
                "line_num": heading_line,
                "start_index": page_index + 1,
                "end_index": page_index + len(e.pages),
                "text": body,
            }
        )
        page_index += len(e.pages)

    markdown = "\n".join(lines).rstrip() + "\n"
    _fill_ranges(root)
    _number(root, iter(range(100_000)))
    tree = {"doc_name": "Journals", "line_count": markdown.count("\n"), "structure": [root]}
    (out_dir / "timeline.md").write_text(markdown)
    (out_dir / "timeline_pageindex_tree.json").write_text(json.dumps(tree, indent=2, ensure_ascii=False) + "\n")
    dated = [e for e in entries if e.date != UNDATED]
    return {
        "timeline": str(out_dir / "timeline.md"),
        "tree": str(out_dir / "timeline_pageindex_tree.json"),
        "journals": [j[0].journal for j in journals],
        "entries": len(entries),
        "pages": page_index,
        "first_date": dated[0].date if dated else None,
        "last_date": dated[-1].date if dated else None,
        "estimated_or_unsure": sum(bool(e.marker) for e in entries),
        "undated_entries": len(entries) - len(dated),
    }


def _node(title: str, line_num: int) -> dict:
    return {"title": title, "line_num": line_num, "nodes": []}


def _fill_ranges(node: dict) -> tuple[int, int]:
    """start_index/end_index of a section = the pages of its entries."""
    if "nodes" not in node:
        return node["start_index"], node["end_index"]
    spans = [_fill_ranges(child) for child in node["nodes"]]
    node["start_index"] = min(s for s, _ in spans) if spans else 0
    node["end_index"] = max(e for _, e in spans) if spans else 0
    return node["start_index"], node["end_index"]


def _number(node: dict, ids) -> None:
    """node_id: 4+ digits in document order, root 0000 (PageIndex schema)."""
    node["node_id"] = f"{next(ids):04d}"
    for child in node.get("nodes", []):
        _number(child, ids)
