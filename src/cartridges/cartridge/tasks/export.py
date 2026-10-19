"""Tasks as a Markdown file for reading and review: grouped by kind, each with
its style, focus pages, status or reject reason and the exemplars it was
written from."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from cartridges.cartridge.tasks.types import KINDS, Task


def render_md(tasks: Sequence[Task], parents: Mapping[UUID, Task], title: str) -> str:
    lines = [f"# {title}", "", f"{len(tasks)} tasks.", ""]
    for kind in KINDS:
        of_kind = [t for t in tasks if t.kind == kind]
        if not of_kind:
            continue
        lines += [f"## {kind} ({len(of_kind)})", ""]
        for i, t in enumerate(of_kind, 1):
            status = t.status if t.status == "validated" else f"rejected: {t.reject_reason}"
            meta = [status, t.origin, t.split, t.style or "", t.framework or "", t.role or ""]
            lines += [f"### {i}. {' · '.join(m for m in meta if m)}", "", *(f"> {line}" for line in t.text.splitlines()), ""]
            if t.source_citations:
                lines.append(f"- focus pages: {', '.join(t.source_citations)}")
            if t.ref_citations:
                lines.append(f"- cited: {', '.join(t.ref_citations)}")
            for q in t.ref_evidence:
                lines.append(f'- quote: "{q}"')
            for pid in t.parent_ids:
                parent = parents.get(pid)
                lines.append(f"- exemplar: {parent.text if parent else pid}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
