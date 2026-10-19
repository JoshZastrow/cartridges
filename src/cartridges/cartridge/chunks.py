"""Random token-bounded chunks of the journal timeline, with page citations.

A chunk is a run of consecutive pages in timeline order (batch.timeline), its
text exactly as in teacher_context.md: one "## [<journal>:p.<N>] date: <label>"
heading per page, pages joined by "---". So `char_span` indexes
teacher_context.md, and a chunk works wherever datagen.qa takes a Passage
(it has the same `citations`, `pages` and `text`): qa.Teacher and qa.verify
accept it unchanged.

Self-study samples the whole corpus. QA tasks for training (GRPO) pass
`exclude_pages`, the held-out pages eval writes to
data/cartridge/eval/heldout_pages.json, so no training pair touches them.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cartridges.batch.outputs import CHARS_PER_TOKEN
from cartridges.datagen.qa import PROMPT, verify

_HEADER = "# Journals\n\n"  # teacher_context.md starts with this (outputs.collate)
_SEP = "\n\n---\n"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True)
class Chunk:
    id: str  # first and last page anchor, e.g. "Morning Pages:p.3..Morning Pages:p.5"
    text: str
    citations: tuple[str, ...]  # "<journal>:p.<N>", one per page, in order
    pages: tuple[tuple[int, str, str], ...]  # (page number, date label, text), as Passage.pages
    date_range: tuple[str, str] | None  # first and last YYYY-MM-DD; None if no page is dated
    char_span: tuple[int, int]  # teacher_context.md[a:b] == text


@dataclass(frozen=True)
class _Page:
    citation: str
    number: int
    label: str
    text: str
    section: str  # the page as in teacher_context.md
    start: int  # offset of `section` in teacher_context.md


def approx_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def timeline_pages(out_dir: Path) -> list[_Page]:
    """Every page of the timeline in order, with its offset in teacher_context.md."""
    from cartridges.batch.timeline import page_labels, timeline_entries

    pages: list[_Page] = []
    start = len(_HEADER)
    for e in timeline_entries(out_dir):
        for (n, text), label in zip(e.pages, page_labels(e)):
            section = f"## [{e.journal}:p.{n}] date: {label}\n\n{text}"
            pages.append(_Page(f"{e.journal}:p.{n}", n, label, text, section, start))
            start += len(section) + len(_SEP)
    return pages


def sample_chunks(
    timeline_path: Path,
    n: int,
    min_tokens: int = 512,
    max_tokens: int = 4096,
    seed: int = 0,
    exclude_pages: set[str] = frozenset(),
    count_tokens: Callable[[str], int] = approx_tokens,
) -> list[Chunk]:
    """`n` chunks of pages, deterministic for a seed.

    Each chunk starts at a page drawn in proportion to its tokens (so short
    and long pages are covered alike per token) and grows forward, then
    backward, while it fits a length drawn from [min_tokens, max_tokens].
    It never includes a
    page in `exclude_pages` ("<journal>:p.<N>"): those end a run, so a run
    shorter than `min_tokens` gives a shorter chunk. A single page longer than
    `max_tokens` stays whole.
    `timeline_path` is teacher_context.md (the corpus); its directory holds
    the journal transcripts. Pass a tokenizer's count for exact token bounds;
    the default estimate is within 1% for Qwen3.5 on these journals.
    """
    pages = timeline_pages(Path(timeline_path).parent)
    tokens = [count_tokens(p.section) for p in pages]
    ok = [p.citation not in exclude_pages for p in pages]
    starts = [i for i in range(len(pages)) if ok[i]]
    if not starts:
        raise ValueError("every page is excluded")
    weights = [tokens[i] for i in starts]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        target = rng.randint(min_tokens, max_tokens)
        a = b = rng.choices(starts, weights)[0]
        total = tokens[a]
        while b + 1 < len(pages) and ok[b + 1] and total + tokens[b + 1] <= target:
            b += 1
            total += tokens[b]
        while a > 0 and ok[a - 1] and total + tokens[a - 1] <= target:
            a -= 1
            total += tokens[a]
        out.append(_chunk(pages[a : b + 1]))
    return out


def _chunk(pages: list[_Page]) -> Chunk:
    text = _SEP.join(p.section for p in pages)
    dates = [p.label[:10] for p in pages if p.label[:4].isdigit()]
    citations = tuple(p.citation for p in pages)
    return Chunk(
        id=citations[0] if len(citations) == 1 else f"{citations[0]}..{citations[-1]}",
        text=text,
        citations=citations,
        pages=tuple((p.number, p.label, p.text) for p in pages),
        date_range=(min(dates), max(dates)) if dates else None,
        char_span=(pages[0].start, pages[0].start + len(text)),
    )


def qa_messages(chunk: Chunk, n: int = 4) -> list[dict]:
    """The datagen.qa prompt for `n` question/answer pairs about the chunk."""
    return [{"role": "user", "content": PROMPT.format(n=n, example=chunk.citations[0], text=chunk.text)}]


def grounded_qa(chunk: Chunk, reply: str) -> list[dict]:
    """The pairs in a teacher's JSON reply to qa_messages() that pass
    datagen.qa.verify (cited pages are in the chunk, evidence is on them), in
    the qa_heldout.jsonl format plus `chunk_id`. A <think> block and code
    fences around the JSON are stripped."""
    try:
        pairs = json.loads(_FENCE.sub("", _THINK.sub("", reply).strip()))["pairs"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    if not isinstance(pairs, list):
        return []
    fields = ("question", "answer", "answer_type", "citations", "evidence")
    return [
        {**{k: p[k] for k in fields}, "chunk_id": chunk.id}
        for p in pairs
        if isinstance(p, dict) and verify(p, chunk)
    ]
