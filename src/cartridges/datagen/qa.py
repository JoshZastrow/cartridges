"""Question/answer pairs from journal passages, checkable by script.

A passage is a run of consecutive pages from one timeline entry
(batch.timeline), short enough for a teacher prompt. The teacher writes
questions about the passage only, so the passage's pages are the ground-truth
citations. Each pair carries:

- question: stands alone (names a date or a detail, never "this entry")
- answer: short canonical form (a number, a date, a few words, a list)
- answer_type: number | date | short_text | list | synthesis
- citations: ["<journal>:p.<N>", ...], pages of the passage
- evidence: a quote from a cited page

verify() keeps a pair only if its citations are pages of the passage and its
evidence fuzzy-matches the text of a cited page, which rejects made-up quotes.
The torch dataset that samples passages is in datagen.qa_dataset.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

ANSWER_TYPES = ("number", "date", "short_text", "list", "synthesis")
EVIDENCE_MIN_SCORE = 90  # rapidfuzz partial_ratio, 0-100; OCR noise and spacing differ a little


@dataclass(frozen=True)
class Passage:
    journal: str
    pages: tuple[tuple[int, str, str], ...]  # (page number, date label, text)

    @property
    def id(self) -> str:
        first, last = self.pages[0][0], self.pages[-1][0]
        return f"{self.journal}:p.{first}" + (f"-{last}" if last != first else "")

    @property
    def citations(self) -> list[str]:
        return [f"{self.journal}:p.{n}" for n, _, _ in self.pages]

    @property
    def text(self) -> str:
        """The pages as in teacher_context.md: one [journal:p.N] heading each."""
        return "\n\n---\n".join(
            f"## [{self.journal}:p.{n}] date: {label}\n\n{text}" for n, label, text in self.pages
        )


def passages(out_dir: Path, max_chars: int = 8000, min_chars: int = 300) -> list[Passage]:
    """Every timeline entry in `out_dir`, cut into runs of consecutive pages of
    at most `max_chars` (a longer single page stays whole). Runs shorter than
    `min_chars` (title pages, fragments) are left out."""
    from cartridges.batch.timeline import page_labels, timeline_entries

    out: list[Passage] = []
    for entry in timeline_entries(out_dir):
        run: list[tuple[int, str, str]] = []
        for (n, text), label in zip(entry.pages, page_labels(entry)):
            if run and sum(len(t) for _, _, t in run) + len(text) > max_chars:
                out.append(Passage(entry.journal, tuple(run)))
                run = []
            run.append((n, label, text))
        if run:
            out.append(Passage(entry.journal, tuple(run)))
    return [p for p in out if sum(len(t) for _, _, t in p.pages) >= min_chars]


PROMPT = """Below are pages from a handwritten journal, transcribed by OCR. Each page \
starts with a heading "## [<journal>:p.<N>] date: <date>". A date marked "(?)" or \
"(est.)" is uncertain; "(cont.)" means the page continues the entry before it.

Write {n} questions the journal's author could ask about these pages, each with its \
answer taken from the pages. Rules:
- Each question must stand alone, as if asked about the whole journal: name the date \
or a distinctive detail. Never write "this entry", "this page" or "the passage".
- Prefer answers that can be checked exactly: a number (e.g. a mood rating), a date, \
a few words, or a short list. At most one question may need a sentence-long synthesis.
- Do not ask about an uncertain date, and skip words that look garbled by OCR.
- answer: the shortest complete answer. evidence: a verbatim quote (one sentence or \
line) from the page that supports it. citations: the page anchors used, like \
"{example}".

Reply with JSON only: {{"pairs": [{{"question": "...", "answer": "...", \
"answer_type": "number|date|short_text|list|synthesis", "citations": ["..."], \
"evidence": "..."}}]}}

{text}"""


class Teacher:
    """Writes pairs for a passage with a chat model behind an OpenAI-compatible
    endpoint (e.g. vLLM). The API key comes from TEACHER_API_KEY. The client is
    created on first use, so the object pickles into DataLoader workers."""

    def __init__(self, base_url: str, model: str, pairs_per_passage: int = 4, temperature: float = 0.7):
        self.base_url, self.model = base_url, model
        self.pairs_per_passage, self.temperature = pairs_per_passage, temperature
        self._client = None

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_client": None}

    def __call__(self, passage: Passage) -> list[dict]:
        """The pairs as the teacher wrote them (unchecked); [] if the reply is not
        the expected JSON."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=os.environ.get("TEACHER_API_KEY", "EMPTY"))
        prompt = PROMPT.format(n=self.pairs_per_passage, example=passage.citations[0], text=passage.text)
        reply = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        try:
            pairs = json.loads(reply.choices[0].message.content or "")["pairs"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []
        return [p for p in pairs if isinstance(p, dict)] if isinstance(pairs, list) else []


def verify(pair: dict, passage: Passage) -> bool:
    """True if the pair is complete, cites only pages of the passage, and its
    evidence is found (fuzzily) on a cited page."""
    fields = ("question", "answer", "evidence")
    if not all(isinstance(pair.get(f), str) and pair[f].strip() for f in fields):
        return False
    if pair.get("answer_type") not in ANSWER_TYPES:
        return False
    cited = pair.get("citations")
    pages = dict(zip(passage.citations, (t for _, _, t in passage.pages)))
    if not isinstance(cited, list) or not cited or not all(c in pages for c in cited):
        return False
    evidence = _normalize(pair["evidence"])
    return any(fuzz.partial_ratio(evidence, _normalize(pages[c])) >= EVIDENCE_MIN_SCORE for c in cited)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
