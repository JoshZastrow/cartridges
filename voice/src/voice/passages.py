"""Journal -> dated passages with pages, topic labels, held-out pages (brief §1b, §5).

The journal is the teacher context file of the text cartridge
(`data/journals/output/<date>/teacher_context.md`): pages that start with a
heading `## [<journal>:p.<page>] date: YYYY-MM-DD ...`. A passage is one to
several consecutive sentences of one page:

    passages.jsonl: {passage_id, entry_date (YYYY-MM-DD), pages [int], text,
                     topic (sexuality|anger|other), borderline (bool), override (include|exclude|null)}

`passage_id` is `<journal-slug>-p<page:03d>-<k:02d>`, so the journal a page
number belongs to is in the id. Topic labels come from the Qwen Teacher
through an injected `generate(batch, **kw) -> list[Completion | None]`
(`shared.teacher.modal_generate`), one JSON-constrained call per passage.
Topic rule (TOPIC_RULE; brief §1b as of 7f7e4e6): filter what is spoken,
not what is read. Every passage that the author has not excluded is a
*source* (context, prompt source), whatever its label; every generated
question and response must pass the topic classifier instead. Only the
journal digest stays *clean*: `override == "include"`, or `topic == "other"`,
`borderline == False` and no `exclude` override.

The author can override labels with `data/voice/passage_overrides.csv`
(columns as the review export `excluded_passages.csv`; only rows with a
`decision` count). An overridden passage keeps the Teacher's `topic` and
`borderline` and gains `override`: `include` admits it to the clean (digest) pool,
`exclude` removes it everywhere. Downstream classifiers still see the
Teacher's label.

Held-out passages are chosen at the page level: every source passage of a
held-out page is held out, so no training passage shares a page with an eval
passage. Pages are drawn once per journal year (v0) and then frozen: a later
build passes the previous pages and only the passage ids are recomputed, so
neither an override nor a rule change moves a page between eval and training.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Topic = Literal["sexuality", "anger", "other"]
TOPICS: tuple[str, ...] = ("sexuality", "anger", "other")
TOPIC_RULE = "spoken-1: sources any label; generated questions and responses filtered; digest clean (brief §1b, 7f7e4e6)"

PAGE_HEADING = re.compile(r"^## \[(?P<journal>[^:\]]+):p\.(?P<page>\d+)\][^\n]*?date:\s*(?P<date>\d{4}-\d{2}-\d{2})[^\n]*$", re.M)
# Markdown markers that carry no words: bullets, quotes, headings, arrows, rules.
LINE_MARKER = re.compile(r"^\s*(?:[-*>#]+|->|\d+[.)])\s+")
RULE = re.compile(r"^\s*[-=_*]{2,}\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[\"“‘(]?[A-Z0-9])")


class Page(BaseModel):
    journal: str
    page: int
    date: str
    text: str


class Passage(BaseModel):
    passage_id: str
    entry_date: str
    pages: list[int]
    text: str
    topic: Topic | None = None  # None until labelled
    borderline: bool = False
    override: Literal["include", "exclude"] | None = None  # the author's decision; labels stay the Teacher's

    @property
    def year(self) -> int:
        return int(self.entry_date[:4])

    @property
    def source(self) -> bool:
        """Usable as context or prompt source (any label) unless the author excluded it."""
        return self.override != "exclude"

    @property
    def clean(self) -> bool:
        """Usable in the digest: labelled other and not borderline, or included by the author."""
        if self.override is not None:
            return self.override == "include"
        return self.topic == "other" and not self.borderline


def slug(journal: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", journal.lower()).strip("-")


def parse_pages(md: str) -> list[Page]:
    """Every `## [<journal>:p.N] date: ...` page of the teacher context file."""
    heads = list(PAGE_HEADING.finditer(md))
    pages = []
    for h, nxt in zip(heads, heads[1:] + [None]):
        body = md[h.end() : nxt.start() if nxt else len(md)]
        pages.append(Page(journal=h["journal"], page=int(h["page"]), date=h["date"], text=body.strip()))
    return pages


def sentences(text: str) -> list[str]:
    """Sentences of a page body. Each non-empty line is its own unit (journal
    lines are often bullets without final punctuation), split further at
    sentence ends; markdown markers and rules are dropped."""
    out = []
    for line in text.splitlines():
        if not line.strip() or RULE.match(line):
            continue
        line = LINE_MARKER.sub("", line).strip()
        out.extend(s.strip() for s in SENTENCE_END.split(line) if s.strip())
    return out


def _words(s: str) -> int:
    return len(s.split())


def segment_page(page: Page, min_words: int = 40, max_words: int = 120) -> list[Passage]:
    """Consecutive sentences grouped greedily to >= min_words without passing
    max_words (one longer sentence stands alone). A short tail joins the
    previous passage when that stays <= max_words."""
    groups: list[list[str]] = []
    cur: list[str] = []
    for s in sentences(page.text):
        if cur and sum(map(_words, cur)) + _words(s) > max_words:
            groups.append(cur)
            cur = []
        cur.append(s)
        if sum(map(_words, cur)) >= min_words:
            groups.append(cur)
            cur = []
    if cur:
        if groups and sum(map(_words, groups[-1] + cur)) <= max_words:
            groups[-1] += cur
        else:
            groups.append(cur)
    return [
        Passage(passage_id=f"{slug(page.journal)}-p{page.page:03d}-{k:02d}", entry_date=page.date,
                pages=[page.page], text=" ".join(g))
        for k, g in enumerate(groups)
    ]


def segment(pages: Iterable[Page], min_words: int = 40, max_words: int = 120) -> list[Passage]:
    return [p for page in pages for p in segment_page(page, min_words, max_words)]


def page_key(passage_id: str) -> str:
    """`<journal-slug>-p<page>`: the page a passage belongs to."""
    return passage_id.rsplit("-", 1)[0]


# ---------------------------------------------------------------- topic labels

LABEL_SYSTEM = (
    "You label short passages from a personal journal for a content filter. "
    "topic is 'sexuality' if the passage is about sex, sexual desire, sexual acts or sexual relationships; "
    "'anger' if it is mainly about the writer's anger, rage, resentment or a hostile outburst at someone; "
    "otherwise 'other'. borderline is true if the passage touches either topic at all, even in passing, "
    "or if you are unsure. Answer with JSON only."
)
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string", "enum": list(TOPICS)}, "borderline": {"type": "boolean"}},
    "required": ["topic", "borderline"],
}


def label_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": LABEL_SYSTEM}, {"role": "user", "content": f"Passage:\n{text}"}]


def parse_label(text: str | None) -> tuple[Topic, bool] | None:
    try:
        obj = json.loads(text or "")
        topic, borderline = obj["topic"], obj["borderline"]
    except (ValueError, KeyError, TypeError):
        return None
    if topic not in TOPICS or not isinstance(borderline, bool):
        return None
    return topic, borderline


def classify(texts: list[str], generate: Callable, seed: int = 0) -> list[tuple[Topic, bool]]:
    """Topic label per text, greedy and JSON-constrained. A call that fails or
    does not parse is retried once; if it still fails the text is labelled
    ("other", borderline=True), i.e. excluded."""
    kw = dict(max_tokens=32, temperature=0.0, seed=seed, json_schema=LABEL_SCHEMA, with_context=False)
    labels = [parse_label(c.text if c else None) for c in generate([label_messages(t) for t in texts], **kw)]
    retry = [i for i, lab in enumerate(labels) if lab is None]
    if retry:
        again = generate([label_messages(texts[i]) for i in retry], **kw)
        for i, c in zip(retry, again):
            labels[i] = parse_label(c.text if c else None)
    return [lab or ("other", True) for lab in labels]


def label(passages: list[Passage], generate: Callable, seed: int = 0) -> list[Passage]:
    labels = classify([p.text for p in passages], generate, seed)
    return [p.model_copy(update={"topic": t, "borderline": b}) for p, (t, b) in zip(passages, labels)]


def topic_counts(passages: list[Passage]) -> dict:
    """Counts for the summary comment: per topic, borderline, clean, excluded."""
    c = Counter(p.topic for p in passages)
    return {
        "total": len(passages),
        **{f"topic/{t}": c.get(t, 0) for t in TOPICS},
        "borderline": sum(p.borderline for p in passages),
        "clean": sum(p.clean for p in passages),
        "excluded": sum(not p.clean for p in passages),
    }


# ---------------------------------------------------------------- author overrides

DECISIONS = ("include", "exclude")


def read_overrides(path: Path) -> dict[str, str]:
    """passage_id -> include | exclude from the overrides CSV; rows without a decision are ignored."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            decision = (row.get("decision") or "").strip().lower()
            if not decision:
                continue
            if decision not in DECISIONS:
                raise ValueError(f"{path}: {row['passage_id']}: decision {decision!r} is not one of {DECISIONS}")
            out[row["passage_id"]] = decision
    return out


def apply_overrides(passages: list[Passage], overrides: dict[str, str]) -> list[Passage]:
    """Set `override` on the named passages (labels untouched). Unknown ids raise."""
    unknown = set(overrides) - {p.passage_id for p in passages}
    if unknown:
        raise ValueError(f"overrides name passages that do not exist: {sorted(unknown)[:5]}")
    return [p.model_copy(update={"override": overrides[p.passage_id]}) if p.passage_id in overrides else p
            for p in passages]


# ---------------------------------------------------------------- held-out split


def select_heldout_pages(passages: list[Passage], frac: float = 0.1, min_per_year: int = 3, seed: int = 0,
                         previous: list[str] | None = None) -> list[str]:
    """Held-out page keys. With `previous` (the pages of an earlier build), those
    pages, frozen. Otherwise `frac` of the pages with clean passages in every
    journal year (at least `min_per_year`), drawn with `seed` (how v0 drew them).
    Sorted."""
    if previous is not None:
        known = {page_key(p.passage_id) for p in passages}
        missing = set(previous) - known
        if missing:
            raise ValueError(f"previous held-out pages not in the journal: {sorted(missing)[:5]}")
        return sorted(previous)
    by_year: dict[int, set[str]] = defaultdict(set)
    for p in passages:
        if p.clean:
            by_year[p.year].add(page_key(p.passage_id))
    rng = random.Random(seed)
    held = []
    for year in sorted(by_year):
        keys = sorted(by_year[year])
        k = min(len(keys), max(min_per_year, round(frac * len(keys))))
        held += rng.sample(keys, k)
    return sorted(held)


def heldout_passages(passages: list[Passage], heldout_pages: Iterable[str]) -> list[Passage]:
    pages = set(heldout_pages)
    return [p for p in passages if p.source and page_key(p.passage_id) in pages]


# ---------------------------------------------------------------- IO


def write_jsonl(rows: Iterable[BaseModel | dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write((r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r)) + "\n")


def read_passages(path: Path) -> list[Passage]:
    return [Passage.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
