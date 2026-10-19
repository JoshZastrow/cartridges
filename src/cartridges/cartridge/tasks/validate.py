"""Pure checks on Synthesizer drafts. validate_draft never raises on bad data:
it returns a Task with status "rejected" and the first reject reason, in this
order: unparseable, empty, too_long, mentions_document, duplicate, ungrounded,
uncertain_date, heldout_page, near_eval, near_exemplar, instruction_echo."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Collection, Mapping

from rapidfuzz import fuzz

from cartridges.cartridge.tasks.types import Draft, Page, Recipe, Task
from cartridges.datagen.qa import EVIDENCE_MIN_SCORE

MAX_WORDS = 400
NEAR_EVAL_JACCARD = 0.5
ECHO_JACCARD = 0.6  # vs the Synthesizer's own kind and style instructions
_DOCUMENT = re.compile(r"\bthe journals?\b|\bpages?\b|\bthe author\b", re.IGNORECASE)
_TASK_NS = uuid.UUID("6f1c2a52-3c1e-4a8e-9a57-1d3f0b8a7c11")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def task_id(hash_: str) -> uuid.UUID:
    """A task's id follows from its text hash, so reruns give the same ids."""
    return uuid.uuid5(_TASK_NS, hash_)


def word_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9']+", normalize_text(text)))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def grounded(draft: Draft, recipe: Recipe, pages: Mapping[str, Page]) -> bool:
    """At least one citation and one quote; every citation a focus page; every
    quote found (fuzzily) on one of the cited pages. Quotes are not matched to
    citations by position: models list them in a different order."""
    cited = [pages[c] for c in draft.ref_citations if c in recipe.citations and c in pages]
    if not draft.ref_evidence or not cited or len(cited) != len(draft.ref_citations):
        return False
    texts = [normalize_text(p.text) for p in cited]
    return all(
        any(fuzz.partial_ratio(normalize_text(e), t) >= EVIDENCE_MIN_SCORE for t in texts) for e in draft.ref_evidence
    )


def _instruction_words(recipe: Recipe) -> list[frozenset[str]]:
    """Word sets of the kind and style instructions, the recipe's own kind
    filled in with its framework or role."""
    from cartridges.cartridge.tasks.taxonomy import KIND_INSTRUCTIONS, STYLE_INSTRUCTIONS

    own = KIND_INSTRUCTIONS[recipe.kind].format(framework=recipe.framework, role=recipe.role)
    return [word_set(t) for t in (own, *KIND_INSTRUCTIONS.values(), *STYLE_INSTRUCTIONS.values())]


def validate_draft(
    draft: Draft | None,
    recipe: Recipe,
    pages: Mapping[str, Page],
    seen_hashes: Collection[str],
    eval_words: Collection[frozenset[str]] = (),
    raw: str = "",
    exemplar_words: Collection[frozenset[str]] = (),
) -> Task:
    """The draft as a synthetic train Task, validated or rejected. `raw` is the
    reply, stored as the text when the draft did not parse; `eval_words` and
    `exemplar_words` are word_set()s of the eval tasks and of the exemplars
    (human questions and their templates)."""
    text = draft.text.strip() if draft else raw.strip()
    ok = draft is not None and grounded(draft, recipe, pages)
    cited = [pages[c] for c in (draft.ref_citations if draft else ()) if c in pages]
    words = word_set(text)
    h = text_hash(text)
    if draft is None:
        reason = "unparseable"
    elif not text:
        reason = "empty"
    elif len(text.split()) > MAX_WORDS:
        reason = "too_long"
    elif _DOCUMENT.search(text):
        reason = "mentions_document"
    elif h in seen_hashes:
        reason = "duplicate"
    elif not ok:
        reason = "ungrounded"
    elif recipe.kind == "recall" and any(p.date_uncertain for p in cited):
        reason = "uncertain_date"
    elif recipe.kind == "recall" and any(p.heldout for p in cited):
        reason = "heldout_page"
    elif any(jaccard(words, e) >= NEAR_EVAL_JACCARD for e in eval_words):
        reason = "near_eval"
    elif any(jaccard(words, e) >= NEAR_EVAL_JACCARD for e in exemplar_words):
        reason = "near_exemplar"
    elif any(jaccard(words, e) >= ECHO_JACCARD for e in _instruction_words(recipe)):
        reason = "instruction_echo"
    else:
        reason = None
    return Task(
        task_id=task_id(h),
        text=text,
        text_hash=h,
        kind=recipe.kind,
        style=recipe.style,
        framework=recipe.framework,
        role=recipe.role,
        origin="synthetic",
        split="train",
        status="rejected" if reason else "validated",
        reject_reason=reason,
        source_citations=recipe.citations,
        ref_citations=draft.ref_citations if draft else (),
        ref_evidence=draft.ref_evidence if draft else (),
        grounded=ok,
        parent_ids=recipe.exemplar_ids,
    )
