"""The Task Synthesizer (docs/plans/task-db.md, section 8).

The served model, with the full journal in context (teacher.generate(...,
with_context=True), the same cached prefix as the Teacher), writes one task per
Recipe: a kind, a style, focus pages and a few human exemplars, all in the
user turn. plan_recipes, render_prompt and parse_draft are pure; synthesize()
takes its IO as arguments.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from cartridges.cartridge.chunks import _FENCE, _THINK
from cartridges.cartridge.tasks.sampler import quotas
from cartridges.cartridge.tasks.taxonomy import (
    CHUNK_PAGES,
    CHUNKS_PER_KIND,
    FRAMEWORKS,
    KIND_INSTRUCTIONS,
    MIN_YEARS_APART,
    ROLES,
    STYLE_INSTRUCTIONS,
)
from cartridges.cartridge.tasks.types import (
    Draft,
    Kind,
    Messages,
    Page,
    Recipe,
    Sample,
    Style,
    SynthConfig,
    SynthReport,
    Task,
    TaskQuery,
)
from cartridges.cartridge.tasks.validate import validate_draft, word_set

POOL = 1_000_000  # "all of them" for a sample query

logger = logging.getLogger(__name__)

PROMPT = """Write one message that the author of the journals above might send today to an AI assistant \
that knows their journals. It will be used as a training task, so it must read like the author's own \
message: personal, specific and in their voice.

What to ask: {kind}
How to write it: {style}

Write in the first person (I, me, my), as the author. Use real names, places and details from their life \
where they fit. Never mention journals, pages, entries or "the author". Do not ask about a date marked \
"(?)" or "(est.)". The examples below only show how the author talks; do not copy them.

Examples of messages the author has sent:
{exemplars}

Focus on these pages (the whole journal above is context):

{focus}

Reply with JSON only: {{"task": "...", "citations": ["<journal>:p.<N>", ...], "evidence": ["...", ...]}}. \
citations: the focus pages a good answer would draw on. evidence: one short verbatim quote from each cited \
page, in the same order."""


# ---------------------------------------------------------------- plan


def _run(pages: Sequence[Page], i: int, allowed: Sequence[bool]) -> list[int]:
    """Up to CHUNK_PAGES consecutive allowed pages of one journal from i."""
    out = [i]
    while (
        len(out) < CHUNK_PAGES and out[-1] + 1 < len(pages) and allowed[out[-1] + 1]
        and pages[out[-1] + 1].journal == pages[i].journal
    ):
        out.append(out[-1] + 1)
    return out


def _starts(kind: Kind, k: int, pages: Sequence[Page], allowed: Sequence[bool], rng: random.Random) -> list[int]:
    """k start pages: change_over_time two at least MIN_YEARS_APART apart;
    other multi-chunk kinds in distinct years where possible."""
    ok = [i for i, a in enumerate(allowed) if a]
    dated = [i for i in ok if pages[i].date]
    if kind == "change_over_time" and dated:
        a = rng.choice(dated)
        far = [i for i in dated if abs((pages[i].date - pages[a].date).days) >= 365 * MIN_YEARS_APART]
        return [a, rng.choice(far or dated)]
    if k == 1:
        return [rng.choice(ok)]
    years = sorted({pages[i].date.year for i in dated})
    starts = [rng.choice([i for i in dated if pages[i].date.year == y]) for y in rng.sample(years, min(k, len(years)))]
    return starts + rng.sample(ok, k - len(starts))


def plan_recipes(
    n: int,
    kind_mix: Mapping[Kind, float],
    style_mix: Mapping[Style, float],
    pages: Sequence[Page],
    exemplars_by_kind: Mapping[Kind, Sequence[Task]],
    seed: int,
    exemplars_per_prompt: int = 3,
) -> list[Recipe]:
    """n recipes, kinds by largest-remainder quotas of the mix, deterministic
    for the seed. Recall focus pages are never held out or uncertainly dated.
    A kind without exemplars borrows them from the other kinds."""
    rng = random.Random(seed)
    kinds = [k for k, q in quotas(n, kind_mix).items() for _ in range(q)]
    rng.shuffle(kinds)
    styles, weights = list(style_mix), list(style_mix.values())
    every = [t for ts in exemplars_by_kind.values() for t in ts]
    out = []
    for kind in kinds:
        allowed = [not (p.heldout or p.date_uncertain) if kind == "recall" else True for p in pages]
        lo, hi = CHUNKS_PER_KIND[kind]
        focus = sorted({j for i in _starts(kind, rng.randint(lo, hi), pages, allowed, rng) for j in _run(pages, i, allowed)})
        pool = list(exemplars_by_kind.get(kind) or every)
        out.append(Recipe(
            kind=kind,
            style=rng.choices(styles, weights)[0],
            framework=rng.choice(FRAMEWORKS) if kind == "framework_lens" else None,
            role=rng.choice(ROLES) if kind == "role_request" else None,
            citations=tuple(pages[j].citation for j in focus),
            exemplar_ids=tuple(t.task_id for t in rng.sample(pool, min(exemplars_per_prompt, len(pool)))),
        ))
    return out


# ---------------------------------------------------------------- prompt and reply


def render_prompt(recipe: Recipe, focus: Sequence[Page], exemplars: Sequence[str]) -> Messages:
    """The user turn; the journal prefix is added by teacher.generate(with_context=True)."""
    kind = KIND_INSTRUCTIONS[recipe.kind].format(framework=recipe.framework, role=recipe.role)
    pages = "\n\n---\n".join(
        f"## [{p.citation}] date: {p.date or 'unknown'}{' (?)' if p.date_uncertain else ''}\n\n{p.text}" for p in focus
    )
    content = PROMPT.format(
        kind=kind,
        style=STYLE_INSTRUCTIONS[recipe.style],
        exemplars="\n".join(f"- {e}" for e in exemplars) or "- (none)",
        focus=pages,
    )
    return [{"role": "user", "content": content}]


class DraftReply(BaseModel):
    """The reply asked for in PROMPT; its JSON schema constrains decoding."""

    model_config = ConfigDict(extra="forbid")

    task: str
    citations: list[str] = Field(max_length=4)
    evidence: list[Annotated[str, Field(max_length=300)]] = Field(max_length=4)


def parse_draft(reply: str) -> Draft | None:
    """The JSON reply as a Draft; None unless it has a string "task"."""
    try:
        obj = json.loads(_FENCE.sub("", _THINK.sub("", reply).strip()))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("task"), str):
        return None

    def strings(key: str) -> tuple[str, ...]:
        value = obj.get(key)
        return tuple(v for v in value if isinstance(v, str)) if isinstance(value, list) else ()

    # Models copy the page heading's brackets: "[Morning Pages:p.63]".
    citations = tuple(c.strip().strip("[]").strip() for c in strings("citations"))
    return Draft(text=obj["task"], ref_citations=citations, ref_evidence=strings("evidence"))


# ---------------------------------------------------------------- orchestrator


def distinct_2(texts: Sequence[str]) -> float:
    grams = [tuple(w[i : i + 2]) for t in texts for w in [t.lower().split()] for i in range(len(w) - 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def _exemplars(sample: Callable[[TaskQuery], Sample], kind: Kind, seed: int) -> tuple[Task, ...]:
    """The kind's exemplar templates, plus human exemplars that have none (seed.py
    leaves a question with nothing to replace untemplated)."""
    query = TaskQuery(split="exemplar", origins=frozenset({"human", "human_template"}), n=POOL, seed=seed, mix={kind: 1.0})
    tasks = sample(query).tasks
    templated = {p for t in tasks if t.origin == "human_template" for p in t.parent_ids}
    return tuple(t for t in tasks if t.task_id not in templated)


def synthesize(
    config: SynthConfig,
    *,
    fetch_pages: Callable[[str], list[Page]],
    fetch_hashes: Callable[[], set[str]],
    sample: Callable[[TaskQuery], Sample],
    generate: Callable[..., list],
    insert: Callable[[list[Task]], int],
) -> SynthReport:
    """Plan, prompt, generate, validate and insert config.n tasks (validated
    and rejected), config.batch_size prompts at a time, inserting each batch as
    it arrives. The gen_runs row for config.run_id must exist. `generate`
    takes a batch of message lists and returns one completion (with .text) or
    None per prompt, like data.modal_generate."""
    pages = fetch_pages(config.corpus_id)
    by_citation = {p.citation: p for p in pages}
    exemplars = {kind: _exemplars(sample, kind, config.seed) for kind in config.kind_mix}
    texts = {t.task_id: t.text for ts in exemplars.values() for t in ts}
    eval_words = [word_set(t.text) for t in sample(TaskQuery(split="eval", n=POOL, seed=0)).tasks]
    exemplar_words = [word_set(t.text) for t in sample(TaskQuery(split="exemplar", n=POOL, seed=0)).tasks]

    recipes = plan_recipes(config.n, config.kind_mix, config.style_mix, pages, exemplars, config.seed,
                           config.exemplars_per_prompt)
    prompts = [render_prompt(r, [by_citation[c] for c in r.citations], [texts[i] for i in r.exemplar_ids])
               for r in recipes]
    tasks, drafts, n_inserted = synth_batches(
        config, recipes, prompts, by_citation, set(fetch_hashes()), eval_words, exemplar_words, generate, insert
    )
    return synth_report(tasks, drafts, n_inserted)


def synth_batches(
    config: SynthConfig,
    recipes: Sequence[Recipe],
    prompts: Sequence[Messages],
    pages: Mapping[str, Page],
    seen: set[str],
    eval_words: Collection[frozenset[str]],
    exemplar_words: Collection[frozenset[str]],
    generate: Callable[..., list],
    insert: Callable[[list[Task]], int],
    metadata: Sequence[dict] | None = None,
) -> tuple[list[Task], list[Draft | None], int]:
    """Generate, validate and insert one task per (recipe, prompt),
    config.batch_size at a time, inserting each batch as it arrives (a crash
    keeps what was paid for). `seen` (text hashes) grows as tasks are made;
    metadata[i] goes on task i. Returns (tasks, drafts, rows inserted)."""
    tasks: list[Task] = []
    drafts: list[Draft | None] = []
    n_inserted = 0
    for start in range(0, len(recipes), config.batch_size):
        batch = slice(start, start + config.batch_size)
        replies = generate(list(prompts[batch]), max_tokens=config.max_tokens, temperature=config.temperature,
                           with_context=True, json_schema=DraftReply.model_json_schema())
        new = []
        for i, (recipe, reply) in enumerate(zip(recipes[batch], replies, strict=True), start):
            raw = reply.text if reply is not None else ""
            draft = parse_draft(raw)
            drafts.append(draft)
            task = validate_draft(draft, recipe, pages, seen, eval_words, raw=raw, exemplar_words=exemplar_words)
            seen.add(task.text_hash)
            update = {"corpus_id": config.corpus_id, "run_id": config.run_id}
            new.append(task.model_copy(update=update | ({"metadata": metadata[i]} if metadata else {})))
        n_inserted += insert(new)
        tasks += new
        logger.info("synthesized %d/%d (%d validated)", len(tasks), len(recipes),
                    sum(t.status == "validated" for t in tasks))
    return tasks, drafts, n_inserted


def synth_report(tasks: Sequence[Task], drafts: Sequence[Draft | None], n_inserted: int) -> SynthReport:
    valid = [t for t in tasks if t.status == "validated"]
    parsed = [t for t, d in zip(tasks, drafts) if d is not None]
    return SynthReport(
        n_drafts=len(tasks),
        n_validated=len(valid),
        n_inserted=n_inserted,
        by_kind=dict(Counter(t.kind for t in valid)),
        reject_reasons=dict(Counter(t.reject_reason for t in tasks if t.reject_reason)),
        grounded_rate=sum(bool(t.grounded) for t in parsed) / len(parsed) if parsed else 0.0,
        distinct_2=distinct_2([t.text for t in valid]),
    )


def revalidate(
    stored: Sequence[Task],
    pages: Mapping[str, Page],
    other_hashes: Collection[str],
    eval_words: Collection[frozenset[str]],
    exemplar_words: Collection[frozenset[str]],
) -> list[Task]:
    """Stored synthetic drafts (in creation order) re-checked with the current
    parse and validation, with no GPU. An unparseable row's text is its raw
    reply, so it is parsed again; the others are rebuilt from their text,
    citations and quotes. Ids, provenance and run stay; `other_hashes` are the
    text hashes of every other task (duplicates)."""
    seen = set(other_hashes)
    out = []
    for t in stored:
        recipe = Recipe(kind=t.kind, style=t.style, framework=t.framework, role=t.role,
                        citations=t.source_citations, exemplar_ids=t.parent_ids)
        if t.reject_reason == "unparseable":
            draft = parse_draft(t.text)
        else:
            draft = Draft(text=t.text, ref_citations=tuple(c.strip().strip("[]").strip() for c in t.ref_citations),
                          ref_evidence=t.ref_evidence)
        new = validate_draft(draft, recipe, pages, seen, eval_words, raw=t.text, exemplar_words=exemplar_words)
        if new.reject_reason == "duplicate":  # keep the stored text: text_hash is unique
            new = new.model_copy(update={"text": t.text, "text_hash": t.text_hash})
        seen.add(new.text_hash)
        out.append(new.model_copy(update={"task_id": t.task_id, "corpus_id": t.corpus_id, "run_id": t.run_id}))
    return out
