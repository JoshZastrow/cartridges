import datetime as dt
import json
import re
from collections import Counter
from types import SimpleNamespace

import pytest

from cartridges.cartridge.tasks.sampler import select
from cartridges.cartridge.tasks.synthesize import parse_draft, plan_recipes, render_prompt, synthesize
from cartridges.cartridge.tasks.taxonomy import KIND_MIX, STYLE_MIX
from cartridges.cartridge.tasks.types import Page, SynthConfig, Task
from cartridges.cartridge.tasks.validate import task_id, text_hash


def make_pages():
    """Two journals over 2021-2025, ten pages a year; some held out or uncertain."""
    pages = []
    for j, journal in enumerate(("A", "B")):
        for year in range(2021 + 2 * j, 2024 + 2 * j):
            for k in range(10):
                n = len([p for p in pages if p.journal == journal]) + 1
                pages.append(Page(
                    citation=f"{journal}:p.{n}", journal=journal, page_no=n, date=dt.date(year, 1 + k, 1),
                    date_uncertain=k == 3, heldout=k == 5, text=f"On this day in {year} I walked with friend {n}.",
                ))
    return pages


PAGES = make_pages()
BY_CITATION = {p.citation: p for p in PAGES}


def human(text, kind, origin="human_template", split="exemplar", parents=()):
    h = text_hash(text)
    return Task(task_id=task_id(h), text=text, text_hash=h, kind=kind, origin=origin, split=split,
                status="validated", parent_ids=parents)


ORIGINAL = human("What does my enneagram say about me?", "framework_lens", origin="human")
STORE = [
    ORIGINAL,
    human("What does my {framework} say about me?", "framework_lens", parents=(ORIGINAL.task_id,)),
    human("How am I self sabotaging", "self_synthesis", origin="human"),  # untemplated
    human("What did I do in Lisbon in May 2022?", "recall", split="eval", origin="human"),
]


def by_kind(tasks):
    out = {}
    for t in tasks:
        out.setdefault(t.kind, []).append(t)
    return out


def test_plan_recipes_follows_the_taxonomy():
    recipes = plan_recipes(200, KIND_MIX, STYLE_MIX, PAGES, by_kind(STORE[:3]), seed=0)
    assert recipes == plan_recipes(200, KIND_MIX, STYLE_MIX, PAGES, by_kind(STORE[:3]), seed=0)
    counts = Counter(r.kind for r in recipes)
    assert counts == {k: round(200 * w) for k, w in KIND_MIX.items()}
    for r in recipes:
        focus = [BY_CITATION[c] for c in r.citations]
        assert (r.framework is not None) == (r.kind == "framework_lens")
        assert (r.role is not None) == (r.kind == "role_request")
        assert len(r.exemplar_ids) >= 1
        if r.kind == "recall":
            assert not any(p.heldout or p.date_uncertain for p in focus)
            assert len({p.journal for p in focus}) == 1 and len(focus) <= 3
        if r.kind == "change_over_time":
            years = [p.date.year for p in focus]
            assert max(years) - min(years) >= 2


def test_render_prompt_has_focus_pages_and_exemplars():
    r = plan_recipes(1, {"framework_lens": 1.0}, STYLE_MIX, PAGES, by_kind(STORE[:2]), seed=1)[0]
    [msg] = render_prompt(r, [BY_CITATION[c] for c in r.citations], ["What does my {framework} say about me?"])
    assert msg["role"] == "user"
    assert all(f"## [{c}] date: " in msg["content"] for c in r.citations)
    assert r.framework in msg["content"] and "- What does my {framework} say about me?" in msg["content"]


def test_parse_draft():
    reply = '<think>\nok\n</think>\n```json\n{"task": "Why?", "citations": ["A:p.1", 3], "evidence": ["x"]}\n```'
    d = parse_draft(reply)
    assert (d.text, d.ref_citations, d.ref_evidence) == ("Why?", ("A:p.1",), ("x",))
    assert parse_draft('{"task": "Why?", "citations": [" [A:p.1] "], "evidence": ["x"]}').ref_citations == ("A:p.1",)
    assert parse_draft("not json") is None
    assert parse_draft('{"citations": []}') is None


def fake_generate(calls):
    def generate(prompts, **kw):
        calls.append(kw)
        out = []
        for i, [msg] in enumerate(prompts):
            if i == 0:
                out.append(None)  # a failed call
                continue
            c = re.search(r"## \[([^\]]+)\]", msg["content"]).group(1)
            task = f"Task {i}: why do I keep thinking about my walk with friend {c.split('.')[-1]}?"
            quote = BY_CITATION[c].text[:20]
            out.append(SimpleNamespace(text=json.dumps({"task": task, "citations": [c], "evidence": [quote]})))
        return out

    return generate


def test_synthesize_with_fakes():
    calls, inserted = [], []
    config = SynthConfig(n=20, seed=0, model="m", run_id="run-1", corpus_id="c1", kind_mix=KIND_MIX,
                         style_mix=STYLE_MIX, prompt_version="v")
    report = synthesize(
        config,
        fetch_pages=lambda corpus_id: PAGES,
        fetch_hashes=set,
        sample=lambda q: select(STORE, q),
        generate=fake_generate(calls),
        insert=lambda tasks: inserted.extend(tasks) or len(tasks),
    )
    from cartridges.cartridge.tasks.synthesize import DraftReply

    assert calls == [{"max_tokens": 512, "temperature": 0.6, "with_context": True,
                      "json_schema": DraftReply.model_json_schema()}]
    assert len(inserted) == report.n_drafts == report.n_inserted == 20
    assert all(t.run_id == "run-1" and t.corpus_id == "c1" and t.origin == "synthetic" for t in inserted)
    assert report.reject_reasons["unparseable"] == 1
    assert report.n_validated + sum(report.reject_reasons.values()) == 20
    # Exemplars: the template, not its original; the untemplated question as is.
    shown = {i for t in inserted for i in t.parent_ids}
    assert ORIGINAL.task_id not in shown and STORE[2].task_id in shown


def test_synthesize_inserts_each_batch_as_it_arrives():
    calls, inserted = [], []
    good = fake_generate(calls)

    def generate(prompts, **kw):
        if len(calls) == 2:
            raise TimeoutError("container gone")
        return good(prompts, **kw)

    config = SynthConfig(n=20, seed=0, model="m", run_id="run-1", corpus_id="c1", kind_mix=KIND_MIX,
                         style_mix=STYLE_MIX, prompt_version="v", batch_size=8)
    with pytest.raises(TimeoutError):
        synthesize(config, fetch_pages=lambda c: PAGES, fetch_hashes=set, sample=lambda q: select(STORE, q),
                   generate=generate, insert=lambda tasks: inserted.append(len(tasks)) or len(tasks))
    assert inserted == [8, 8]  # the two finished batches are kept


def test_revalidate_rechecks_stored_drafts_in_place():
    from cartridges.cartridge.tasks.synthesize import revalidate
    from cartridges.cartridge.tasks.validate import validate_draft

    recipe = plan_recipes(1, {"situation_help": 1.0}, STYLE_MIX, PAGES, by_kind(STORE[:3]), seed=0)[0]
    c = recipe.citations[0]
    reply = json.dumps({"task": "Why do I keep thinking about that walk?", "citations": [f"[{c}]"],
                        "evidence": [BY_CITATION[c].text[:20]]})
    # As stored by the old validator: brackets kept, so rejected as ungrounded.
    old = validate_draft(parse_draft(reply).model_copy(update={"ref_citations": (f"[{c}]",)}), recipe, BY_CITATION,
                         set()).model_copy(update={"run_id": "r", "corpus_id": "c"})
    unparsed = validate_draft(None, recipe, BY_CITATION, set(), raw=reply.replace("walk", "hike"))
    assert (old.reject_reason, unparsed.reject_reason) == ("ungrounded", "unparseable")

    fixed, reparsed = revalidate([old, unparsed], BY_CITATION, set(), [], [])
    assert fixed.status == "validated" and fixed.task_id == old.task_id and fixed.run_id == "r"
    assert fixed.ref_citations == (c,)
    assert reparsed.status == "validated" and reparsed.text.endswith("hike?")
    assert revalidate([old], BY_CITATION, {fixed.text_hash}, [], [])[0].reject_reason == "duplicate"
