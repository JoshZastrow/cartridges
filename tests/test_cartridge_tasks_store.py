"""Task Store round trip. The DDL check always runs; the database test runs only
if CARTRIDGES_TEST_DATABASE_URL is set (never DATABASE_URL). Use a Neon branch
or a local Postgres."""

import datetime as dt
import os
import re
import uuid

import pytest

from cartridges.cartridge.tasks import store
from cartridges.cartridge.tasks.sampler import sample_tasks
from cartridges.cartridge.tasks.types import ChatLabel, ChatPrompt, ChatRow, Corpus, GenRun, Label, Page, Task, TaskQuery
from cartridges.cartridge.tasks.validate import task_id, text_hash

DATABASE_URL = os.environ.get("CARTRIDGES_TEST_DATABASE_URL")


def test_schema_is_idempotent():
    statements = [re.sub(r"(?m)^--.*\n", "", s).strip() for s in store.SCHEMA.split(";") if s.strip()]
    # CREATE ... IF NOT EXISTS, a CHECK dropped (IF EXISTS) and re-added, or ADD/DROP COLUMN IF (NOT) EXISTS.
    allowed = (r"CREATE (SCHEMA|TABLE|INDEX) IF NOT EXISTS"
               r"|ALTER TABLE \S+ (DROP CONSTRAINT IF EXISTS|ADD CONSTRAINT|(ADD|DROP) COLUMN IF (NOT )?EXISTS)")
    assert all(re.match(allowed, s) for s in statements), statements


@pytest.mark.skipif(not DATABASE_URL, reason="CARTRIDGES_TEST_DATABASE_URL is not set")
def test_round_trip():
    prefix = f"test-{uuid.uuid4().hex[:8]}"
    corpus = Corpus(corpus_id=prefix, name="teacher_context.md", n_tokens=10)
    pages = [Page(citation=f"{prefix}:p.{n}", journal=prefix, page_no=n, date=dt.date(2022, 1, n) if n < 3 else None,
                  date_uncertain=n == 2, heldout=n == 1, text=f"page {n}") for n in (1, 2, 3)]

    def task(text, **kw):
        h = text_hash(f"{prefix} {text}")
        fields = {"kind": "recall", "origin": "synthetic", "split": "train", "status": "validated", "grounded": True,
                  "ref_citations": (pages[0].citation,), "ref_evidence": ("page 1",), **kw}
        return Task(task_id=task_id(h), text=f"{prefix} {text}", text_hash=h, run_id=prefix, corpus_id=prefix, **fields)

    conn = store.connect(DATABASE_URL)
    try:
        store.init(conn)
        store.init(conn)  # idempotent
        store.insert_corpus(conn, corpus, pages)
        store.insert_corpus(conn, corpus, [p.model_copy(update={"heldout": False}) for p in pages])
        assert [p.heldout for p in store.fetch_pages(conn, prefix)] == [False] * 3
        store.insert_run(conn, GenRun(run_id=prefix, component="synthesize", config={"n": 3}))

        a, b = task("a"), task("b", style="typed")
        rejected = task("c", status="rejected", reject_reason="empty")
        human = task("d", origin="human", split="eval", grounded=None, parent_ids=(a.task_id,))
        assert store.insert_tasks(conn, [a, b, rejected, human]) == 4
        assert store.insert_tasks(conn, [a]) == 0  # same text_hash
        assert text_hash(f"{prefix} c") in store.fetch_hashes(conn)

        fetch = lambda q: [t for t in store.fetch_tasks(conn, q) if t.text.startswith(prefix)]  # noqa: E731
        train = sample_tasks(fetch, TaskQuery(split="train", n=10, seed=0))
        assert {t.text for t in train.tasks} == {a.text, b.text}
        assert fetch(TaskQuery(split="eval", n=10, seed=0)) == [human]
        assert fetch(TaskQuery(split="train", n=10, seed=0, styles=frozenset({"typed"}))) == [b]
        assert len(fetch(TaskQuery(split="train", n=10, seed=0, run_ids=frozenset({prefix})))) == 2
        assert fetch(TaskQuery(split="train", n=10, seed=0, run_ids=frozenset({"other-run"}))) == []

        label = Label(label_id=uuid.uuid4(), task_id=a.task_id, run_id=prefix, teacher_model="m", corpus_id=prefix,
                      instructions_hash="h", answer="ok", finish_reason="stop", n_prompt_tokens=1,
                      n_completion_tokens=1, status="validated")
        assert store.insert_labels(conn, [label, label.model_copy(update={"label_id": uuid.uuid4()})]) == 1
        assert fetch(TaskQuery(split="train", n=10, seed=0, unlabeled_by="m")) == [b]
        assert fetch(TaskQuery(split="train", n=10, seed=0, unlabeled_by=("m", prefix, "h"))) == [b]
        assert len(fetch(TaskQuery(split="train", n=10, seed=0, unlabeled_by=("m", prefix, "new")))) == 2  # relabel
        listed = store.list_tasks(conn, origin="synthetic", run_ids=[prefix])  # any status
        assert {t.task_id for t in listed} == {a.task_id, b.task_id, rejected.task_id}
        fixed = rejected.model_copy(update={"status": "validated", "reject_reason": None, "grounded": True})
        assert store.update_validation(conn, [fixed]) == 1
        assert fixed in store.list_tasks(conn, run_ids=[prefix], status="validated", origin="synthetic")
        store.update_validation(conn, [rejected])
        chat = task("from chat", origin="chat", split="eval", grounded=None, ref_citations=(), ref_evidence=(),
                    metadata={"conversation_id": "conv-1"})
        assert store.insert_tasks(conn, [chat]) == 1
        assert store.list_tasks(conn, origin="chat", run_ids=[prefix]) == [chat]
        psy = task("psych item", origin="psych_test", kind="framework_lens", framework="VIA", grounded=None,
                   ref_citations=(), ref_evidence=(), metadata={"instrument_id": "ipip_via", "item_no": 1})
        assert store.insert_tasks(conn, [psy]) == 1
        mixed = fetch(TaskQuery(split="train", n=10, seed=0, origin_mix={"psych_test": 0.5, "synthetic": 0.5},
                                mix={"recall": 1.0}))
        assert psy in mixed and {t.origin for t in mixed} == {"psych_test", "synthetic"}  # kind filter off
        on_eval = label.model_copy(update={"label_id": uuid.uuid4(), "task_id": human.task_id})
        rejected_label = label.model_copy(update={"label_id": uuid.uuid4(), "task_id": b.task_id,
                                                  "status": "rejected", "reject_reason": "refusal"})
        store.insert_labels(conn, [on_eval, rejected_label])
        assert store.fetch_labels(conn, [prefix]) == [label]  # not the eval task's, not the rejected one
        assert store.fetch_labels(conn, [prefix], split="eval") == [on_eval]  # the held-out validation set
        human_train = task("human in train", origin="human", split="train", grounded=None, ref_citations=(),
                           ref_evidence=())
        store.insert_tasks(conn, [human_train])
        on_human = label.model_copy(update={"label_id": uuid.uuid4(), "task_id": human_train.task_id})
        store.insert_labels(conn, [on_human])
        assert on_human not in store.fetch_labels(conn, [prefix])  # human tasks never train
        relabel = on_human.model_copy(update={"status": "rejected", "reject_reason": "item_as_evidence"})
        assert store.update_labels(conn, [relabel]) == 1
        assert store.delete_labels(conn, [on_human.label_id]) == 1
        assert store.targets_manifest(conn, prefix)["labels"] == {"validated": 2, "rejected": 1}
        summary = store.label_summary(conn, [prefix])
        assert summary["synthetic"]["n"] == 2 and summary["synthetic"]["validated"] == 1
        assert summary["synthetic"]["cite_rate"] == 0 and summary["human"]["n"] == 1
        assert summary["synthetic"]["reject_reasons"] == {"refusal": 1}
        samples = store.label_samples(conn, prefix, per_origin=1)
        assert [(x["origin"], x["answer"]) for x in samples] == [("human", "ok"), ("synthetic", "ok")]  # validated
        assert store.fetch_labels(conn, [prefix], status="rejected") == [rejected_label]
        assert store.fetch_labels(conn, ["other-run"]) == []

        store.insert_run(conn, GenRun(run_id=f"{prefix}-c", component="classify", config={}))
        prompts = [ChatPrompt(conversation_id=f"{prefix}-{i}", create_time=None, title=None, model_slug=None,
                              text=f"{prefix} q{i}", text_hash=text_hash(f"{prefix} q{i}")) for i in range(3)]
        labels = [ChatLabel(category="relationships", task_kind="recall", relevant=True, reason="r", raw_reply="{}"),
                  ChatLabel(category="technical", relevant=False, raw_reply="{}"),
                  ChatLabel(category="unparsed", raw_reply="oops")]
        dup = prompts[0].model_copy(update={"conversation_id": f"{prefix}-dup"})  # same text, other conversation
        assert store.upsert_chat_prompts(conn, [*prompts, dup]) == 4

        def unlabeled(model, phash):
            todo, hits = store.fetch_unlabeled_chats(conn, model, phash)
            return [p.text_hash for p in todo if p.conversation_id.startswith(prefix)], hits

        assert sorted(unlabeled("m", "h1")[0]) == sorted(p.text_hash for p in prompts)  # one per text
        rows = [ChatRow(prompt=p, label=lab, classifier_model="m", prompt_hash="h1", run_id=f"{prefix}-c")
                for p, lab in zip(prompts, labels)]
        assert store.insert_chat_labels(conn, rows) == 2  # the unparsed one is not cached
        assert store.insert_chat_labels(conn, rows) == 2  # DO NOTHING on the second write
        todo, hits = unlabeled("m", "h1")
        assert todo == [prompts[2].text_hash] and hits >= 2
        assert len(unlabeled("m", "h2")[0]) == 3 and len(unlabeled("other", "h1")[0]) == 3
        promotable = [(p.conversation_id, lab.task_kind, run) for p, lab, run in store.fetch_promotable_chats(conn, "m", "h1")
                      if p.conversation_id.startswith(prefix)]
        # Relevant with a kind: the text's two conversations (promote() keeps one).
        assert sorted(promotable) == [(f"{prefix}-0", "recall", f"{prefix}-c"), (f"{prefix}-dup", "recall", f"{prefix}-c")]
    finally:
        conn.execute("DELETE FROM cartridge.chat_labels WHERE run_id = %s", (f"{prefix}-c",))
        conn.execute("DELETE FROM cartridge.chat_prompts WHERE conversation_id LIKE %s", (f"{prefix}%",))
        conn.execute("DELETE FROM cartridge.gen_runs WHERE run_id = %s", (f"{prefix}-c",))
        conn.execute("DELETE FROM cartridge.labels WHERE run_id = %s", (prefix,))
        conn.execute("DELETE FROM cartridge.tasks WHERE run_id = %s", (prefix,))
        conn.execute("DELETE FROM cartridge.gen_runs WHERE run_id = %s", (prefix,))
        conn.execute("DELETE FROM cartridge.corpus_pages WHERE corpus_id = %s", (prefix,))
        conn.execute("DELETE FROM cartridge.corpora WHERE corpus_id = %s", (prefix,))
        conn.close()
