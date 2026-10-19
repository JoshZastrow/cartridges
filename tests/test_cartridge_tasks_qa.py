import contextlib
import uuid

import pytest

from cartridges.cartridge.tasks import qa, store, targets
from cartridges.cartridge.tasks.types import Label, TargetRow, Task
from cartridges.cartridge.tasks.validate import task_id, text_hash


def task(text, origin="chat", split="train"):
    h = text_hash(text)
    return Task(task_id=task_id(h), text=text, text_hash=h, kind="recall", origin=origin, split=split,
                status="validated")


def label(t, uri=None, row=None, status="validated"):
    return Label(label_id=uuid.uuid5(uuid.NAMESPACE_URL, str(t.task_id)), task_id=t.task_id, run_id="r",
                 teacher_model="m", corpus_id="c", instructions_hash="h", answer=f"answer to {t.text}",
                 finish_reason="stop", n_prompt_tokens=3, n_completion_tokens=2, citations=("A:p.1",),
                 status=status, reject_reason=None if status == "validated" else "refusal",
                 targets_uri=uri, targets_row=row)


@pytest.fixture
def labeled(tmp_path):
    tasks = [task("what did I do at the lake?"), task("psych item", origin="psych_test"), task("refused")]
    uri = str(tmp_path / "part.parquet")
    labels = [label(tasks[0], uri, 0), label(tasks[1], uri, 1), label(tasks[2], status="rejected")]
    rows = [TargetRow(label_id=str(lab.label_id), task_id=str(lab.task_id), prompt_ids=[1, 2, 3],
                      completion_ids=[4, 5], topk_ids=[[4, 9], [5, 9]], topk_logprobs=[[-0.1, -2.0], [-0.2, -3.0]])
            for lab in labels[:2]]
    targets.write_shard(rows, uri)
    return tasks, labels


def test_pairs_join_labels_targets_and_tasks(labeled):
    tasks, labels = labeled
    by_id = {t.task_id: t for t in tasks}
    out = qa.pairs(labels, targets.read_labeled(labels), by_id)
    assert [p.origin for p in out] == ["chat", "psych_test"]  # the rejected label is left out
    p = out[0]
    assert (p.question, p.answer, p.citations) == ("what did I do at the lake?", "answer to what did I do at the lake?",
                                                   ("A:p.1",))
    assert p.prompt_ids == [1, 2, 3] and p.completion_ids == [4, 5] and p.topk_ids == [[4, 9], [5, 9]]
    only_chat = qa.pairs(labels, targets.read_labeled(labels), by_id, origins={"chat"})
    assert [p.task_id for p in only_chat] == [str(tasks[0].task_id)]


def test_load_qa_reads_the_store(labeled, monkeypatch):
    tasks, labels = labeled
    seen = {}

    def fetch_labels(conn, run_ids, split="train"):
        seen["args"] = (run_ids, split)
        return labels

    monkeypatch.setattr(store, "fetch_labels", fetch_labels)
    monkeypatch.setattr(store, "fetch_tasks_by_id", lambda conn, ids: {t.task_id: t for t in tasks if t.task_id in ids})
    out = qa.load_qa("eval", ["r1", "r2"], connect=lambda: contextlib.nullcontext(None))
    assert seen["args"] == (["r1", "r2"], "eval") and len(out) == 2


def test_table_rows(labeled):
    tasks, labels = labeled
    rows = qa.table_rows(labels, {t.task_id: t for t in tasks})
    assert len(rows) == 2 and set(rows[0]) == set(qa.TABLE_COLUMNS)
    assert rows[0]["citations"] == "A:p.1" and rows[0]["n_tokens"] == 2
