"""Question/answer pairs for the KV-cache stage: each validated teacher label of
a task, with the task's question, the teacher's answer and its top-k targets.

A thin layer over the training read path: store.fetch_labels (labels whose
task is in the split, train by default) and targets.read_labeled (their
Parquet rows, one per label, checked against the label id).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from cartridges.cartridge.tasks.types import Label, Split, Task


class QAPair(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    origin: str
    kind: str
    question: str
    answer: str
    citations: tuple[str, ...]
    prompt_ids: list[int]  # the chat-templated question, without the journal (the student's input)
    completion_ids: list[int]  # the teacher's answer
    topk_ids: list[list[int]]  # the teacher's top-k next tokens at each answer position
    topk_logprobs: list[list[float]]


def pairs(labels: Sequence[Label], examples: Iterable, tasks: Mapping[UUID, Task],
          origins: Iterable[str] | None = None) -> list[QAPair]:
    """Join labels (as from fetch_labels), their examples (targets.read_labeled
    yields one per validated label with a shard, in order) and their tasks."""
    keep = set(origins) if origins is not None else None
    labeled = [lab for lab in labels if lab.status == "validated" and lab.targets_uri is not None]
    out = []
    for lab, ex in zip(labeled, examples, strict=True):
        task = tasks[lab.task_id]
        if keep is not None and task.origin not in keep:
            continue
        out.append(QAPair(task_id=str(task.task_id), origin=task.origin, kind=task.kind, question=task.text,
                          answer=lab.answer, citations=lab.citations, prompt_ids=ex.prompt_ids,
                          completion_ids=ex.completion_ids, topk_ids=ex.topk_ids, topk_logprobs=ex.topk_logprobs))
    return out


def load_qa(split: Split, label_runs: Sequence[str], origins: Iterable[str] | None = None, connect=None) -> list[QAPair]:
    """The QA pairs of these label runs whose tasks are in `split`. Reads the
    shards at the labels' targets_uri (/cache/targets/... on the
    cartridges-cartridge-cache Volume), so call it where that Volume is mounted."""
    from cartridges.cartridge.tasks import store, targets

    with (connect or store.connect)() as conn:
        labels = store.fetch_labels(conn, label_runs, split=split)
        tasks = store.fetch_tasks_by_id(conn, {lab.task_id for lab in labels})
    return pairs(labels, targets.read_labeled(labels), tasks, origins)


TABLE_COLUMNS = ["question", "answer", "origin", "kind", "citations", "n_tokens"]


def table_rows(labels: Sequence[Label], tasks: Mapping[UUID, Task]) -> list[dict]:
    """One browsable row per validated label (no top-k: that stays in Parquet)."""
    return [
        {"question": tasks[lab.task_id].text, "answer": lab.answer, "origin": tasks[lab.task_id].origin,
         "kind": tasks[lab.task_id].kind, "citations": ", ".join(lab.citations), "n_tokens": lab.n_completion_tokens}
        for lab in labels if lab.status == "validated"
    ]
