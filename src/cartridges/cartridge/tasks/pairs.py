"""Question-generator eval: blinded journal / no-journal answer pairs, annotated
by a human in a W&B Weave annotation queue.

For each validated task of one synthesis run, the deployed Teacher answers
twice, greedily and with the same token budget: with the full journal in context
(the teacher's ICL answer) and without it (the bare model). The two answers are
shown as A and B in a seeded random order. The mapping is kept only in a
local JSONL, never in Weave (the user is the annotator). The human
rates whether the question is realistic, which answer is better, and which has
factual errors. export_annotations() un-blinds the feedback.

Tasks come from the Task Sampler: run_query() asks for the synthetic train
tasks of the given runs in the default kind mix (taxonomy.KIND_MIX).

Pure stages:
    run_query(run_ids, n, seed)                the TaskQuery
    blind(tasks, journal, bare, seed)          -> Pair (A/B order seeded per task)
    unblind(pair, values)                      feedback -> journal vs no_journal row
Orchestrator (IO injected; tests pass fakes):
    make_pairs(tasks, generate, seed)          two batches of teacher calls
Weave IO:
    log_pairs, publish_specs, create_queue, read_feedback, export_annotations
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cartridges.cartridge.tasks.types import Sample, Task, TaskQuery

PROJECT = "conle-ai/cartridges-selfstudy"
QUEUE = "qgen-eval-v1"
MAX_TOKENS = 1536
OUT_DIR = Path("data/cartridge/annotations")

REALISM = ("I'd ask this", "plausible", "never")
PREFERENCE = ("A much better", "A better", "tie", "B better", "B much better")
FACTUAL_ERRORS = ("none", "A", "B", "both")
# name -> (description, JSON schema of the value)
SPECS: dict[str, tuple[str, dict]] = {
    "question_realism": ("Would you ask this question about your own journal?",
                         {"type": "string", "enum": list(REALISM)}),
    "preference": ("Which answer is better for you?", {"type": "string", "enum": list(PREFERENCE)}),
    "factual_errors": ("Which answers state something false about your life?",
                       {"type": "string", "enum": list(FACTUAL_ERRORS)}),
    "notes": ("One sentence on why.", {"type": "string"}),
}


class Pair(BaseModel):
    """One blinded pair. answer_a/answer_b are what the annotator sees."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    question: str
    kind: str
    style: str | None
    framework: str | None
    role: str | None
    focus_pages: tuple[str, ...]
    journal_answer: str
    no_journal_answer: str
    journal_is_a: bool
    sample_hash: str = ""  # the Sample the task came from: a pairs file resumes only for the same sample
    call_id: str | None = None  # set once logged to Weave

    @property
    def answer_a(self) -> str:
        return self.journal_answer if self.journal_is_a else self.no_journal_answer

    @property
    def answer_b(self) -> str:
        return self.no_journal_answer if self.journal_is_a else self.journal_answer


# ---------------------------------------------------------------- pure


def run_query(run_ids: Sequence[str], n: int, seed: int) -> TaskQuery:
    """n validated synthetic train tasks of the runs, in the default kind mix."""
    from cartridges.cartridge.tasks.taxonomy import KIND_MIX

    return TaskQuery(split="train", origins=frozenset({"synthetic"}), run_ids=frozenset(run_ids), n=n, seed=seed,
                     mix=KIND_MIX)


def blind(tasks: Sequence[Task], journal: Sequence[str], bare: Sequence[str], seed: int,
          sample_hash: str = "") -> list[Pair]:
    """Pairs with the journal answer as A or B by a coin flip per task."""
    rng = random.Random(seed)
    return [
        Pair(task_id=str(t.task_id), question=t.text, kind=t.kind, style=t.style, framework=t.framework,
             role=t.role, focus_pages=t.source_citations, journal_answer=j, no_journal_answer=b,
             journal_is_a=rng.random() < 0.5, sample_hash=sample_hash)
        for t, j, b in zip(tasks, journal, bare, strict=True)
    ]


def make_pairs(tasks: Sequence[Task], generate: Callable, seed: int, sample_hash: str = "") -> tuple[list[Pair], int]:
    """Answer every task with and without the journal (greedy, MAX_TOKENS), then
    blind. `generate(messages_list, **kw) -> list[Completion | None]` as
    data.modal_generate; the journal batch finishes before the no-journal batch
    starts. Tasks where either call failed are dropped; returns
    (pairs, n_dropped)."""
    # A bare user turn for both answers, without label.task_messages' citation
    # reminder: the no-journal model can't cite pages, so the reminder would turn its
    # answers into "the journals don't say".
    batch = [[{"role": "user", "content": t.text}] for t in tasks]
    # One batch, then the other, never overlapping: cascade attention only engages
    # when every running request shares the journal prefix. generate must return
    # only after all its calls are done (modal_generate drains its .map).
    journal = generate(batch, max_tokens=MAX_TOKENS, temperature=0.0, with_context=True)
    bare = generate(batch, max_tokens=MAX_TOKENS, temperature=0.0, with_context=False)
    kept = [(t, j.text, b.text) for t, j, b in zip(tasks, journal, bare, strict=True) if j is not None and b is not None]
    pairs = blind([t for t, _, _ in kept], [j for _, j, _ in kept], [b for _, _, b in kept], seed, sample_hash)
    return pairs, len(tasks) - len(kept)


_PREFERENCE_A = {"A much better": 2, "A better": 1, "tie": 0, "B better": -1, "B much better": -2}


def unblind(pair: Pair, values: dict) -> dict:
    """One export row: the annotator's values with A/B mapped back to
    journal/no_journal. preference is +2 (journal much better) .. -2."""
    side = {"A": "journal" if pair.journal_is_a else "no_journal", "B": "no_journal" if pair.journal_is_a else "journal"}
    pref = values.get("preference")
    errors = values.get("factual_errors")
    return {
        "task_id": pair.task_id,
        "question": pair.question,
        "kind": pair.kind,
        "journal_answer": pair.journal_answer,
        "no_journal_answer": pair.no_journal_answer,
        "realism": values.get("question_realism"),
        "preference": None if pref is None else _PREFERENCE_A[pref] * (1 if pair.journal_is_a else -1),
        "factual_errors": side.get(errors, errors),
        "notes": values.get("notes"),
        "call_id": pair.call_id,
    }


def write_pairs(path: Path, pairs: Sequence[Pair]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(p.model_dump_json() + "\n" for p in pairs))
    return path


def read_pairs(path: Path) -> list[Pair]:
    return [Pair.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- Weave IO


def log_pairs(pairs: Sequence[Pair], save: Callable[[list[Pair]], object] = lambda pairs: None) -> list[Pair]:
    """One `qgen_pair` op call per pair not yet logged (weave.init first). The
    inputs are the question and its recipe, the output is {answer_a,
    answer_b}; the un-blinding key is not sent. `save(pairs)` runs after each
    call, so a failure part way keeps the call ids logged so far. Returns the
    pairs with call_id."""
    import weave

    @weave.op(name="qgen_pair")
    def qgen_pair(question: str, kind: str, style: str | None, framework: str | None, role: str | None,
                  focus_pages: list[str]) -> dict:
        return {"answer_a": current.answer_a, "answer_b": current.answer_b}

    out = list(pairs)
    for i, current in enumerate(out):
        if current.call_id:
            continue
        with weave.attributes({"task_id": current.task_id}):  # the A/B key stays in the local JSONL only
            _, call = qgen_pair.call(question=current.question, kind=current.kind, style=current.style,
                                     framework=current.framework, role=current.role,
                                     focus_pages=list(current.focus_pages))
        out[i] = current.model_copy(update={"call_id": call.id})
        save(out)
    return out


def publish_specs() -> list[str]:
    """Publish the human annotation specs; returns their ref URIs."""
    import weave
    from weave.flow.annotation_spec import AnnotationSpec

    return [
        weave.publish(AnnotationSpec(name=name, description=desc, field_schema=schema), name).uri()
        for name, (desc, schema) in SPECS.items()
    ]


def create_queue(client, name: str, scorer_refs: list[str], call_ids: list[str]) -> str:
    """The annotation queue `name` holding `call_ids` (created if missing; calls
    already in it are added again only if the server allows duplicates)."""
    existing = client.list_annotation_queues(name=name)
    queue_id = existing[0].id if existing else client.create_annotation_queue(
        name=name, scorer_refs=scorer_refs,
        description="Journal vs no-journal answers to synthetic questions. Rate realism, preference, errors; say why.",
    )
    client.add_calls_to_annotation_queue(
        queue_id, call_ids=call_ids, display_fields=["inputs.question", "output.answer_a", "output.answer_b"]
    )
    return queue_id


def read_feedback(client, call_ids: Sequence[str]) -> dict[str, dict]:
    """{call_id: {spec name: value}}, the latest annotation of each spec."""
    prefix = "wandb.annotation."
    out = {}
    for call_id in call_ids:
        values, seen = {}, {}
        for f in client.get_call(call_id).feedback:
            name = f.feedback_type.removeprefix(prefix)
            if f.feedback_type.startswith(prefix) and name in SPECS and f.created_at >= seen.get(name, f.created_at):
                values[name], seen[name] = f.payload.get("value"), f.created_at
        out[call_id] = values
    return out


def export_annotations(queue: str = QUEUE, out_dir: Path = OUT_DIR, project: str = PROJECT) -> Path:
    """Read the queue's feedback and write <out_dir>/<queue>.jsonl, un-blinded
    with the pairs written by run()."""
    import weave

    client = weave.init(project)
    pairs = read_pairs(out_dir / f"{queue}.pairs.jsonl")
    feedback = read_feedback(client, [p.call_id for p in pairs])
    path = out_dir / f"{queue}.jsonl"
    path.write_text("".join(json.dumps(unblind(p, feedback.get(p.call_id, {}))) + "\n" for p in pairs))
    return path


def run(sample: Callable[[TaskQuery], Sample], run_ids: Sequence[str], generate: Callable, n: int = 100,
        seed: int = 0, queue: str = QUEUE, out_dir: Path = OUT_DIR, project: str = PROJECT) -> dict:
    """Pairs for n tasks of the synthesis runs (sampler.sample_tasks over the
    store), logged to Weave in an annotation queue. Raises if a kind falls
    short of its quota.

    Every pair (answers, A/B key, task_id) is written to
    <out_dir>/<queue>.pairs.jsonl before anything goes to Weave, and again as
    each call is logged. If that file exists for the same sample, generation
    is skipped and only the pairs without a call id are logged: a Weave
    failure never costs GPU time twice. A file for a different sample is never
    overwritten (it holds the key of an earlier queue)."""
    import weave

    s = sample(run_query(run_ids, n, seed))
    if sum(s.shortfall.values()):
        raise ValueError(f"kind quotas not met (shortfall {dict(s.shortfall)}): top up with another synthesis run")
    path = out_dir / f"{queue}.pairs.jsonl"
    if path.exists():
        pairs, dropped = read_pairs(path), None
        if any(p.sample_hash != s.sample_hash for p in pairs):
            raise FileExistsError(f"{path} holds pairs of another sample; move it away or use another queue name")
    else:
        pairs, dropped = make_pairs(s.tasks, generate, seed, s.sample_hash)
        write_pairs(path, pairs)
    client = weave.init(project)
    pairs = log_pairs(pairs, save=lambda ps: write_pairs(path, ps))
    queue_id = create_queue(client, queue, publish_specs(), [p.call_id for p in pairs])
    return {"run_ids": list(run_ids), "sample_hash": s.sample_hash, "pairs": len(pairs), "dropped": dropped,
            "resumed": dropped is None, "queue": queue, "queue_id": queue_id, "pairs_path": str(path)}
