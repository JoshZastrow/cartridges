"""Human seed (docs/plans/task-db.md, section 10) and the corpus pages.

The user's questions (data/cartridge/user_questions.jsonl) become origin
"human" tasks, split into `eval` and `exemplar` deterministically and
stratified by kind; a row's own `split` is kept. The kind is `kind_v2` when
present, else `kind`, and must be a taxonomy kind. Each exemplar is then
rewritten into a placeholder template (origin "human_template", parent_ids =
[original]) for the Synthesizer. A question with nothing to replace stays
untemplated: its template would equal it and text_hash is unique.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from cartridges.cartridge.chunks import approx_tokens
from cartridges.cartridge.tasks.sampler import quotas
from cartridges.cartridge.tasks.types import KINDS, Corpus, Messages, Page, Task
from cartridges.cartridge.tasks.validate import normalize_text, task_id, text_hash

N_EVAL = 8

TEMPLATE_PROMPT = """Rewrite this message as a reusable template. Replace each specific name, place, \
date, event, book, course and framework with a short placeholder in braces, such as {{person}}, \
{{place}}, {{month year}}, {{event}}, {{book}} or {{framework}}. Keep everything else exactly as written: \
wording, tone, length, typos and fillers. Reply with the template only.

Message:
{text}"""


def corpus_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_corpus(corpus_path: Path, heldout: set[str]) -> tuple[Corpus, list[Page]]:
    """The corpus (teacher_context.md) and its pages in timeline order; its
    directory holds the journal transcripts."""
    import datetime as dt

    from cartridges.batch.timeline import UNDATED, timeline_entries

    text = Path(corpus_path).read_text()
    corpus = Corpus(
        corpus_id=corpus_id(text),
        name=Path(corpus_path).name,
        n_tokens=approx_tokens(text),
    )
    pages = [
        Page(
            citation=f"{e.journal}:p.{n}",
            journal=e.journal,
            page_no=n,
            date=dt.date.fromisoformat(e.date) if e.date != UNDATED else None,
            date_uncertain=bool(e.marker),
            heldout=f"{e.journal}:p.{n}" in heldout,
            text=body,
        )
        for e in timeline_entries(Path(corpus_path).parent)
        for n, body in e.pages
    ]
    return corpus, pages


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def human_tasks(rows: Sequence[dict], n_eval: int = N_EVAL, seed: int = 0) -> tuple[list[Task], list[str]]:
    """Validated human tasks and the errors of rows left out (empty, unknown
    kind or split, duplicate). Rows without a split fill the eval quota left
    after rows that have one, by kind share (largest remainder), uniformly
    within a kind; the rest are exemplars."""
    errors, kept, seen = [], [], set()
    for i, row in enumerate(rows):
        text = str(row.get("question", "")).strip()
        kind = row.get("kind_v2") or row.get("kind")
        split = row.get("split")
        h = text_hash(text)
        if not text:
            errors.append(f"row {i}: empty question")
        elif kind not in KINDS:
            errors.append(f"row {i}: unknown kind {kind!r}")
        elif split not in (None, "eval", "exemplar"):
            errors.append(f"row {i}: unknown split {split!r}")
        elif h in seen:
            errors.append(f"row {i}: duplicate question")
        else:
            seen.add(h)
            kept.append((text, h, kind, split))

    open_rows = sorted((r for r in kept if r[3] is None), key=lambda r: r[1])
    n_left = max(0, n_eval - sum(r[3] == "eval" for r in kept))
    to_eval: set[str] = set()
    if open_rows and n_left:
        counts = Counter(r[2] for r in open_rows)
        rng = random.Random(seed)
        for kind, q in quotas(min(n_left, len(open_rows)), {k: c / len(open_rows) for k, c in counts.items()}).items():
            to_eval |= {r[1] for r in rng.sample([r for r in open_rows if r[2] == kind], q)}
    tasks = [
        Task(
            task_id=task_id(h), text=text, text_hash=h, kind=kind, origin="human",
            split=split or ("eval" if h in to_eval else "exemplar"), status="validated",
        )
        for text, h, kind, split in kept
    ]
    return tasks, errors


def template_prompt(task: Task) -> Messages:
    return [{"role": "user", "content": TEMPLATE_PROMPT.format(text=task.text)}]


def to_template(task: Task, reply: str) -> Task | None:
    """The exemplar's template task; None if the reply is empty or unchanged."""
    text = reply.strip()
    if not text or normalize_text(text) == normalize_text(task.text):
        return None
    h = text_hash(text)
    return Task(
        task_id=task_id(h), text=text, text_hash=h, kind=task.kind, origin="human_template", split="exemplar",
        status="validated", parent_ids=(task.task_id,), run_id=task.run_id,
    )


def seed(
    rows: Sequence[dict],
    *,
    run_id: str,
    generate: Callable[..., list] | None,
    insert: Callable[[list[Task]], int],
    n_eval: int = N_EVAL,
    seed: int = 0,
) -> dict:
    """Insert the human tasks (committed before any teacher call), then a
    template of each exemplar. The gen_runs row for run_id must exist.
    `generate` is as in synthesize(); None inserts the human tasks only."""
    tasks, errors = human_tasks(rows, n_eval, seed)
    tasks = [t.model_copy(update={"run_id": run_id}) for t in tasks]
    inserted_human = insert(tasks)  # before the teacher calls, which may fail or hang
    exemplars = [t for t in tasks if t.split == "exemplar"] if generate else []
    replies = generate([template_prompt(t) for t in exemplars], max_tokens=512, temperature=0.0,
                       with_context=False) if exemplars else []
    templates = [
        tpl for t, r in zip(exemplars, replies, strict=True) if r is not None
        for tpl in [to_template(t, r.text)] if tpl is not None
    ]
    return {
        "errors": errors,
        "splits": dict(Counter(t.split for t in tasks)),
        "kinds": dict(Counter(t.kind for t in tasks)),
        "inserted_human": inserted_human,
        "inserted_templates": insert(templates),
        "untemplated": sum(t.split == "exemplar" for t in tasks) - len(templates),
    }
