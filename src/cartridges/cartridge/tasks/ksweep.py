"""The exemplar-count sweep: how many real questions to show the Synthesizer.

The pool is the user's real, non-eval questions (origin human, human_template
and chat). A question and its template stay together, and the pool is split by
seed into half A (shown as examples) and half B (a reference never shown). The
same recipes (kind, style, focus pages) are used for every k, so runs are
paired; only the k examples change, drawn per prompt from half A, the recipe's
kind first. Every draft is kept, validated or not, for review in W&B.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from uuid import UUID

from cartridges.cartridge.tasks.types import KINDS, Kind, Recipe, Task

EXEMPLAR_CHARS = 600
KIND_FLOOR = 0.05
REAL_ORIGINS = ("human", "human_template", "chat")
TABLE_COLUMNS = ["task_id", "question", "kind", "style", "k", "status", "reject_reason", "focus_pages", "exemplars",
                 "len_chars", "nn_sim_generated", "nn_sim_reference", "nn_sim_shown"]  # nn_*: tasks.neighbors
REFERENCE_COLUMNS = ["question", "kind", "origin", "split"]


def real_pool(tasks: Sequence[Task]) -> list[Task]:
    """Validated real questions outside eval."""
    return [t for t in tasks if t.origin in REAL_ORIGINS and t.split != "eval" and t.status == "validated"]


def split_halves(pool: Sequence[Task], seed: int) -> tuple[list[Task], list[Task]]:
    """(half A, half B) by question: a template goes with its original."""
    def root(t: Task) -> UUID:
        return t.parent_ids[0] if t.origin == "human_template" and t.parent_ids else t.task_id

    roots = sorted({root(t) for t in pool})
    random.Random(seed).shuffle(roots)
    a = set(roots[: len(roots) // 2])
    return [t for t in pool if root(t) in a], [t for t in pool if root(t) not in a]


def kind_mix(chat_tasks: Sequence[Task], floor: float = KIND_FLOOR) -> dict[Kind, float]:
    """The kinds' shares among the chat questions, each at least `floor`, summing to 1."""
    counts = Counter(t.kind for t in chat_tasks)
    total = sum(counts.values()) or 1
    raw = {k: max(counts[k] / total, floor) for k in KINDS}
    return {k: v / sum(raw.values()) for k, v in raw.items()}


def with_exemplars(recipes: Sequence[Recipe], half_a: Sequence[Task], k: int, seed: int) -> list[Recipe]:
    """The recipes with k examples each from half A, the recipe's kind first
    and other kinds after, deterministic for (seed, k, recipe index)."""
    out = []
    for i, r in enumerate(recipes):
        rng = random.Random(f"{seed}:{k}:{i}")
        same = sorted((t for t in half_a if t.kind == r.kind), key=lambda t: t.task_id)
        other = sorted((t for t in half_a if t.kind != r.kind), key=lambda t: t.task_id)
        picked = rng.sample(same, min(k, len(same)))
        picked += rng.sample(other, min(k - len(picked), len(other)))
        out.append(r.model_copy(update={"exemplar_ids": tuple(t.task_id for t in picked)}))
    return out


def exemplar_texts(recipe: Recipe, by_id: Mapping[UUID, Task]) -> list[str]:
    return [by_id[i].text[:EXEMPLAR_CHARS] for i in recipe.exemplar_ids]


def table_rows(tasks: Sequence[Task], recipes: Sequence[Recipe], by_id: Mapping[UUID, Task], k: int) -> list[dict]:
    """One W&B table row per draft; exemplars_shown (the texts, for
    tasks.neighbors) stays in the JSONL, `exemplars` is its readable join."""
    rows = []
    for t, r in zip(tasks, recipes, strict=True):
        shown = exemplar_texts(r, by_id)
        rows.append({"task_id": str(t.task_id), "question": t.text, "kind": t.kind, "style": t.style, "k": k,
                     "status": t.status, "reject_reason": t.reject_reason, "focus_pages": ", ".join(t.source_citations),
                     "exemplars": "\n---\n".join(shown), "exemplars_shown": shown, "len_chars": len(t.text)})
    return rows


def reference_rows(half_b: Sequence[Task], eval_tasks: Sequence[Task]) -> list[dict]:
    return [{"question": t.text, "kind": t.kind, "origin": t.origin, "split": "eval" if t.split == "eval" else "B"}
            for t in [*half_b, *eval_tasks]]
