"""The Task Sampler: pure, deterministic, stratified selection of tasks
(docs/plans/task-db.md, section 7).

select() keeps only validated tasks of the query's split that match every
filter, so a fetch that returns too much can't leak eval tasks. It cannot see
labels: `unlabeled_by` is applied by store.fetch_tasks only.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence

from cartridges.cartridge.tasks.types import KINDS, Kind, Sample, Task, TaskQuery


def quotas(n: int, mix: Mapping[Kind, float]) -> dict[Kind, int]:
    """n split by the mix with the largest-remainder method (ties: KINDS order)."""
    kinds = [k for k in KINDS if k in mix]
    exact = {k: n * mix[k] for k in kinds}
    out = {k: int(exact[k]) for k in kinds}
    by_remainder = sorted(kinds, key=lambda k: (-(exact[k] - out[k]), KINDS.index(k)))
    for k in by_remainder[: n - sum(out.values())]:
        out[k] += 1
    return out


def _matches(task: Task, query: TaskQuery) -> bool:
    return (
        task.status == "validated"
        and task.split == query.split
        and (query.origins is None or task.origin in query.origins)
        and (query.styles is None or task.style in query.styles)
        and (query.run_ids is None or task.run_id in query.run_ids)
        and task.task_id not in query.exclude
        and (query.origin_mix is None or task.origin in query.origin_mix)
        and (query.mix is None or query.origin_mix is not None or task.kind in query.mix)
    )


def select(candidates: Sequence[Task], query: TaskQuery) -> Sample:
    """Up to query.n tasks without replacement. With a mix, each kind gets its
    largest-remainder quota, uniform within the kind; a short kind is reported
    in `shortfall` and not refilled from others. With an origin_mix, each
    origin gets its quota first and the mix applies within it, except
    psych_test, which is spread evenly over frameworks and, within one, over
    instruments (so no large test dominates); shortfalls are per
    "origin/kind". Same candidates (any order) and query give the same Sample."""
    pool = sorted({t.task_id: t for t in candidates if _matches(t, query)}.values(), key=lambda t: t.task_id)
    rng = random.Random(query.seed)
    if query.origin_mix is None:
        tasks, short = _by_kind(pool, query.n, query.mix, rng)
        shortfall = dict(short)
    else:
        tasks, shortfall = [], {}
        for origin, want in split(query.n, query.origin_mix).items():
            of_origin = [t for t in pool if t.origin == origin]
            if origin == "psych_test":
                picked = _by_framework(of_origin, want, rng)
                short = {"framework_lens": want - len(picked)}
            else:
                mix = {k: w for k, w in (query.mix or {}).items() if w} or None
                picked, short = _by_kind(of_origin, want, mix, rng)
                if mix is None:
                    short = {"any": want - len(picked)}
            tasks += picked
            shortfall |= {f"{origin}/{k}": v for k, v in short.items()}
    ids = ",".join(str(t.task_id) for t in tasks)
    digest = hashlib.sha256(f"{_canonical(query)}|{ids}".encode()).hexdigest()
    return Sample(query=query, tasks=tuple(tasks), shortfall=shortfall, sample_hash=digest)


def split(n: int, weights: Mapping[str, float]) -> dict[str, int]:
    """n split by the weights with the largest-remainder method (ties: key order)."""
    keys = list(weights)
    total = sum(weights.values())
    exact = {k: n * weights[k] / total for k in keys}
    out = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - out[k]), keys.index(k)))[: n - sum(out.values())]:
        out[k] += 1
    return out


def _by_kind(pool: Sequence[Task], n: int, mix: Mapping[Kind, float] | None, rng: random.Random):
    if mix is None:
        return rng.sample(list(pool), min(n, len(pool))), {}
    tasks, shortfall = [], {}
    for kind, want in quotas(n, mix).items():
        of_kind = [t for t in pool if t.kind == kind]
        tasks += rng.sample(of_kind, min(want, len(of_kind)))
        shortfall[kind] = max(0, want - len(of_kind))
    return tasks, shortfall


def _by_framework(pool: Sequence[Task], n: int, rng: random.Random) -> list[Task]:
    """n tasks taken one at a time, round-robin over frameworks and, within a
    framework, over its instruments (metadata instrument_id), each queue
    shuffled: an even spread that refills from frameworks with items left."""
    queues: dict[str, dict[str, list[Task]]] = {}
    for t in pool:
        queues.setdefault(t.framework or "", {}).setdefault(t.metadata.get("instrument_id", ""), []).append(t)
    frameworks = sorted(queues)
    rng.shuffle(frameworks)
    turns = {f: 0 for f in frameworks}
    for f in frameworks:
        for items in queues[f].values():
            rng.shuffle(items)
    out: list[Task] = []
    while len(out) < n and any(any(q.values()) for q in queues.values()):
        for f in frameworks:
            instruments = [i for i in sorted(queues[f]) if queues[f][i]]
            if not instruments or len(out) >= n:
                continue
            out.append(queues[f][instruments[turns[f] % len(instruments)]].pop())
            turns[f] += 1
    return out


def _canonical(query: TaskQuery) -> str:
    """The query as JSON with sets sorted: frozenset order varies by process."""
    q = query.model_dump(mode="json")
    return json.dumps({k: sorted(v) if isinstance(v, list) else v for k, v in q.items()}, sort_keys=True)


def sample_tasks(fetch: Callable[[TaskQuery], list[Task]], query: TaskQuery) -> Sample:
    return select(fetch(query), query)
