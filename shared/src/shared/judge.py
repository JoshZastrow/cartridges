"""Side-by-side rubric judge: system X's answers against system Y's (the
references), over a `generate` callable for the judge model.

Copied from `cartridges.cartridge.tasks.judge` (and the call shape of
`cartridges.cartridge.train_distill.task_judge`) at commit a9a7588. Changed:
the rubric is loaded from an explicit path (no cartridge rubric directory);
`with_context` (the judge seeing its teacher's context block) is a parameter
instead of always on; a task can carry `context` (e.g. the matching journal
passages) that the rubric template places with `{context}`; `judge()` takes
the answers already produced (the voice project transcribes them from a
served endpoint) instead of an answering callable. The journal-citation and
qgen-calibration helpers are not copied.

Each pair is judged twice with A and B swapped. A verdict counts only if both
orders agree, otherwise it is a tie and the inconsistency rate is reported as
judge noise. A factual error is charged to a system only if both orders flag
it.

Pure stages:
    load_rubric(path)                          -> Rubric (text, version = file stem, hash)
    judge_messages(rubric, question, a, b, context)
    parse_verdict(text)                        -> Verdict | None
    combine(item, v_xy, v_yx)                  both orders -> Outcome for X vs Y
    summarize(outcomes, seed)                  win/tie/loss, error rates, CIs, by origin and kind
    stratified_subset(items, n, seed)          even across sources, round-robin over kinds
Orchestrators:
    judge_pairs(items, generate, rubric)       one batch of 2N judge calls
    judge(answers, tasks, references, generate, rubric) -> (outcomes, summary, items)

`generate(messages_list, **kw) -> list[Completion | None]` is any batch
generator over the judge model whose completions have `.text`, e.g.
`shared.teacher.modal_generate(teacher_cls(...))`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

MAX_TOKENS = 256
NON_INFERIOR_NON_LOSS = 0.45  # X wins or ties at least this often ...
# ... and makes no more factual errors than Y.

Winner = Literal["A", "B", "tie"]
Errors = Literal["none", "A", "B", "both"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rubric(_Frozen):
    version: str
    text: str  # a str.format template with {question}, {answer_a}, {answer_b} and optionally {context}

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]


class Verdict(_Frozen):
    winner: Winner
    factual_errors: Errors
    reason: str = ""


class Task(_Frozen):
    """What the judge needs to know about one task besides the answers."""

    task_id: str
    origin: str = ""
    kind: str = ""
    question: str
    context: str = ""  # shown to the judge where the rubric has {context}


class Item(_Frozen):
    """One task with both systems' answers."""

    task_id: str
    origin: str
    kind: str
    question: str
    x: str  # system X's answer
    y: str  # system Y's answer
    context: str = ""


class Outcome(_Frozen):
    task_id: str
    origin: str
    kind: str
    winner: Literal["x", "y", "tie"]
    consistent: bool  # both orders gave the same winner
    x_error: bool
    y_error: bool
    parsed: bool  # both verdicts parsed
    reasons: tuple[str, str] = ("", "")
    verdicts: tuple[Verdict | None, Verdict | None] = (None, None)  # (A=X order, A=Y order), as the judge wrote them


# ---------------------------------------------------------------- pure


def load_rubric(path: str | Path) -> Rubric:
    path = Path(path)
    return Rubric(version=path.stem, text=path.read_text())


def judge_messages(rubric: Rubric, question: str, answer_a: str, answer_b: str, context: str = "") -> list[dict]:
    text = rubric.text.format(question=question, answer_a=answer_a, answer_b=answer_b, context=context)
    return [{"role": "user", "content": text}]


def parse_verdict(text: str | None) -> Verdict | None:
    """The verdict in a judge reply (thinking and code fences removed); None if
    it is not the expected JSON."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    try:
        return Verdict.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def combine(item: Item, v_xy: Verdict | None, v_yx: Verdict | None) -> Outcome:
    """v_xy judged A=X, B=Y; v_yx judged A=Y, B=X."""
    to_x = {"A": "x", "B": "y", "tie": "tie"}
    to_y = {"A": "y", "B": "x", "tie": "tie"}
    if v_xy is None or v_yx is None:
        return Outcome(task_id=item.task_id, origin=item.origin, kind=item.kind, winner="tie", consistent=False,
                       x_error=False, y_error=False, parsed=False, verdicts=(v_xy, v_yx))
    w1, w2 = to_x[v_xy.winner], to_y[v_yx.winner]
    flagged1 = {"none": set(), "A": {"x"}, "B": {"y"}, "both": {"x", "y"}}[v_xy.factual_errors]
    flagged2 = {"none": set(), "A": {"y"}, "B": {"x"}, "both": {"x", "y"}}[v_yx.factual_errors]
    errors = flagged1 & flagged2
    return Outcome(task_id=item.task_id, origin=item.origin, kind=item.kind, winner=w1 if w1 == w2 else "tie",
                   consistent=w1 == w2, x_error="x" in errors, y_error="y" in errors, parsed=True,
                   reasons=(v_xy.reason, v_yx.reason), verdicts=(v_xy, v_yx))


def bootstrap_ci(values: Sequence[float], seed: int = 0, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def _stats(outcomes: Sequence[Outcome], seed: int) -> dict:
    n = len(outcomes)
    if not n:
        return {"n": 0}
    non_loss = [float(o.winner != "y") for o in outcomes]
    err_diff = [float(o.x_error) - float(o.y_error) for o in outcomes]
    win, tie = sum(o.winner == "x" for o in outcomes) / n, sum(o.winner == "tie" for o in outcomes) / n
    x_err, y_err = sum(o.x_error for o in outcomes) / n, sum(o.y_error for o in outcomes) / n
    return {
        "n": n,
        "win": win,
        "tie": tie,
        "loss": 1 - win - tie,
        "non_loss": sum(non_loss) / n,
        "non_loss_ci": bootstrap_ci(non_loss, seed),
        "x_error_rate": x_err,
        "y_error_rate": y_err,
        "error_diff_ci": bootstrap_ci(err_diff, seed),
        "inconsistency": sum(not o.consistent for o in outcomes if o.parsed) / max(1, sum(o.parsed for o in outcomes)),
        "parse_failures": sum(not o.parsed for o in outcomes),
        "non_inferior": sum(non_loss) / n >= NON_INFERIOR_NON_LOSS and x_err <= y_err,
    }


def summarize(outcomes: Sequence[Outcome], seed: int = 0) -> dict[str, dict]:
    """Stats overall ("all"), per origin ("origin:<o>") and per kind ("kind:<k>")."""
    out = {"all": _stats(outcomes, seed)}
    for field in ("origin", "kind"):
        for value in sorted({getattr(o, field) for o in outcomes}):
            out[f"{field}:{value}"] = _stats([o for o in outcomes if getattr(o, field) == value], seed)
    return out


def stratified_subset(items: Sequence, n: int, seed: int = 0, source=lambda i: i.origin) -> list:
    """n items split as evenly as possible across sources (a source with fewer
    items gives all of them, the rest go to the others), and within a source
    round-robin over kinds in a seeded order. Deterministic for a seed."""
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for it in sorted(items, key=lambda i: str(i.task_id)):
        groups.setdefault(source(it), []).append(it)
    quota, left = dict.fromkeys(groups, 0), min(n, len(items))
    while left:
        open_ = [s for s in sorted(groups) if quota[s] < len(groups[s])]
        for s in open_[:left]:
            quota[s] += 1
        left -= min(left, len(open_))
    out = []
    for s in sorted(groups):
        by_kind: dict[str, list] = {}
        for it in groups[s]:
            by_kind.setdefault(it.kind, []).append(it)
        for v in by_kind.values():
            rng.shuffle(v)
        kinds, picked = sorted(by_kind), []
        while len(picked) < quota[s]:
            for k in kinds:
                if by_kind[k] and len(picked) < quota[s]:
                    picked.append(by_kind[k].pop())
        out += picked
    return out


# ---------------------------------------------------------------- orchestration


def judge_pairs(items: Sequence[Item], generate: Callable, rubric: Rubric, with_context: bool = False) -> list[Outcome]:
    """Judge every item in both orders in one batch of 2N calls (greedy,
    JSON-constrained). with_context: the judge also sees its own context block
    (a Teacher's full journal); per-item context goes through {context}."""
    batch = [judge_messages(rubric, i.question, i.x, i.y, i.context) for i in items]
    batch += [judge_messages(rubric, i.question, i.y, i.x, i.context) for i in items]
    replies = generate(batch, max_tokens=MAX_TOKENS, temperature=0.0, with_context=with_context,
                       json_schema=Verdict.model_json_schema())
    verdicts = [parse_verdict(r.text if r is not None else None) for r in replies]
    n = len(items)
    return [combine(i, verdicts[k], verdicts[n + k]) for k, i in enumerate(items)]


def judge(
    answers: Sequence[str], tasks: Sequence[Task], references: Sequence[str], generate: Callable, rubric: Rubric,
    with_context: bool = False, seed: int = 0,
) -> tuple[list[Outcome], dict[str, dict], list[Item]]:
    """X = `answers` vs Y = `references` on `tasks` (all in the same order).
    Returns (outcomes, summary, items)."""
    items = [Item(task_id=t.task_id, origin=t.origin, kind=t.kind, question=t.question, x=x, y=y, context=t.context)
             for t, x, y in zip(tasks, answers, references, strict=True)]
    outcomes = judge_pairs(items, generate, rubric, with_context)
    return outcomes, summarize(outcomes, seed), items


def judge_config(rubric: Rubric, judge_model: str, x: str, y: str) -> dict:
    """The run config that identifies a comparison."""
    return {"judge_model": judge_model, "judge_prompt_version": rubric.version, "judge_prompt_hash": rubric.hash,
            "x": x, "y": y}


def log_comparison(run, outcomes: Sequence[Outcome], summary: dict[str, dict], prefix: str = "judge") -> None:
    """Per-item W&B table <prefix>/items and summary keys <prefix>/<stratum>/<stat>
    (e.g. judge/origin:story/non_loss) on a shared.tracking.TrackedRun."""
    import wandb

    columns = list(Outcome.model_fields)
    table = wandb.Table(columns=columns, data=[[json.dumps(v) if isinstance(v, (tuple, list)) else v
                                                for v in o.model_dump(mode="json").values()] for o in outcomes])
    scalars = {f"{prefix}/{stratum}/{k}": v for stratum, stats in summary.items() for k, v in stats.items()
               if isinstance(v, (int, float))}
    scalars |= {f"{prefix}/{stratum}/{k}_{end}": v for stratum, stats in summary.items()
                for k in ("non_loss_ci", "error_diff_ci") if isinstance(stats.get(k), tuple)
                for end, v in zip(("low", "high"), stats[k], strict=True)}
    run.log(scalars | {f"{prefix}/items": table})
    run.run.summary.update(scalars)
