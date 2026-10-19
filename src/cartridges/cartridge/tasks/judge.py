"""Side-by-side judge for open-ended eval tasks (no reference answer): system X
(e.g. the cartridge) against system Y (e.g. the teacher with the journal in
context).

The judge model reads the journal (with_context=True, so it can check
grounding) and a versioned rubric (rubrics/judge-<version>.md; its version and
hash go into the run config). It compares the two answers blind, and each pair
is judged twice with A and B swapped. A verdict counts only if both orders
agree: otherwise it is a tie, and the inconsistency rate is reported as judge
noise. A factual error is charged to a system only if both orders flag it.

The judge is the same model as the teacher, so it may prefer the teacher's
answers. Gate: agreement() with the user's annotations of the qgen pairs
(tasks/pairs.py export, journal vs no-journal answers). The judge model is a
parameter of the caller's `generate`.

Pure stages:
    load_rubric(version)                      -> Rubric (text, version, hash)
    judge_messages(rubric, question, a, b)    the judge's user turn
    parse_verdict(text)                       -> Verdict | None
    combine(v_xy, v_yx)                       both orders -> Outcome for X vs Y
    summarize(outcomes, seed)                 win/tie/loss, error rates, CIs, by origin and kind
    calibration_items(annotations)            the qgen pairs as items (X = journal answer)
    agreement(outcomes, annotations)          judge vs human on the qgen pairs
    stratified_subset(items, n, seed)         even across sources, round-robin over kinds
    citation_summary(answers, strata, pages)  cite rate and validity of generated answers
Orchestrator:
    judge_pairs(items, generate, rubric)      one batch of 2N judge calls
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

RUBRIC_DIR = Path(__file__).parent / "rubrics"
MAX_TOKENS = 256
NON_INFERIOR_NON_LOSS = 0.45  # X wins or ties at least this often ...
# ... and makes no more factual errors than Y.

Winner = Literal["A", "B", "tie"]
Errors = Literal["none", "A", "B", "both"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rubric(_Frozen):
    version: str
    text: str  # a str.format template with {question}, {answer_a}, {answer_b}

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]


class Verdict(_Frozen):
    winner: Winner
    factual_errors: Errors
    reason: str = ""


class Item(_Frozen):
    """One task with both systems' answers."""

    task_id: str
    origin: str
    kind: str
    question: str
    x: str  # system X's answer
    y: str  # system Y's answer


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


def load_rubric(version: str = "v0", root: Path = RUBRIC_DIR) -> Rubric:
    return Rubric(version=version, text=(root / f"judge-{version}.md").read_text())


def judge_messages(rubric: Rubric, question: str, answer_a: str, answer_b: str) -> list[dict]:
    return [{"role": "user", "content": rubric.text.format(question=question, answer_a=answer_a, answer_b=answer_b)}]


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


def calibration_items(annotations: Sequence[dict]) -> list[Item]:
    """The qgen pairs as judge items: X = the journal answer, Y = the no-journal answer."""
    return [Item(task_id=r["task_id"], origin="synthetic", kind=r["kind"], question=r["question"],
                 x=r["journal_answer"], y=r["no_journal_answer"]) for r in annotations]


def agreement(outcomes: Sequence[Outcome], annotations: Sequence[dict]) -> dict:
    """Judge vs human on the qgen pairs, with X = the journal answer and Y = the
    no-journal answer. annotations are pairs.export_annotations rows; the
    human's preference (+2..-2, journal minus no-journal) is reduced to its sign.
    Returns accuracy and Cohen's kappa over the three classes, on pairs both
    labelled."""
    human = {r["task_id"]: r["preference"] for r in annotations if r.get("preference") is not None}
    sign = {"x": 1, "y": -1, "tie": 0}
    pairs = [(sign[o.winner], (h > 0) - (h < 0)) for o in outcomes if (h := human.get(o.task_id)) is not None]
    if not pairs:
        return {"n": 0, "accuracy": None, "kappa": None}
    n = len(pairs)
    observed = sum(j == h for j, h in pairs) / n
    expected = sum((sum(j == c for j, _ in pairs) / n) * (sum(h == c for _, h in pairs) / n) for c in (-1, 0, 1))
    return {"n": n, "accuracy": observed, "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


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


def citation_summary(answers: Sequence[str], strata: Sequence[str], pages: set[str]) -> dict[str, dict]:
    """Citation validity of generated answers: cite rate (any [journal:p.N]) and
    validity (cites, and every cited page exists in the corpus), overall and per
    stratum (e.g. the task's source)."""
    from cartridges.cartridge.evals import parse_citations

    rows = [(s, parse_citations(a)) for a, s in zip(answers, strata, strict=True)]

    def stats(cells: list) -> dict:
        n = len(cells)
        return {"n": n, "citation_rate": sum(bool(c) for _, c in cells) / n if n else None,
                "citation_validity": sum(bool(c) and all(p in pages for p in c) for _, c in cells) / n if n else None}

    return {"all": stats(rows)} | {f"origin:{s}": stats([r for r in rows if r[0] == s]) for s in sorted(set(strata))}


# ---------------------------------------------------------------- orchestration


def judge_pairs(items: Sequence[Item], generate: Callable, rubric: Rubric) -> list[Outcome]:
    """Judge every item in both orders in one batch of 2N calls (all with the
    journal in context, greedy, JSON-constrained). `generate(messages_list,
    **kw) -> list[Completion | None]` as data.modal_generate over the judge
    model."""
    batch = [judge_messages(rubric, i.question, i.x, i.y) for i in items]
    batch += [judge_messages(rubric, i.question, i.y, i.x) for i in items]
    replies = generate(batch, max_tokens=MAX_TOKENS, temperature=0.0, with_context=True,
                       json_schema=Verdict.model_json_schema())
    verdicts = [parse_verdict(r.text if r is not None else None) for r in replies]
    n = len(items)
    return [combine(i, verdicts[k], verdicts[n + k]) for k, i in enumerate(items)]


def judge_config(rubric: Rubric, judge_model: str, x: str, y: str) -> dict:
    """The run config that identifies a comparison."""
    return {"judge_model": judge_model, "judge_prompt_version": rubric.version, "judge_prompt_hash": rubric.hash,
            "x": x, "y": y}


def log_comparison(run, outcomes: Sequence[Outcome], summary: dict[str, dict], prefix: str = "judge") -> None:
    """Per-item W&B table <prefix>/items and summary keys <prefix>/<stratum>/<stat>
    (e.g. judge/origin:chat/non_loss) on a tracking.TrackedRun."""
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


SIDE_BY_SIDE_SYSTEMS = ("teacher", "cartridge", "base")


def side_by_side_rows(tasks: Sequence, answers: dict[str, dict[str, dict]], vs_teacher: dict[str, dict[str, Outcome]],
                      pages: set[str]) -> list[dict]:
    """One review row per task: the question, every system's answer (answers[system][task_id] =
    {"text", "finish_reason", "n_tokens"}), each comparison's verdicts in both orders
    (vs_teacher[system][task_id], X = that system, Y = the teacher), the consistency
    flag, and per answer its citation validity, citation count, length and truncation."""
    from cartridges.cartridge.evals import parse_citations

    rows = []
    for t in tasks:
        tid = str(t.task_id)
        row = {"task_id": tid, "origin": t.origin, "kind": t.kind, "question": t.text}
        for system in SIDE_BY_SIDE_SYSTEMS:
            a = answers.get(system, {}).get(tid)
            text = a["text"] if a else None
            cites = parse_citations(text) if text else []
            row |= {f"{system}_answer": text, f"{system}_n_tokens": a.get("n_tokens") if a else None,
                    f"{system}_truncated": (a.get("finish_reason") == "length") if a else None,
                    f"{system}_n_citations": len(cites),
                    f"{system}_citation_valid": bool(cites) and all(c in pages for c in cites)}
        for system, by_task in vs_teacher.items():
            o = by_task.get(tid)
            v1, v2 = o.verdicts if o else (None, None)
            row |= {f"{system}_vs_teacher_winner": o.winner if o else None,
                    f"{system}_vs_teacher_consistent": o.consistent if o else None,
                    f"{system}_vs_teacher_x_error": o.x_error if o else None,
                    f"{system}_vs_teacher_y_error": o.y_error if o else None}
            for name, v in ((f"{system}_first", v1), (f"{system}_swapped", v2)):  # first: A = system, B = teacher
                row |= {f"{name}_winner": v.winner if v else None, f"{name}_factual_errors": v.factual_errors if v else None,
                        f"{name}_reason": v.reason if v else None}
        rows.append(row)
    return rows
