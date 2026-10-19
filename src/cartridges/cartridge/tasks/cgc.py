"""Context gap closed (CGC): the single-value cartridge metric.

On held-out eval tasks, the teacher's answers are ground truth. Each system
scores the same answer tokens teacher-forced (no decoding):

    base      the base model, no journal
    cartridge the base model + cartridge, no journal (prompt = label.answer_messages)
    teacher   the base model with the journal in context

    CGC = (NLL_base - NLL_cart) / (NLL_base - NLL_teacher)

with NLL summed over tokens and pooled over tasks (token-weighted). 1.0 means
the cartridge explains the teacher's answers as well as the full journal does;
0 means no better than the base model. The 95% CI comes from a bootstrap over
tasks. The perplexities exp(NLL / tokens) of all three systems are reported too.

The report is segmented FIRST by source (chat, synthetic, psych_test, human),
THEN by kind within each source. A cell with fewer than MIN_CELL tasks is
flagged small; a cell where the teacher does not beat base has no CGC.

Pure stages:
    to_scored(row, base, cart, teacher)   per-token logprobs -> ScoredTask
    assemble(tasks, base, cart, teacher)  per-task logprobs -> ScoredTasks
    crosscheck(rows, teacher)             HF teacher vs the vLLM label top-k (sanity)
Orchestration (IO injected):
    scorer_session(model_id)              TeacherScorer, chunked, spawn+cancel, stop by id
    final_cgc(cartridge, model_id, labels, tasks, ...)   trainer -> scorer -> assemble
    cgc_stats(tasks, seed)                CGC, CI, perplexities for one cell
    segment(tasks, seed)                  {"all": .., "<source>": {"all": .., "kind:<k>": ..}}
    report_markdown(segments, meta)       the report artifact's text
IO:
    log_report(run, tasks, segments, meta, out_path)
"""

from __future__ import annotations

import contextlib
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

MIN_CELL = 10  # tasks; smaller cells are flagged
N_BOOT = 2000
SOURCES = ("chat", "synthetic", "psych_test", "human")
_SOURCE_OF = {"chat": "chat", "synthetic": "synthetic", "psych_test": "psych_test", "human": "human",
              "human_template": "human"}


TOKEN_TYPES = ("citation", "number", "name", "other")


class ScoredTask(BaseModel):
    """One eval task's teacher answer scored by the three systems (NLLs summed
    over the answer's tokens), plus the diagnostics: top-1 agreement counts and
    the NLL sums per token type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    origin: str
    kind: str
    n_tokens: int
    nll_base: float
    nll_cart: float
    nll_teacher: float
    n_top1: int = 0  # tokens with argmax data
    top1_base: int = 0  # tokens where the system's argmax is the teacher's token
    top1_cart: int = 0
    by_type: dict[str, tuple[int, float, float, float]] = {}  # type -> (tokens, nll_base, nll_cart, nll_teacher)

    @property
    def source(self) -> str:
        return _SOURCE_OF.get(self.origin, self.origin)


# ---------------------------------------------------------------- pure


def to_scored(task_id: str, origin: str, kind: str, base: Sequence[float], cart: Sequence[float],
              teacher: Sequence[float], completion_ids: Sequence[int] | None = None,
              base_argmax: Sequence[int] | None = None, cart_argmax: Sequence[int] | None = None,
              types: Sequence[str] | None = None) -> ScoredTask:
    """From per-token logprobs of the same answer tokens. If a system scored a
    shorter prefix (a length cap), all three use the common prefix. With the
    teacher's completion_ids and both argmaxes, counts top-1 agreement; with
    per-token types (token_types), sums the NLLs per type."""
    n = min(len(base), len(cart), len(teacher))
    extra: dict = {}
    if completion_ids is not None and base_argmax is not None and cart_argmax is not None:
        m = min(n, len(base_argmax), len(cart_argmax))
        extra |= {"n_top1": m, "top1_base": sum(base_argmax[i] == completion_ids[i] for i in range(m)),
                  "top1_cart": sum(cart_argmax[i] == completion_ids[i] for i in range(m))}
    if types is not None:
        by_type: dict[str, list] = {}
        for i in range(min(n, len(types))):
            acc = by_type.setdefault(types[i], [0, 0.0, 0.0, 0.0])
            acc[0] += 1
            acc[1] -= base[i]
            acc[2] -= cart[i]
            acc[3] -= teacher[i]
        extra["by_type"] = {k: tuple(v) for k, v in by_type.items()}
    return ScoredTask(task_id=task_id, origin=origin, kind=kind, n_tokens=n, nll_base=-sum(base[:n]),
                      nll_cart=-sum(cart[:n]), nll_teacher=-sum(teacher[:n]), **extra)


def token_types(pieces: Sequence[str]) -> list[str]:
    """Each answer token's type, from its decoded text in context: "citation"
    (inside a [journal:p.N] span), "number" (has a digit: counts, dates, ratings),
    "name" (a capitalized word not at a sentence start: people, places), "other"."""
    from cartridges.cartridge.evals import _CITATION

    text = "".join(pieces)
    spans = [m.span() for m in _CITATION.finditer(text)]
    out, pos = [], 0
    for piece in pieces:
        start, end = pos, pos + len(piece)
        pos = end
        word = piece.strip()
        prev = text[:start].rstrip(" ")
        if any(a < end and start < b for a, b in spans):
            out.append("citation")
        elif any(ch.isdigit() for ch in piece):
            out.append("number")
        elif word[:1].isupper() and word.isalpha() and prev and prev[-1] not in ".!?:\n*#-\"(":
            out.append("name")
        elif out and out[-1] == "name" and piece[:1].isalpha():  # the rest of a split word ("All" + "ie")
            out.append("name")
        else:
            out.append("other")
    for i in range(1, len(out) - 1):  # a date separator between digits ("11/20", "2021-11") is part of the number
        if out[i] == "other" and pieces[i].strip() in ("/", "-", ".", ":") and out[i - 1] == out[i + 1] == "number":
            out[i] = "number"
    return out


def chosen_logprobs(completion_ids: Sequence[int], topk_ids: Sequence[Sequence[int]],
                    topk_logprobs: Sequence[Sequence[float]]) -> list[float | None]:
    """The logprob of each completion token from a top-k row (None where it is
    not in the top-k). For the teacher's own greedy labels the token is the
    argmax, so this cross-checks the HF teacher score."""
    out = []
    for tok, ids, lps in zip(completion_ids, topk_ids, topk_logprobs, strict=True):
        out.append(lps[ids.index(tok)] if tok in ids else None)
    return out


def _cgc(tasks: Sequence[ScoredTask]) -> float | None:
    base = sum(t.nll_base for t in tasks)
    gap = base - sum(t.nll_teacher for t in tasks)
    return (base - sum(t.nll_cart for t in tasks)) / gap if gap > 0 else None


def _ppl(tasks: Sequence[ScoredTask], field: str) -> float | None:
    n = sum(t.n_tokens for t in tasks)
    return math.exp(sum(getattr(t, field) for t in tasks) / n) if n else None


def _diagnostics(tasks: Sequence[ScoredTask]) -> dict:
    """top1_cart / top1_base: share of answer tokens where the system's argmax is
    the teacher's token. Per token type t: tokens_t, the mean NLL per token of
    each system (nll_base_t, nll_cart_t, nll_teacher_t) and cgc_t (the gap
    closed on that type's tokens)."""
    n1 = sum(t.n_top1 for t in tasks)
    out = {"top1_cart": sum(t.top1_cart for t in tasks) / n1 if n1 else None,
           "top1_base": sum(t.top1_base for t in tasks) / n1 if n1 else None}
    for kind in TOKEN_TYPES:
        cells = [t.by_type[kind] for t in tasks if kind in t.by_type]
        n = sum(c[0] for c in cells)
        if not n:
            continue
        b, c, te = (sum(x[i] for x in cells) for i in (1, 2, 3))
        out |= {f"tokens_{kind}": n, f"nll_base_{kind}": b / n, f"nll_cart_{kind}": c / n,
                f"nll_teacher_{kind}": te / n, f"cgc_{kind}": (b - c) / (b - te) if b - te > 0 else None}
    return out


def cgc_stats(tasks: Sequence[ScoredTask], seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """One cell: n tasks and tokens, CGC with a 95% bootstrap CI over tasks, the
    three perplexities, flags, and the diagnostics (top-1 agreement, per token
    type)."""
    out = {"n": len(tasks), "tokens": sum(t.n_tokens for t in tasks), "cgc": _cgc(tasks),
           "ppl_base": _ppl(tasks, "nll_base"), "ppl_cart": _ppl(tasks, "nll_cart"),
           "ppl_teacher": _ppl(tasks, "nll_teacher"), "small": len(tasks) < MIN_CELL, "cgc_ci": None,
           **_diagnostics(tasks)}
    if len(tasks) >= 2 and out["cgc"] is not None:
        rng = random.Random(seed)
        boots = sorted(v for _ in range(n_boot)
                       if (v := _cgc([tasks[rng.randrange(len(tasks))] for _ in tasks])) is not None)
        if boots:
            out["cgc_ci"] = (boots[int(0.025 * len(boots))], boots[min(len(boots) - 1, int(0.975 * len(boots)))])
    return out


def segment(tasks: Sequence[ScoredTask], seed: int = 0) -> dict:
    """{"all": stats, "<source>": {"all": stats, "kind:<k>": stats, ...}, ...},
    sources in SOURCES order (others after), kinds sorted."""
    out: dict = {"all": cgc_stats(tasks, seed)}
    present = {t.source for t in tasks}
    for source in [s for s in SOURCES if s in present] + sorted(present - set(SOURCES)):
        cell = [t for t in tasks if t.source == source]
        out[source] = {"all": cgc_stats(cell, seed)}
        for kind in sorted({t.kind for t in cell}):
            out[source][f"kind:{kind}"] = cgc_stats([t for t in cell if t.kind == kind], seed)
    return out


def _fmt(v, digits: int = 2) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _row(label: str, s: dict) -> str:
    ci = s["cgc_ci"]
    cgc = _fmt(s["cgc"]) + (f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "")
    flag = " ⚠ small" if s["small"] else ""
    return (f"| {label}{flag} | {s['n']} | {s['tokens']} | {cgc} | {_fmt(s['ppl_base'])} | {_fmt(s['ppl_cart'])} "
            f"| {_fmt(s['ppl_teacher'])} |")


def report_markdown(segments: dict, meta: dict) -> str:
    """Headline CGC, then one table per source (its kinds as rows)."""
    head = segments["all"]
    lines = [
        "# Cartridge eval: context gap closed",
        "",
        "CGC = (NLL_base − NLL_cart) / (NLL_base − NLL_teacher), token-weighted over the teacher's answers to "
        "held-out eval tasks; 95% bootstrap CI over tasks. 1.0 = as good as the full journal in context, "
        "0 = no better than the base model.",
        "",
        *[f"- {k}: `{v}`" for k, v in meta.items()],
        "",
        f"**Headline CGC: {_fmt(head['cgc'])}**"
        + (f" (95% CI {_fmt(head['cgc_ci'][0])}–{_fmt(head['cgc_ci'][1])})" if head["cgc_ci"] else "")
        + f" over {head['n']} tasks, {head['tokens']} tokens.",
        "",
    ]
    header = ["| cell | n | tokens | CGC [95% CI] | ppl base | ppl cartridge | ppl teacher |", "|---|---|---|---|---|---|---|"]
    lines += ["## By source", "", *header, _row("all", head)]
    lines += [_row(source, cells["all"]) for source, cells in segments.items() if source != "all"]
    for source, cells in segments.items():
        if source == "all":
            continue
        lines += ["", f"## {source} by kind", "", *header]
        lines += [_row(name.removeprefix("kind:"), s) for name, s in cells.items() if name != "all"]
    lines += ["", f"Cells under {MIN_CELL} tasks are flagged ⚠ small; read their CGC as indicative only. "
              "A – CGC means the teacher did not beat base in that cell."]
    lines += _diagnostics_markdown(segments)
    return "\n".join(lines) + "\n"


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def _diagnostics_markdown(segments: dict) -> list[str]:
    """Top-1 agreement by source and kind, and the per-token-type NLL split
    (overall and by source). Empty if the scores carried neither."""
    cells = [("all", segments["all"])]
    for source, by in segments.items():
        if source != "all":
            cells += [(source, by["all"])] + [(f"{source} / {k.removeprefix('kind:')}", v) for k, v in by.items()
                                               if k != "all"]
    lines: list[str] = []
    if segments["all"].get("top1_cart") is not None:
        lines += ["", "## Diagnostics: top-1 agreement", "",
                  "Share of the teacher's answer tokens that are each system's argmax (teacher-forced).", "",
                  "| cell | n | top-1 cartridge | top-1 base |", "|---|---|---|---|"]
        lines += [f"| {name}{' ⚠ small' if st['small'] else ''} | {st['n']} | {_pct(st.get('top1_cart'))} "
                  f"| {_pct(st.get('top1_base'))} |" for name, st in cells]
    if any(f"tokens_{t}" in segments["all"] for t in TOKEN_TYPES):
        lines += ["", "## Diagnostics: NLL per token by token type", "",
                  "citation = inside a [journal:p.N] span; number = has a digit (dates, counts, ratings); "
                  "name = capitalized word not at a sentence start; other = the rest. Mean NLL per token; "
                  "gap closed = (base − cartridge) / (base − teacher) on that type's tokens.", "",
                  "| cell | type | tokens | NLL base | NLL cartridge | NLL teacher | gap closed |",
                  "|---|---|---|---|---|---|---|"]
        for name, st in [c for c in cells if " / " not in c[0]]:
            for t in TOKEN_TYPES:
                if f"tokens_{t}" in st:
                    lines.append(f"| {name} | {t} | {st[f'tokens_{t}']} | {_fmt(st[f'nll_base_{t}'], 3)} "
                                 f"| {_fmt(st[f'nll_cart_{t}'], 3)} | {_fmt(st[f'nll_teacher_{t}'], 3)} "
                                 f"| {_fmt(st[f'cgc_{t}'])} |")
    return lines


def flat_summary(segments: dict) -> dict:
    """Run-summary keys: eval/cgc, eval/cgc_ci_low/high, eval/ppl_*, the
    diagnostics (eval/top1_cart, eval/top1_base, eval/cgc_<type>, ...), and the
    same under eval/<source>/ and eval/<source>/<kind>/."""
    def keys(prefix: str, s: dict) -> dict:
        out = {f"{prefix}{k}": v for k, v in s.items()
               if k not in ("cgc_ci", "small") and isinstance(v, (int, float)) and not isinstance(v, bool)}
        if s["cgc_ci"]:
            out |= {f"{prefix}cgc_ci_low": s["cgc_ci"][0], f"{prefix}cgc_ci_high": s["cgc_ci"][1]}
        return out

    out = keys("eval/", segments["all"])
    for source, cells in segments.items():
        if source != "all":
            for name, s in cells.items():
                out |= keys(f"eval/{source}/" if name == "all" else f"eval/{source}/{name.removeprefix('kind:')}/", s)
    return out


def crosscheck(rows: Sequence[dict], teacher: Sequence[Sequence[float]]) -> dict:
    """HF teacher logprobs vs the vLLM label's top-k (rows carry completion_ids,
    topk_ids, topk_logprobs): the share of tokens found in the label's top-k and
    the mean |difference| on those. A sanity check, not part of CGC."""
    found, diffs = 0, []
    total = 0
    for row, lps in zip(rows, teacher, strict=True):
        label = chosen_logprobs(row["completion_ids"][: len(lps)], row["topk_ids"][: len(lps)],
                                row["topk_logprobs"][: len(lps)])
        total += len(label)
        for v, hf in zip(label, lps):
            if v is not None:
                found += 1
                diffs.append(abs(v - hf))
    return {"tokens": total, "in_label_topk": found / total if total else None,
            "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None}


# ---------------------------------------------------------------- orchestration


@contextlib.contextmanager
def scorer_session(model_id: str, chunk: int = 40, get_timeout_s: float = 25 * 60, make=None, stop=None):
    """score_ref(rows, with_context) over the deployed TeacherScorer.score_rows
    (HF, label ids, chosen-token logprobs; with_context=False is base). Rows go
    in chunks of `chunk` per call (each call has a 30 min limit); each call is
    spawned and, on any exception or Ctrl-C, cancelled so no pending input
    restarts a container. On exit only this session's container is stopped
    (teacher.stop(container_ids=[...]), never the whole app). make/stop are
    teacher.teacher_cls/teacher.stop, injectable for tests."""
    if make is None or stop is None:
        from cartridges.cartridge import teacher

        make, stop = make or teacher.teacher_cls, stop or teacher.stop
    scorer = make("TeacherScorer", model_id)
    container_id = scorer.info.remote()["container_id"]

    def score_ref(rows: Sequence[dict], with_context: bool) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(rows), chunk):
            ids = [{"prompt_ids": r["prompt_ids"], "completion_ids": r["completion_ids"]} for r in rows[i : i + chunk]]
            call = scorer.score_rows.spawn(ids, with_context=with_context)
            try:
                out += call.get(timeout=get_timeout_s)
            except BaseException:
                call.cancel()
                raise
        return out

    try:
        yield score_ref
    finally:
        stop(container_ids=[container_id])


def assemble(tasks: Sequence, base: dict[str, dict], cart: dict[str, dict], teacher: dict[str, Sequence[float]],
             targets: dict[str, dict] | None = None, types: dict[str, Sequence[str]] | None = None) -> list[ScoredTask]:
    """ScoredTasks keyed by task_id. base and cart are trainer's score_tasks rows
    ({"logprobs", optional "argmax"}; one HF path with and without the
    cartridge, so the numerator shares numerics; by_task), teacher the
    TeacherScorer.score_rows(with_context=True) logprobs (keyed), all on the same
    Target Store ids. targets (target_rows, by task) give completion_ids for
    top-1 agreement; types (token_types per task) the per-type split. A task
    missing any of the three scores is skipped."""
    out = []
    for task in tasks:
        tid = str(task.task_id)
        if tid in base and tid in cart and tid in teacher:
            ids = (targets or {}).get(tid, {}).get("completion_ids")
            out.append(to_scored(tid, task.origin, task.kind, base[tid]["logprobs"], cart[tid]["logprobs"],
                                 teacher[tid], completion_ids=ids, base_argmax=base[tid].get("argmax"),
                                 cart_argmax=cart[tid].get("argmax"), types=(types or {}).get(tid)))
    return out


def keyed(rows: Sequence[dict], scores: Sequence[Sequence[float]]) -> dict[str, Sequence[float]]:
    """TeacherScorer.score_rows output (aligned with rows) as {task_id: logprobs}."""
    return {str(r["task_id"]): lps for r, lps in zip(rows, scores, strict=True)}


def by_task(rows: Sequence[dict]) -> dict[str, dict]:
    """Rows with a "task_id" (trainer's score_tasks rows, target_rows) as {task_id: row}."""
    return {str(r["task_id"]): r for r in rows}


def target_rows(labels: Sequence, read_labeled) -> list[dict]:
    """{"task_id", "prompt_ids", "completion_ids", "topk_ids", "topk_logprobs"} per
    validated label with targets, from read_labeled(labels) (targets.read_labeled:
    the label's exact Target Store ids)."""
    valid = [lab for lab in labels if lab.status == "validated" and lab.targets_uri is not None]
    return [{"task_id": str(lab.task_id), "prompt_ids": ex.prompt_ids, "completion_ids": ex.completion_ids,
             "topk_ids": ex.topk_ids, "topk_logprobs": ex.topk_logprobs}
            for lab, ex in zip(valid, read_labeled(valid), strict=True)]


def final_cgc(cartridge: str, model_id: str, labels: Sequence, tasks: Sequence, *, read_labeled, score_cart,
              session, pieces=None, seed: int = 0) -> tuple[list[ScoredTask], dict, dict]:
    """The final CGC, steps strictly in order (one GPU pool at a time):
    1. score_cart(cartridge, label dicts, model_id) -> {"cartridge", "base"} rows
       (trainer's train_distill.score_tasks: base and cartridge on one HF path);
    2. the teacher term only (with_context=True) in a scorer session (the base
       term is not scored there);
    3. assemble (with top-1 agreement when the rows carry "argmax", and the
       token-type split when pieces(ids) -> decoded token texts is given),
       segment. Returns (scored tasks, segments, crosscheck of the HF teacher
       against the vLLM labels)."""
    rows = target_rows(labels, read_labeled)
    if not rows:  # checked before any GPU call
        raise ValueError("no validated eval labels with targets: nothing to score")
    trainer_out = score_cart(cartridge, [lab.model_dump(mode="json") for lab in labels], model_id)
    with session(model_id) as score_ref:
        teacher = score_ref(rows, True)
    types = {r["task_id"]: token_types(pieces(r["completion_ids"])) for r in rows} if pieces else None
    scored = assemble(tasks, by_task(trainer_out["base"]), by_task(trainer_out["cartridge"]), keyed(rows, teacher),
                      by_task(rows), types)
    if not scored or not sum(t.n_tokens for t in scored):
        raise ValueError(f"no task scored by all three systems ({len(rows)} rows; trainer base "
                         f"{len(trainer_out['base'])}, cartridge {len(trainer_out['cartridge'])}, teacher {len(teacher)})")
    return scored, segment(scored, seed), crosscheck(rows, teacher)


# ---------------------------------------------------------------- IO

_CELL_COLUMNS = ["n", "tokens", "cgc", "cgc_ci_low", "cgc_ci_high", "ppl_base", "ppl_cart", "ppl_teacher",
                 "top1_cart", "top1_base", *(f"cgc_{t}" for t in TOKEN_TYPES), "small"]


def _cell(s: dict) -> list:
    ci = s["cgc_ci"] or (None, None)
    return [s["n"], s["tokens"], s["cgc"], ci[0], ci[1], s["ppl_base"], s["ppl_cart"], s["ppl_teacher"],
            s.get("top1_cart"), s.get("top1_base"), *(s.get(f"cgc_{t}") for t in TOKEN_TYPES), s["small"]]


def segment_tables(segments: dict) -> dict:
    """W&B tables eval/by_source (source, n, cgc, ...) and eval/by_kind
    (source, kind, n, cgc, ...), sorted by source then kind; "all" rows first."""
    import wandb

    by_source = [["all", *_cell(segments["all"])]]
    by_source += [[src, *_cell(cells["all"])] for src, cells in segments.items() if src != "all"]
    by_kind = [[src, name.removeprefix("kind:"), *_cell(s)] for src, cells in segments.items() if src != "all"
               for name, s in cells.items() if name != "all"]
    return {"eval/by_source": wandb.Table(columns=["source", *_CELL_COLUMNS], data=by_source),
            "eval/by_kind": wandb.Table(columns=["source", "kind", *_CELL_COLUMNS], data=by_kind)}



def log_report(run, tasks: Sequence[ScoredTask], segments: dict, meta: dict, out_path: Path) -> Path:
    """Write the markdown report, log it as an artifact `cgc-report`, the W&B
    tables eval/cgc_tasks (per task), eval/by_source and eval/by_kind, and
    flat_summary() into the run summary (run is a tracking.TrackedRun,
    job_type eval, config.system = "cartridge"). Refuses an empty eval."""
    import wandb

    if not tasks or not segments["all"]["tokens"]:
        raise ValueError("empty CGC eval (0 tasks or 0 tokens): not logging it")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_markdown(segments, meta))
    columns = ["task_id", "source", "origin", "kind", "n_tokens", "nll_base", "nll_cart", "nll_teacher"]
    table = wandb.Table(columns=columns, data=[[t.task_id, t.source, t.origin, t.kind, t.n_tokens, t.nll_base,
                                                 t.nll_cart, t.nll_teacher] for t in tasks])
    summary = flat_summary(segments)
    run.log(summary | {"eval/cgc_tasks": table, **segment_tables(segments)})
    run.run.summary.update(summary | {"eval/cgc_segments": json.dumps(segments)})
    run.log_artifact(out_path, type="eval-report", name="cgc-report")
    return out_path
