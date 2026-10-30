"""`voice report`: the section 8 report of one experiment group, from W&B only.

    uv run voice report --group exp-YYYYMMDD-<slug> [--stage eval-full] [--out voice/outputs/report/<group>]

Reads every eval run of the group (job_type `eval`, run name `<backend>/<system>/<stage>`,
tag `baseline` or `candidate`) and its `report.json` (saved to the run by
`voice eval`), then writes into `--out` and logs as a `voice-report` artifact
(from a run `report/<stage>` in the same group, tag `report`):

    table.md, table.csv          every row, one column per run: value [ci_low, ci_high] (n)
    frontier-interaction.svg     fidelity/judge against interaction/fdb15, one point per run
    frontier-latency.svg         fidelity/judge against responsiveness/model_latency_s
    side-by-side.md              the eval runs' side-by-side transcripts, joined by prompt id

Baselines are drawn hollow. A run whose row is missing is left off that plot.
The pure parts (`table`, `frontier_svg`, `side_by_side_md`) take plain dicts.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ENTITY, PROJECT = "conle-ai", "voice"
FIDELITY = "fidelity/judge"
FRONTIERS = {"frontier-interaction.svg": "interaction/fdb15", "frontier-latency.svg": "responsiveness/model_latency_s"}


# ---------------------------------------------------------------- pure


def fmt(r: dict | None) -> str:
    if not r or r.get("value") is None:
        return "-"
    lo, hi = r.get("ci_low"), r.get("ci_high")
    ci = f" [{lo:.3g}, {hi:.3g}]" if lo is not None and hi is not None else ""
    return f"{r['value']:.3g}{ci} (n={r['n']})"


def table(runs: dict[str, dict]) -> tuple[list[str], list[list[str]]]:
    """runs: name -> report.json rows. Header and one line per metric (sorted), runs as columns."""
    names = sorted(runs)
    metrics = sorted({m for rows in runs.values() for m in rows})
    return ["metric", *names], [[m, *(fmt(runs[n].get(m)) for n in names)] for m in metrics]


def table_md(header: list[str], lines: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(line) + " |" for line in lines]) + "\n"


def table_csv(runs: dict[str, dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run", "metric", "value", "ci_low", "ci_high", "n"])
    for name in sorted(runs):
        for m, r in sorted(runs[name].items()):
            w.writerow([name, m, r.get("value"), r.get("ci_low"), r.get("ci_high"), r.get("n")])
    return buf.getvalue()


def points(runs: dict[str, dict], baselines: set[str], x: str, y: str) -> list[dict]:
    """Plot points for runs that have both rows."""
    out = []
    for name in sorted(runs):
        rx, ry = runs[name].get(x), runs[name].get(y)
        if rx and ry and rx.get("value") is not None and ry.get("value") is not None:
            out.append({"name": name, "baseline": name in baselines, "x": rx, "y": ry})
    return out


def frontier_svg(pts: list[dict], x_label: str, y_label: str, w: int = 560, h: int = 400, pad: int = 60) -> str:
    """Scatter with CI whiskers in both directions; baselines hollow, candidates filled."""

    def span(key: str) -> tuple[float, float]:
        vals = [v for p in pts for v in (p[key]["value"], p[key].get("ci_low"), p[key].get("ci_high")) if v is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        return (lo - 0.05, hi + 0.05) if hi - lo < 1e-9 else (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    (x0, x1), (y0, y1) = span("x"), span("y")
    sx = lambda v: pad + (v - x0) / (x1 - x0) * (w - 2 * pad)  # noqa: E731
    sy = lambda v: h - pad - (v - y0) / (y1 - y0) * (h - 2 * pad)  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="black"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="black"/>',
           f'<text x="{w / 2}" y="{h - 20}" text-anchor="middle">{html.escape(x_label)}</text>',
           f'<text x="18" y="{h / 2}" text-anchor="middle" transform="rotate(-90 18 {h / 2})">{html.escape(y_label)}</text>']
    for v in (x0, x1):
        out.append(f'<text x="{sx(v):.1f}" y="{h - pad + 14}" text-anchor="middle">{v:.3g}</text>')
    for v in (y0, y1):
        out.append(f'<text x="{pad - 6}" y="{sy(v):.1f}" text-anchor="end">{v:.3g}</text>')
    for p in pts:
        cx, cy = sx(p["x"]["value"]), sy(p["y"]["value"])
        for key, horizontal in (("x", True), ("y", False)):
            lo, hi = p[key].get("ci_low"), p[key].get("ci_high")
            if lo is None or hi is None:
                continue
            if horizontal:
                out.append(f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" stroke="grey"/>')
            else:
                out.append(f'<line x1="{cx:.1f}" y1="{sy(lo):.1f}" x2="{cx:.1f}" y2="{sy(hi):.1f}" stroke="grey"/>')
        fill = "white" if p["baseline"] else "black"
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}" stroke="black"/>')
        out.append(f'<text x="{cx + 8:.1f}" y="{cy - 6:.1f}">{html.escape(p["name"])}</text>')
    return "\n".join(out + ["</svg>"]) + "\n"


def side_by_side_md(tables: dict[str, list[dict]]) -> str:
    """tables: run name -> side-by-side rows ({id, question, transcript, reference, source, latency_s}).
    One section per prompt id with every run's transcript under the shared question, reference and source."""
    by_id: dict[str, dict[str, dict]] = {}
    for name, rows in sorted(tables.items()):
        for r in rows:
            by_id.setdefault(r["id"], {})[name] = r
    out = []
    for pid in sorted(by_id):
        first = next(iter(by_id[pid].values()))
        out += [f"## {pid} ({first.get('family', '')})", "", f"**Question.** {first['question']}", "",
                f"**Source.** {first['source']}", "", f"**Reference.** {first['reference']}", ""]
        for name, r in by_id[pid].items():
            lat = r.get("latency_s")
            out += [f"**{name}** (latency {lat:.2f} s)" if lat is not None else f"**{name}** (no answer)", "",
                    r["transcript"] or "(silence)", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- W&B


def fetch(group: str, stage: str, api=None) -> list[dict]:
    """Eval runs of `group` at `stage`: [{name, id, baseline, rows, side_by_side}]."""
    import tempfile

    import wandb

    api = api or wandb.Api()
    out = []
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group, "jobType": "eval"}):
        if not run.name.endswith(f"/{stage}") or not {"baseline", "candidate"} & set(run.tags):
            continue
        with tempfile.TemporaryDirectory() as d:
            rows = json.loads(Path(run.file("report.json").download(root=d).name).read_text())["rows"]
        sbs = []
        try:
            art = api.artifact(f"{ENTITY}/{PROJECT}/run-{run.id}-side_by_side:latest")
            t = art.get("side_by_side")
            sbs = [dict(zip(t.columns, r)) for r in t.data]
        except Exception as e:  # a run without the table still contributes its rows
            logger.warning("no side-by-side table in %s: %s", run.name, e)
        out.append({"name": run.name, "id": run.id, "baseline": "baseline" in run.tags, "rows": rows,
                    "side_by_side": sbs})
    return out


def write(runs: list[dict], out: Path) -> dict[str, Path]:
    """The report files for fetched runs."""
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["name"]: r["rows"] for r in runs}
    header, lines = table(rows)
    files = {"table.md": table_md(header, lines), "table.csv": table_csv(rows),
             "side-by-side.md": side_by_side_md({r["name"]: r["side_by_side"] for r in runs})}
    baselines = {r["name"] for r in runs if r["baseline"]}
    for fname, x in FRONTIERS.items():
        files[fname] = frontier_svg(points(rows, baselines, x, FIDELITY), x, FIDELITY)
    paths = {}
    for name, text in files.items():
        (out / name).write_text(text)
        paths[name] = out / name
    return paths


def main(group: str, stage: str = "eval-full", out: Path | None = None) -> int:
    import wandb

    from shared import tracking

    out = out or Path("voice/outputs/report") / group
    runs = fetch(group, stage)
    if not runs:
        logger.error("no eval runs at stage %s in group %s", stage, group)
        return 1
    paths = write(runs, out)
    header, lines = table({r["name"]: r["rows"] for r in runs})
    with tracking.start_run(PROJECT, group, {"group": group, "stage": stage, "runs": [r["id"] for r in runs]},
                            name=f"report/{stage}", job_type="report", gpu="CPU", tags=["report"]) as run:
        run.log({"report/table": wandb.Table(columns=header, data=lines)})
        run.log({f"report/{name}": wandb.Html(paths[name].read_text()) for name in FRONTIERS})
        run.log_artifact(out, "voice-report")
        print(run.url)
    print(paths["table.md"].read_text())
    return 0
