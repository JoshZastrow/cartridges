"""Render a single-page profile (stages.json + gpu.csv + ops.txt) as one
self-contained HTML file (inline SVG, no external assets) -- report.html in
docs/plans/profiling.md. Pure Python so it is testable and re-runnable locally:

uv run python -m cartridges.training.profile_page_report profiles/<run_name>
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from cartridges.training.perf_metrics import TOP_LEVEL_STAGES

STAGE_COLORS = {
    "save_image": "#8d6e63",
    "preprocess": "#f9a825",
    "generate_setup": "#bdbdbd",
    "prefill": "#1e88e5",
    "decode_step": "#e53935",
    "decode_text": "#43a047",
    "other": "#eeeeee",
}


def _fmt(value, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{unit}"
    return f"{value}{unit}"


def _stage_bar(stage_ms: dict[str, float], wall_ms: float) -> str:
    total = max(wall_ms, sum(stage_ms.get(s, 0.0) for s in TOP_LEVEL_STAGES), 1e-9)
    parts = [(s, stage_ms.get(s, 0.0)) for s in TOP_LEVEL_STAGES]
    parts.append(("other", max(0.0, wall_ms - sum(ms for _, ms in parts))))
    x, rects, legend = 0.0, [], []
    for name, ms in parts:
        width = 1000 * ms / total
        if width > 0:
            rects.append(
                f'<rect x="{x:.1f}" y="0" width="{width:.1f}" height="36" fill="{STAGE_COLORS[name]}">'
                f"<title>{name}: {ms:,.0f} ms</title></rect>"
            )
        x += width
        legend.append(
            f'<span class="key"><i style="background:{STAGE_COLORS[name]}"></i>{name} '
            f"{ms:,.0f} ms ({100 * ms / total:.0f}%)</span>"
        )
    return (
        f'<svg viewBox="0 0 1000 36" preserveAspectRatio="none" class="bar">{"".join(rects)}</svg>'
        f'<div class="legend">{"".join(legend)}</div>'
    )


def _line_chart(xs: list[float], series: list[tuple[str, list[float], str]], x_label: str, y_label: str) -> str:
    """Minimal SVG line chart; each series is (name, ys, color)."""
    if not xs or not any(ys for _, ys, _ in series):
        return "<p class='muted'>no data</p>"
    w, h, pad = 1000, 220, 40
    x0, x1 = min(xs), max(xs)
    y1 = max(max(ys) for _, ys, _ in series if ys) or 1.0
    x_span = (x1 - x0) or 1.0

    def px(x):
        return pad + (w - 2 * pad) * (x - x0) / x_span

    def py(y):
        return h - pad - (h - 2 * pad) * y / y1

    lines = [
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" '
        f'points="{" ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))}"><title>{name}</title></polyline>'
        for name, ys, color in series
    ]
    axes = (
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" class="axis"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" class="axis"/>'
        f'<text x="{pad}" y="{pad - 8}" class="tick">{y1:,.1f} {html.escape(y_label)}</text>'
        f'<text x="{pad}" y="{h - 10}" class="tick">{x0:,.2f}</text>'
        f'<text x="{w - pad}" y="{h - 10}" class="tick" text-anchor="end">{x1:,.2f} {html.escape(x_label)}</text>'
    )
    legend = "".join(
        f'<span class="key"><i style="background:{color}"></i>{html.escape(name)}</span>' for name, _, color in series
    )
    return f'<svg viewBox="0 0 {w} {h}" class="chart">{axes}{"".join(lines)}</svg><div class="legend">{legend}</div>'


def _verdict(stages: dict) -> str:
    steps = stages.get("decode_step_ms", {})
    trace = stages.get("trace_stats", {})
    ms = steps.get("p50")
    if not ms:
        return "No decode steps recorded (forward mode or a 1-token output)."
    busy = trace.get("gpu_busy_frac")
    gpu_ms = ms * busy if busy is not None else None
    proc = stages.get("logits_processor_ms_per_step", {})
    proc_total = sum(proc.values())
    syncs = sum(trace.get("sync_ops_per_step", {}).get(k, 0) for k in ("aten::_local_scalar_dense",))
    return (
        f"Each decode step takes {ms:,.1f} ms of wall time (median; profiler on). "
        f"The GPU is busy for {_fmt(gpu_ms)} ms of it ({_fmt(100 * busy if busy is not None else None, 0)}%). "
        f"The rest is host-side: {trace.get('launches_per_step', 0):,.0f} kernel launches, "
        f"{syncs:,.0f} blocking scalar reads (.item()/.tolist()), "
        f"{trace.get('stream_syncs_per_step', 0):,.0f} stream syncs, and "
        f"{proc_total:,.2f} ms in logits processors per step."
    )


def render_report(stages: dict, gpu_rows: list[tuple[float, float, float]], ops_text: str) -> str:
    cfg = stages.get("config", {})
    trace = stages.get("trace_stats", {})
    step_ms = stages.get("decode_step_ms_all", [])
    summary_rows = [
        ("Mode", cfg.get("mode")),
        ("Page", f"{cfg.get('split')}/{cfg.get('index')}"),
        ("Model", f"{cfg.get('model_source')} {cfg.get('hf_revision') or ''} merged={cfg.get('merge_adapter')}"),
        ("Optimizations", ", ".join(cfg.get("optimizations") or []) or "none (current path)"),
        ("Wall time of the recorded infer()", _fmt(stages.get("wall_ms"), 0, " ms")),
        ("Time to first token", _fmt(stages.get("ttft_ms"), 0, " ms")),
        ("Output tokens", stages.get("output_tokens")),
        ("Decode step, median / p90", f"{_fmt(stages.get('decode_step_ms', {}).get('p50'))} / "
                                      f"{_fmt(stages.get('decode_step_ms', {}).get('p90'))} ms"),
        ("Decode tok/s (profiler on)", _fmt(stages.get("decode_tok_s_profiled"))),
        ("Decode tok/s (no profiler, same page)", _fmt(stages.get("decode_tok_s_unprofiled"))),
        ("GPU busy during decode (trace)", _fmt(100 * trace["gpu_busy_frac"], 0, "%") if trace.get("gpu_busy_frac") is not None else "n/a"),
        ("Kernel launches per step", _fmt(trace.get("launches_per_step"), 0)),
        ("Stream syncs per step", _fmt(trace.get("stream_syncs_per_step"), 1)),
        ("Blocking scalar reads per step", _fmt(trace.get("sync_ops_per_step", {}).get("aten::_local_scalar_dense"), 1)),
        ("DtoH copies per step", _fmt(trace.get("dtoh_copies_per_step"), 1)),
        ("GPU gaps > 20 µs per step", _fmt(trace.get("gpu_gaps_per_step"), 0)),
        ("Stage labels cover", _fmt(100 * stages["stage_coverage"], 0, "% of wall time") if stages.get("stage_coverage") else "n/a"),
        ("Output identical to unprofiled run", stages.get("output_matches_unprofiled")),
        ("Peak GPU memory", _fmt(stages.get("peak_mem_gb"), 2, " GB")),
    ]
    proc_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{ms:,.3f}</td></tr>"
        for name, ms in sorted(stages.get("logits_processor_ms_per_step", {}).items(), key=lambda kv: -kv[1])
    ) or "<tr><td colspan=2>none</td></tr>"
    nested = {k: v for k, v in stages.get("stage_ms", {}).items() if k not in TOP_LEVEL_STAGES}
    nested_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:,.1f}</td></tr>" for k, v in sorted(nested.items(), key=lambda kv: -kv[1])
    )
    t0 = gpu_rows[0][0] if gpu_rows else 0.0
    gpu_chart = _line_chart(
        [t - t0 for t, _, _ in gpu_rows],
        [("GPU util %", [u for _, u, _ in gpu_rows], "#e53935"),
         ("memory GB x10", [m / 100 for _, _, m in gpu_rows], "#1e88e5")],
        "s", "",
    )
    step_chart = _line_chart(
        list(range(1, len(step_ms) + 1)), [("decode step wall ms", step_ms, "#e53935")], "step", "ms"
    )
    files = ["trace.json.gz (ui.perfetto.dev)", "stacks_cpu.folded / stacks_gpu.folded (speedscope.app)",
             "stages.json", "ops.txt", "gpu.csv", "output.txt"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Profile {html.escape(str(cfg.get('run_name', '')))}</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#d2d2d7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#f5f5f7; --muted:#a1a1a6; --line:#3a3a3c; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,system-ui,sans-serif; max-width:1100px; margin:0 auto; padding:16px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin-top:28px; }}
table {{ border-collapse:collapse; }} td,th {{ border-bottom:1px solid var(--line); padding:4px 12px 4px 0; text-align:left; vertical-align:top; }}
.bar {{ width:100%; height:36px; }} .chart {{ width:100%; height:auto; }}
.axis {{ stroke:var(--muted); }} .tick {{ fill:var(--muted); font-size:12px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:12px; margin:6px 0; color:var(--muted); }}
.key i {{ display:inline-block; width:10px; height:10px; margin-right:4px; border-radius:2px; }}
.verdict {{ font-size:16px; padding:12px; border-left:4px solid #e53935; }}
pre {{ overflow-x:auto; font-size:11px; }} .muted {{ color:var(--muted); }}
</style></head><body>
<h1>Single-page profile: {html.escape(str(cfg.get('run_name', '')))}</h1>
<p class="verdict">{html.escape(_verdict(stages))}</p>
<h2>Summary</h2>
<table>{"".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in summary_rows)}</table>
<h2>Stages (one model.infer() call)</h2>
{_stage_bar(stages.get("stage_ms", {}), stages.get("wall_ms") or 0.0)}
<h2>Decode step wall time</h2>
{step_chart}
<h2>Logits processors, CPU ms per decode step</h2>
<table><tr><th>processor</th><th>ms/step</th></tr>{proc_rows}</table>
<h2>Nested labels (total ms)</h2>
<table><tr><th>label</th><th>ms</th></tr>{nested_rows}</table>
<h2>GPU sensor (nvidia-smi, 50 ms)</h2>
{gpu_chart}
<h2>Top operations</h2>
<pre>{html.escape(ops_text)}</pre>
<h2>Files in this folder</h2>
<ul>{"".join(f"<li>{html.escape(f)}</li>" for f in files)}</ul>
</body></html>
"""


def read_gpu_csv(path: Path) -> list[tuple[float, float, float]]:
    with open(path) as f:
        return [(float(r["t_s"]), float(r["util_pct"]), float(r["mem_mib"])) for r in csv.DictReader(f)]


def main(run_dir: str) -> None:
    root = Path(run_dir)
    stages = json.loads((root / "stages.json").read_text())
    gpu_rows = read_gpu_csv(root / "gpu.csv") if (root / "gpu.csv").exists() else []
    ops = (root / "ops.txt").read_text() if (root / "ops.txt").exists() else ""
    (root / "report.html").write_text(render_report(stages, gpu_rows, ops))
    print(f"wrote {root / 'report.html'}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
