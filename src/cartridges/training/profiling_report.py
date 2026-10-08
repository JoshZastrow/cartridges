"""Summarize profiling result JSONs (from `profiling.py ... poll`) locally.

Every row's output is compared to the *current path's* output for the same
page: `--baseline-file`'s rows for `--baseline-step` (default: the Unsloth
load + unmerged adapter + HF generate run, step "baseline").

uv run python -m cartridges.training.profiling_report profiles/unsloth_v1.json profiles/plain_v2.json
"""

from __future__ import annotations

import json

import fire
import jiwer


def report(*files: str, baseline_file: str = "profiles/unsloth_v1.json", baseline_step: str = "baseline"):
    with open(baseline_file) as f:
        baseline = {r["page"]: r["hypothesis"] for r in json.load(f)["rows"] if r["step"] == baseline_step}
    print(
        f"{'run':14} {'step':14} {'page':8} {'out':>5} {'prefill_s':>9} {'decode_t/s':>10} {'wall_s':>7}"
        f" {'util%':>5} {'mem_gb':>6} {'cer_ref':>7} {'cer_vs_cur':>10} match"
    )
    for path in files:
        with open(path) as f:
            result = json.load(f)
        run = result["config"]["run_name"]
        for r in result["rows"]:
            ref = baseline.get(r["page"])
            cer_cur = jiwer.cer(ref, r["hypothesis"]) if ref is not None else None
            print(
                f"{run:14} {r['step']:14} {r['page']:8} {r['output_tokens']:5d} {r['prefill_s']:9.2f}"
                f" {r['decode_tok_s'] or 0:10.1f} {r['page_wall_s']:7.2f} {r['gpu_util_pct'] or 0:5.0f}"
                f" {r['peak_mem_gb']:6.1f} {r['cer_vs_reference']:7.4f}"
                f" {'' if cer_cur is None else f'{cer_cur:10.4f}'} {'' if ref is None else ref == r['hypothesis']}"
            )


if __name__ == "__main__":
    fire.Fire(report)
