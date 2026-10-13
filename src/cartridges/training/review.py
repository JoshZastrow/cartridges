"""Local HTML review page for loop-eval results: prediction vs label, per page.

Reads every `<run>.json` in `data/train/loop_eval/results/` (the prediction
lists `loop_eval` returns), joins page metadata (loop/control set from the
manifest, data-quality flags from `data/train/quality/pages.csv`, whether the
label opens with a markdown heading) and writes one self-contained HTML file.
Page images are linked by relative path, not copied, so the file stays local
and small. Journal text never leaves the machine.

    uv run cartridges eval_review
"""

from __future__ import annotations

import csv
import difflib
import html
import json
import os
import re
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("data/train/loop_eval/results")
DEFAULT_OUT = Path("data/train/loop_eval/review.html")
DEFAULT_MANIFEST = Path("data/train/loop_eval/manifest.jsonl")
DEFAULT_QUALITY_CSV = Path("data/train/quality/pages.csv")
SOURCE_ROOT = Path("data/journals/to_process_manual")


def word_diff_html(reference: str, hypothesis: str) -> str:
    """Word-level diff of `hypothesis` against `reference`: missing words in
    <del>, extra words in <ins>. Whitespace (incl. newlines) is kept."""
    ref = re.findall(r"\S+|\s+", reference)
    hyp = re.findall(r"\S+|\s+", hypothesis)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape("".join(ref[i1:i2])))
            continue
        if op in ("replace", "delete"):
            out.append(f"<del>{html.escape(''.join(ref[i1:i2]))}</del>")
        if op in ("replace", "insert"):
            out.append(f"<ins>{html.escape(''.join(hyp[j1:j2]))}</ins>")
    return "".join(out)


def parse_flags(cell: str) -> list[str]:
    """`pages.csv` flags ("tall_image(5201px, 2.9x median); over_1024_tokens(1356)")
    -> flag names (["over_1024_tokens", "tall_image"])."""
    return sorted({part.split("(")[0].strip() for part in cell.split(";") if part.strip()})


def build_review(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    out: Path = DEFAULT_OUT,
    runs: list[str] | None = None,
    manifest: Path = DEFAULT_MANIFEST,
    quality_csv: Path = DEFAULT_QUALITY_CSV,
) -> dict:
    """Write the review page; returns {"out", "runs", "n_pages"}. `runs` are
    result file stems in display order (default: every file, by name)."""
    from rapidfuzz.distance import Levenshtein  # installed with jiwer

    runs = runs or sorted(p.stem for p in results_dir.glob("*.json"))
    if not runs:
        raise FileNotFoundError(f"no result files in {results_dir}")
    sets = {}
    if manifest.exists():
        sets = {(r["split"], r["index"]): r.get("set", "") for r in map(json.loads, manifest.open())}
    flags = {}
    if quality_csv.exists():
        flags = {(r["journal"], int(r["page"])): parse_flags(r["flags"]) for r in csv.DictReader(quality_csv.open())}

    pages: dict[tuple[str, int], dict] = {}
    for run in runs:
        for p in json.loads((results_dir / f"{run}.json").read_text()):
            key = (p["journal"], p["page"])
            reference = p["reference"]
            page = pages.setdefault(
                key,
                {
                    "journal": p["journal"],
                    "page": p["page"],
                    "split": p["split"],
                    "set": sets.get((p["split"], p["index"]), ""),
                    "flags": flags.get(key, []),
                    "heading_label": reference.lstrip().startswith("#"),
                    "ref_chars": len(reference),
                    "reference": html.escape(reference),
                    "image": os.path.relpath(
                        SOURCE_ROOT / p["journal"] / "_images" / f"page-{p['page']:03d}.png", out.parent
                    ),
                    "runs": {},
                },
            )
            hypothesis = p["hypothesis"]
            page["runs"][run] = (
                {"status": p.get("status", "error"), "cer": None, "ned": None, "looped": False, "diff": ""}
                if hypothesis is None
                else {
                    "status": p.get("status", "ok"),
                    "cer": p.get("cer"),
                    "ned": Levenshtein.normalized_distance(reference, hypothesis),
                    "looped": bool(p.get("looped")),
                    "diff": word_diff_html(reference, hypothesis),
                }
            )

    data = {"runs": runs, "pages": sorted(pages.values(), key=lambda p: (p["journal"], p["page"]))}
    out.parent.mkdir(parents=True, exist_ok=True)
    # "</" can't appear inside the inline <script>; escape it in the JSON.
    out.write_text(_TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return {"out": str(out), "runs": runs, "n_pages": len(pages)}


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Review</title>
<style>
:root {
  --bg: #fafaf9; --panel: #ffffff; --text: #1c1917; --muted: #78716c; --border: #e7e5e4;
  --bad: #b91c1c; --del-bg: #fee2e2; --del: #991b1b; --ins-bg: #dcfce7; --ins: #166534; --sel: #eff6ff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0c0a09; --panel: #1c1917; --text: #e7e5e4; --muted: #a8a29e; --border: #292524;
    --bad: #f87171; --del-bg: #450a0a; --del: #fca5a5; --ins-bg: #052e16; --ins: #86efac; --sel: #172554;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, sans-serif; }
header { padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--panel);
  display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; position: sticky; top: 0; z-index: 2; }
header h1 { font-size: 16px; margin: 0 8px 0 0; }
label { color: var(--muted); font-size: 12px; display: flex; gap: 4px; align-items: center; }
select, input { background: var(--panel); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 3px 6px; font: inherit; }
main { display: grid; grid-template-columns: minmax(320px, 40%) 1fr; height: calc(100vh - 100px); }
#list { overflow: auto; border-right: 1px solid var(--border); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 6px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
th { position: sticky; top: 0; background: var(--panel); cursor: pointer; font-size: 12px; color: var(--muted); }
tr.row { cursor: pointer; }
tr.row:hover, tr.row.sel { background: var(--sel); }
.loop { color: var(--bad); font-weight: 600; }
.tag { font-size: 11px; color: var(--muted); }
#detail { overflow: auto; padding: 12px 16px; }
#detail h2 { font-size: 15px; margin: 0 0 6px; }
.cols { display: grid; grid-template-columns: minmax(200px, 38%) 1fr; gap: 12px; align-items: start; }
.cols img { width: 100%; border: 1px solid var(--border); border-radius: 6px; background: #fff; }
.text { white-space: pre-wrap; background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px; margin: 4px 0 12px; font: 13px/1.5 ui-monospace, monospace; }
del { background: var(--del-bg); color: var(--del); }
ins { background: var(--ins-bg); color: var(--ins); text-decoration: none; }
details summary { cursor: pointer; font-weight: 600; margin-top: 4px; }
.count { color: var(--muted); font-size: 12px; }
@media (max-width: 800px) {
  main { grid-template-columns: 1fr; height: auto; }
  #list { max-height: 45vh; }
  .cols { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>Eval review</h1>
  <label>Run <select id="run"></select></label>
  <label>Compare <select id="base"></select></label>
  <label>Journal <select id="journal"><option value="">all</option></select></label>
  <label>Split <select id="split"><option value="">all</option><option>train</option><option>test</option></select></label>
  <label>Set <select id="set"><option value="">all</option><option>loop</option><option>control</option></select></label>
  <label>Looped <select id="looped"><option value="">any</option><option value="run">in run</option>
    <option value="any">in any run</option><option value="no">not in run</option></select></label>
  <label>Flag <select id="flag"><option value="">any</option><option value="__none">no flags</option></select></label>
  <label><input type="checkbox" id="heading"> heading label</label>
  <label>Min CER <input id="mincer" type="number" step="0.05" min="0" style="width:5em"></label>
  <label>Search <input id="q" placeholder="label text" style="width:9em"></label>
  <span class="count" id="count"></span>
</header>
<main>
  <div id="list"><table><thead><tr id="thead"></tr></thead><tbody id="rows"></tbody></table></div>
  <div id="detail"><p class="count">Select a page.</p></div>
</main>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const runs = DATA.runs, pages = DATA.pages;
let sortKey = "cer", sortDir = -1, selected = null;

$("base").add(new Option("(none)", ""));
for (const r of runs) { $("run").add(new Option(r, r)); $("base").add(new Option(r, r)); }
$("run").value = runs[runs.length - 1];
$("base").value = runs.length > 1 ? runs[0] : "";
for (const j of [...new Set(pages.map(p => p.journal))]) $("journal").add(new Option(j, j));
for (const f of [...new Set(pages.flatMap(p => p.flags))].sort()) $("flag").add(new Option(f, f));

const fmt = (x) => x == null ? "–" : x.toFixed(3);
const cer = (p, r) => p.runs[r] ? p.runs[r].cer : null;

function filtered() {
  const run = $("run").value, q = $("q").value.toLowerCase(), min = parseFloat($("mincer").value);
  return pages.filter(p => {
    const rr = p.runs[run];
    if ($("journal").value && p.journal !== $("journal").value) return false;
    if ($("split").value && p.split !== $("split").value) return false;
    if ($("set").value && p.set !== $("set").value) return false;
    const lv = $("looped").value;
    if (lv === "run" && !(rr && rr.looped)) return false;
    if (lv === "no" && rr && rr.looped) return false;
    if (lv === "any" && !Object.values(p.runs).some(x => x.looped)) return false;
    const fv = $("flag").value;
    if (fv === "__none" && p.flags.length) return false;
    if (fv && fv !== "__none" && !p.flags.includes(fv)) return false;
    if ($("heading").checked && !p.heading_label) return false;
    if (!isNaN(min) && !(rr && rr.cer >= min)) return false;
    if (q && !p.reference.toLowerCase().includes(q)) return false;
    return true;
  });
}

function sortVal(p) {
  const run = $("run").value, base = $("base").value;
  if (sortKey === "page") return p.journal + String(p.page).padStart(4, "0");
  if (sortKey === "delta") return base && cer(p, run) != null && cer(p, base) != null ? cer(p, run) - cer(p, base) : -1e9;
  if (sortKey === "ned") return p.runs[run] ? p.runs[run].ned : -1;
  if (sortKey === "ref") return p.ref_chars;
  return cer(p, run) ?? -1;
}

function render() {
  const run = $("run").value, base = $("base").value;
  const cols = [["page", "Page"], ["set", "Set"], ["cer", "CER"], ["ned", "NED"]];
  if (base) cols.push(["delta", "Δ CER vs compare"]);
  cols.push(["ref", "Label chars"], ["flags", "Flags"]);
  $("thead").innerHTML = cols.map(([k, t]) => `<th data-k="${k}">${t}${k === sortKey ? (sortDir < 0 ? " ▼" : " ▲") : ""}</th>`).join("");
  for (const th of $("thead").children) th.onclick = () => {
    const k = th.dataset.k; if (k === "flags" || k === "set") return;
    sortDir = k === sortKey ? -sortDir : -1; sortKey = k; render();
  };
  const list = filtered().sort((a, b) => (sortVal(a) > sortVal(b) ? 1 : sortVal(a) < sortVal(b) ? -1 : 0) * sortDir);
  $("count").textContent = `${list.length} / ${pages.length} pages`;
  $("rows").innerHTML = list.map((p) => {
    const rr = p.runs[run] || {};
    const d = base && cer(p, run) != null && cer(p, base) != null ? cer(p, run) - cer(p, base) : null;
    const tags = [...p.flags, p.heading_label ? "heading_label" : ""].filter(Boolean).join(", ");
    return `<tr class="row${selected === p ? " sel" : ""}" data-i="${pages.indexOf(p)}">
      <td>${p.journal}/${p.page} <span class="tag">${p.split}</span></td><td>${p.set}</td>
      <td class="${rr.looped ? "loop" : ""}">${fmt(rr.cer)}${rr.looped ? " ↻" : ""}</td><td>${fmt(rr.ned)}</td>
      ${base ? `<td>${d == null ? "–" : (d > 0 ? "+" : "") + d.toFixed(3)}</td>` : ""}
      <td>${p.ref_chars}</td><td class="tag">${tags}</td></tr>`;
  }).join("");
  for (const tr of $("rows").children) tr.onclick = () => { selected = pages[tr.dataset.i]; render(); show(selected); };
}

function show(p) {
  const run = $("run").value, base = $("base").value;
  const order = [run, ...(base && base !== run ? [base] : []), ...runs.filter(r => r !== run && r !== base)];
  const blocks = order.filter(r => p.runs[r]).map((r, i) => {
    const x = p.runs[r];
    const head = `${r} — CER ${fmt(x.cer)}, NED ${fmt(x.ned)}${x.looped ? ' <span class="loop">looped</span>' : ""}${x.status !== "ok" ? " (" + x.status + ")" : ""}`;
    return `<details ${i < 2 ? "open" : ""}><summary>${head}</summary><div class="text">${x.diff || "(no output)"}</div></details>`;
  }).join("");
  $("detail").innerHTML = `<h2>${p.journal} / page ${p.page} <span class="tag">${p.split} · ${p.set || "–"} · ${p.flags.join(", ")}</span></h2>
    <p class="count"><del>red</del> = in the label, missing from the output · <ins>green</ins> = in the output, not in the label</p>
    <div class="cols"><a href="${p.image}" target="_blank"><img src="${p.image}" loading="lazy" alt="page image"></a>
    <div><details open><summary>Label</summary><div class="text">${p.reference}</div></details>${blocks}</div></div>`;
}

for (const id of ["run", "base", "journal", "split", "set", "looped", "flag", "heading", "mincer", "q"])
  $(id).addEventListener("input", () => { render(); if (selected) show(selected); });
render();
</script>
</body>
</html>
"""
