"""Evals for the cartridge pipeline: eval sets, metrics, the GRPO reward.

Eval set (data/cartridge/eval/):
- heldout_pages.json: the passages (datagen.qa.passages) held out of training
  QA/GRPO pairs (self-study still covers the whole corpus, so the test is
  unseen questions about corpus facts). heldout_split() picks every k-th
  passage in timeline order, so the held-out pages spread over all dates.
- qa_heldout.jsonl: verified pairs (datagen.qa.verify) on held-out passages,
  fields question, answer, answer_type, citations, evidence (+ passage_id,
  date); plus "not in journal" items with answer_type "unanswerable" and
  answer ABSTAIN. build_eval_set() writes it with the teacher.

Every system (base, teacher ICL, cartridge) answers with ANSWER_SYSTEM_PROMPT:
a short answer, then [journal:p.N] citations, or ABSTAIN. Per item:
- correct: the judge's verdict (the teacher model at temperature 0, shown the
  reference answer and evidence, not the timeline), or without a judge a
  heuristic: the reference tokens appear in the answer in order, or F1 >= 0.8;
  for an unanswerable item, whether the answer abstains;
- f1: SQuAD token F1 against the reference, citations removed;
- citation_valid: every cited page exists in the corpus;
- citation_hit: some cited page is a reference page or contains the evidence.

The judge and teacher are callables messages -> reply text (answerers take a
batch of messages), so a Modal class, an OpenAI-compatible client or a fake in
tests all fit. evaluate() scores one system; run_eval() wraps it in a
tracking run (job_type eval, one run per system).
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from rapidfuzz import fuzz

from cartridges.datagen.qa import EVIDENCE_MIN_SCORE, PROMPT, Passage, _normalize, verify
from cartridges.datagen.qa import passages as qa_passages

Messages = list[dict]
Judge = Callable[[Messages], str]

ABSTAIN = "NOT IN JOURNAL"
UNANSWERABLE = "unanswerable"

ANSWER_SYSTEM_PROMPT = f"""You answer questions about the author's handwritten \
journals. Pages are cited as [<journal>:p.<N>], e.g. [Auto biography:p.12].
Reply with the shortest complete answer, then the pages that support it, e.g. \
"8 [Auto biography:p.2]". If the journals do not say, reply exactly: {ABSTAIN}"""

JUDGE_PROMPT = """You grade an answer to a question about a personal journal.

Question: {question}
Reference answer: {answer}
Evidence from the journal: {evidence}

Candidate answer: {candidate}

The candidate is CORRECT if it states the same fact as the reference answer \
(wording, date format and units may differ; extra detail is fine if it does not \
contradict the evidence). It is INCORRECT if it is wrong, incomplete for a list, \
contradicts the evidence, or declines to answer. Ignore citations.
Reply with one word: CORRECT or INCORRECT."""

# [Journal:p.N], or the angle form [<Journal:p.N>]; a half-angled one such as the
# placeholder [<journal>:p.N] is kept verbatim, so it fails validation.
_CITATION = re.compile(r"\[(<?)([^\[\]]+?:p\.\d+)(>?)\]")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_ABSTAIN_PATTERNS = re.compile(
    r"not in (the )?journal|(journals?|entries|pages) (do|does) not (say|mention)|"
    r"not mentioned|no (mention|record)|(doesn't|don't|does not|do not) say|i (don't|do not) know",
    re.I,
)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_M = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
      r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_D = r"(\d{1,2})(?:st|nd|rd|th)?"
# Each date form -> (year, month, day) groups; a missing part is None.
_DATES = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b"), lambda m: (m[3], m[1], m[2])),
    (re.compile(rf"\b{_M}\s+{_D}\b(?:,?\s+(\d{{4}})\b)?"), lambda m: (m[3], m[1], m[2])),
    (re.compile(rf"\b{_D}\s+(?:of\s+)?{_M}(?:,?\s+(\d{{4}})\b)?"), lambda m: (m[3], m[2], m[1])),
    (re.compile(rf"\b{_M},?\s+(\d{{4}})\b"), lambda m: (m[2], m[1], None)),
]


def _canonical_date(year: str | None, month: str, day: str | None) -> str:
    """" <year> <month> <day> " as plain numbers, the parts that are given."""
    month = str(_MONTHS.index(month[:3]) + 1) if month[:3].isalpha() else str(int(month))
    year = None if year is None else str(int(year) + 2000 if len(year) == 2 else int(year))
    return " " + " ".join(x for x in (year, month, None if day is None else str(int(day))) if x) + " "


def normalize_dates(text: str) -> str:
    """Write dates as "<year> <month> <day>" numbers, so "Nov 20th, 2021",
    "20 November 2021", "11/20/21" and "2021-11-20" all read "2021 11 20" (and
    "Nov 20" reads "11 20"). Expects lowercase text."""
    for pattern, parts in _DATES:
        text = pattern.sub(lambda m: _canonical_date(*parts(m)), text)
    return text


# --- data ---------------------------------------------------------------


def load_pages(out_dir: Path) -> dict[str, str]:
    """Every page of the timeline in `out_dir` as {"<journal>:p.<N>": text}."""
    from cartridges.batch.timeline import timeline_entries

    return {f"{e.journal}:p.{n}": text for e in timeline_entries(out_dir) for n, text in e.pages}


def heldout_split(passages: list[Passage], frac: float = 0.15, seed: int = 0) -> list[Passage]:
    """Every round(1/frac)-th passage in timeline order, from a seeded offset."""
    step = max(1, round(1 / frac))
    offset = random.Random(seed).randrange(step)
    return passages[offset::step]


def write_split(path: Path, heldout: list[Passage], frac: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = sorted({c for p in heldout for c in p.citations})
    split = {"frac": frac, "seed": seed, "passage_ids": [p.id for p in heldout], "pages": pages}
    path.write_text(json.dumps(split, indent=1) + "\n")


def heldout_pages(path: Path) -> set[str]:
    """The held-out page citations; training QA must not cite any of them."""
    return set(json.loads(path.read_text())["pages"])


def parse_json(reply: str, key: str) -> list:
    """reply[key] when the reply (thinking and code fences removed) is a JSON
    object with a list there; [] otherwise."""
    reply = re.sub(r"^```(?:json)?|```$", "", strip_think(reply).strip()).strip()
    try:
        value = json.loads(reply)[key]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return value if isinstance(value, list) else []


def parse_pairs(reply: str) -> list[dict]:
    """The pairs in a reply to datagen.qa.PROMPT; [] if it is not that JSON."""
    return [p for p in parse_json(reply, "pairs") if isinstance(p, dict)]


def build_heldout(heldout: list[Passage], generate: Callable[[list[Messages]], list[str]], max_items: int = 150,
                  pairs_per_passage: int = 6) -> tuple[list[dict], int]:
    """Verified pairs on held-out passages, at most max_items, spread evenly:
    each passage gives its pairs in turn, one at a time. Duplicate questions
    are dropped. `generate` is the teacher (any context) on a batch of
    prompts; returns the items and the number of pairs written before
    verification."""
    prompts = [[{"role": "user", "content": PROMPT.format(n=pairs_per_passage, example=p.citations[0], text=p.text)}]
               for p in heldout]
    per_passage, n_written = [], 0
    for passage, reply in zip(heldout, generate(prompts), strict=True):
        pairs = parse_pairs(reply)
        n_written += len(pairs)
        per_passage.append([
            {**{k: pair[k] for k in ("question", "answer", "answer_type", "citations", "evidence")},
             "passage_id": passage.id, "date": passage.pages[0][1]}
            for pair in pairs
            if verify(pair, passage)
        ])
    items, seen = [], set()
    for round_ in range(max((len(k) for k in per_passage), default=0)):
        for kept in per_passage:
            if round_ < len(kept) and _normalize(kept[round_]["question"]) not in seen:
                seen.add(_normalize(kept[round_]["question"]))
                items.append(kept[round_])
    return items[:max_items], n_written


UNANSWERABLE_PROMPT = """Write {n} questions the author of these journals might \
plausibly ask about their own life, in the same style as questions about the \
journals (naming dates, people, places or moods), whose answers are NOT in the \
journals. Mix near misses (a real person or month, an unrecorded detail) with \
topics the journals never touch. Reply with JSON only: {{"questions": ["..."]}}"""


def build_unanswerable(generate: Callable[[list[Messages]], list[str]], answer: Callable[[list[Messages]], list[str]],
                       n: int = 30, per_call: int = 15) -> list[dict]:
    """"Not in journal" items: `generate` (the teacher, timeline in context,
    sampled) proposes 2n questions, per_call per reply so a reply fits its
    token budget; we keep those the teacher, answering with the timeline in
    context, abstains on."""
    prompt = [{"role": "user", "content": UNANSWERABLE_PROMPT.format(n=per_call)}]
    replies = generate([prompt] * -(-2 * n // per_call))
    questions = list(dict.fromkeys(q.strip() for r in replies for q in parse_json(r, "questions")
                                   if isinstance(q, str) and q.strip()))
    answers = answer([question_messages(q) for q in questions])
    return [
        {"question": q, "answer": ABSTAIN, "answer_type": UNANSWERABLE, "citations": [], "evidence": ""}
        for q, r in zip(questions, answers, strict=True)
        if is_abstention(r)
    ][:n]


def build_eval_set(
    out_dir: Path,
    split_path: Path,
    write: Callable[[list[Messages]], list[str]],
    answer: Callable[[list[Messages]], list[str]],
    path: Path,
    max_items: int = 150,
    n_unanswerable: int = 30,
) -> dict:
    """Write qa_heldout.jsonl: verified pairs on the held-out passages, then
    the "not in journal" items. `write` (sampled) writes pairs and proposes
    questions, `answer` (greedy) is the abstention filter; both are the teacher
    with the timeline in context. Returns datagen_quality() of the pairs."""
    ids = set(json.loads(split_path.read_text())["passage_ids"])
    held = [p for p in qa_passages(out_dir) if p.id in ids]
    items, n_written = build_heldout(held, write, max_items)
    unanswerable = build_unanswerable(write, answer, n_unanswerable)
    write_jsonl(path, items + unanswerable)
    return datagen_quality(n_written, items, held) | {"n_unanswerable": len(unanswerable)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- metrics ------------------------------------------------------------


def question_messages(question: str) -> Messages:
    return [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": question}]


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def parse_citations(text: str) -> list[str]:
    """Cited anchors in order, deduplicated: [J:p.N] and [<J:p.N>] give "J:p.N"."""
    cites = (c if lt and gt else lt + c + gt for lt, c, gt in _CITATION.findall(text))
    return list(dict.fromkeys(c.strip() for c in cites))


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_PATTERNS.search(strip_think(text)))


def _tokens(text: str) -> list[str]:
    """SQuAD normalization: lowercase, no punctuation, no articles; dates
    in one form (normalize_dates)."""
    text = normalize_dates(_CITATION.sub(" ", strip_think(text)).lower())
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return [t for t in text.split() if t not in ("a", "an", "the")]


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _tokens(prediction), _tokens(reference)
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or not common:
        return float(pred == ref)
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def heuristic_correct(prediction: str, reference: str) -> bool:
    """The reference tokens appear contiguously in the prediction, or F1 >= 0.8."""
    pred, ref = _tokens(prediction), _tokens(reference)
    if ref and any(pred[i : i + len(ref)] == ref for i in range(len(pred) - len(ref) + 1)):
        return True
    return token_f1(prediction, reference) >= 0.8


def citation_metrics(completion: str, item: dict, pages: dict[str, str]) -> dict:
    cited = parse_citations(strip_think(completion))
    evidence = _normalize(item.get("evidence", ""))
    hit = any(
        c in item.get("citations", [])
        or (evidence and c in pages and fuzz.partial_ratio(evidence, _normalize(pages[c])) >= EVIDENCE_MIN_SCORE)
        for c in cited
    )
    return {"n_cited": len(cited), "citation_valid": bool(cited) and all(c in pages for c in cited), "citation_hit": hit}


def judge_correct(judge: Judge, item: dict, completion: str) -> bool:
    prompt = JUDGE_PROMPT.format(
        question=item["question"], answer=item["answer"], evidence=item["evidence"], candidate=strip_think(completion)
    )
    verdict = re.search(r"\b(INCORRECT|CORRECT)\b", strip_think(judge([{"role": "user", "content": prompt}])).upper())
    return bool(verdict) and verdict.group(1) == "CORRECT"


def score_item(item: dict, completion: str, pages: dict[str, str], judge: Judge | None = None) -> dict:
    """The metrics of one answer (see module docstring)."""
    abstained = is_abstention(completion)
    if item["answer_type"] == UNANSWERABLE:
        return {"correct": abstained, "abstained": abstained, "f1": float(abstained),
                "n_cited": 0, "citation_valid": None, "citation_hit": None}
    if abstained:
        correct = False
    elif judge is not None:
        correct = judge_correct(judge, item, completion)
    else:
        correct = heuristic_correct(completion, item["answer"])
    return {"correct": correct, "abstained": abstained, "f1": token_f1(completion, item["answer"]),
            **citation_metrics(completion, item, pages)}


def grounded_reward(completion: str, item: dict, pages: dict[str, str], judge: Judge | None = None) -> float:
    """GRPO reward in [0, 1]: 0.6 correct + 0.2 token F1 + 0.2 citation hit;
    for an unanswerable item, 1 if the answer abstains, else 0."""
    s = score_item(item, completion, pages, judge)
    if item["answer_type"] == UNANSWERABLE:
        return float(s["correct"])
    return 0.6 * s["correct"] + 0.2 * s["f1"] + 0.2 * s["citation_hit"]


def grounded_rewards(completions: list[str], items: list[dict], pages: dict[str, str], judge: Judge | None = None) -> list[float]:
    return [grounded_reward(c, i, pages, judge) for c, i in zip(completions, items, strict=True)]


# --- reports ------------------------------------------------------------


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-item rows (item fields + score_item fields)."""
    answerable = [r for r in rows if r["answer_type"] != UNANSWERABLE]
    unanswerable = [r for r in rows if r["answer_type"] == UNANSWERABLE]
    by_type = {
        t: _mean([r["correct"] for r in answerable if r["answer_type"] == t])
        for t in sorted({r["answer_type"] for r in answerable})
    }
    return {
        "n": len(answerable),
        "n_unanswerable": len(unanswerable),
        "accuracy": _mean([r["correct"] for r in answerable]),
        "token_f1": _mean([r["f1"] for r in answerable]),
        "citation_rate": _mean([r["n_cited"] > 0 for r in answerable]),
        "citation_validity": _mean([r["citation_valid"] for r in answerable]),
        "citation_hit": _mean([r["citation_hit"] for r in answerable]),
        "false_abstention": _mean([r["abstained"] for r in answerable]),
        "abstention_rate": _mean([r["abstained"] for r in unanswerable]),
        "accuracy_by_type": by_type,
    }


def evaluate(
    system: str,
    items: list[dict],
    answer: Callable[[list[Messages]], list[str]],
    pages: dict[str, str],
    judge: Judge | None = None,
    **extra,
) -> dict:
    """Ask `system` every item (one batched call) and score the answers.
    `extra` (e.g. ttft_s, cache_bytes, step) is stored with the summary."""
    completions = answer([question_messages(i["question"]) for i in items])
    rows = [
        {"system": system, **item, "completion": c, **score_item(item, c, pages, judge)}
        for item, c in zip(items, completions, strict=True)
    ]
    return {"system": system, **summarize(rows), **extra, "rows": rows}


def compare(reports: dict[str, dict], threshold: float = 0.9) -> dict:
    """The success check: cartridge accuracy / teacher accuracy >= threshold,
    and the cartridge beats the base model."""
    teacher, cartridge, base = (reports[k]["accuracy"] or 0.0 for k in ("teacher", "cartridge", "base"))
    ratio = cartridge / teacher if teacher else None
    return {"cartridge_over_teacher": ratio, "cartridge_minus_base": cartridge - base,
            "success": ratio is not None and ratio >= threshold and cartridge > base}


def datagen_quality(n_generated: int, items: list[dict], passages: list[Passage]) -> dict:
    """Verify pass rate and how the items cover the corpus by month."""
    months = sorted({p.pages[0][1][:7] for p in passages if p.pages[0][1][:4].isdigit()})
    covered = Counter(i["date"][:7] for i in items if i.get("date", "")[:4].isdigit())
    return {
        "n_generated": n_generated,
        "n_verified": len(items),
        "verify_pass_rate": len(items) / n_generated if n_generated else None,
        "n_passages": len({i.get("passage_id") for i in items}),
        "months_covered": sum(m in covered for m in months) / len(months) if months else None,
        "items_per_month": dict(sorted(covered.items())),
    }


def cache_bytes(n_tokens: int, n_full_layers: int = 16, kv_heads: int = 4, head_dim: int = 256, dtype_bytes: int = 2) -> int:
    """Bytes of the full-attention KV cache for n_tokens (Qwen3.5/3.8 defaults:
    16 full-attention layers, 4 KV heads, head_dim 256, bf16). The linear
    attention states are a constant on top, the same for ICL and cartridge."""
    return 2 * n_full_layers * kv_heads * head_dim * dtype_bytes * n_tokens


def log_report(run, report: dict, step: int | None = None) -> None:
    """Log a report to a tracking.TrackedRun: the scalars as eval/<name> (also
    in the run summary, which tracking.group_summary_markdown reads) and the
    per-item rows as the W&B table eval/items."""
    import wandb

    scalars = {k: v for k, v in report.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    scalars |= {f"accuracy_{t}": v for t, v in report["accuracy_by_type"].items() if v is not None}
    scalars = {f"eval/{k}": v for k, v in scalars.items()}
    rows = report["rows"]
    columns = list(dict.fromkeys(c for r in rows for c in r))  # unanswerable rows lack passage_id, date
    cell = lambda v: json.dumps(v) if isinstance(v, list) else v  # noqa: E731
    table = wandb.Table(columns=columns, data=[[cell(r.get(c)) for c in columns] for r in rows])
    run.log(scalars | {"eval/items": table}, step=step)
    run.run.summary.update(scalars)


def run_eval(
    system: str,
    answer: Callable[[list[Messages]], list[str]],
    group: str,
    model: str,
    *,
    items_path: Path = Path("data/cartridge/eval/qa_heldout.jsonl"),
    out_dir: Path = Path("data/journals/output/2026_10_13"),
    judge: Judge | None = None,
    p: int = 0,
    gpu: str | None = None,
    cartridge_ref: str | None = None,
    config: dict | None = None,
    **extra,
) -> dict:
    """One eval run (job_type eval) of `system` (base | teacher | cartridge)
    on qa_heldout.jsonl, logged in `group`. `extra` (ttft_s, cache_mem_gb, ...)
    goes into the report and the run summary."""
    from cartridges.cartridge import tracking

    items, pages = read_jsonl(items_path), load_pages(out_dir)
    config = {**(config or {}), "system": system, "items": str(items_path), "items_sha256": file_sha256(items_path)}
    with tracking.init_run("eval", model, group, config, p=p, gpu=gpu, role=system) as run:
        if cartridge_ref:
            run.use_artifact(cartridge_ref, type="cartridge")
        columns = list(dict.fromkeys(k for r in items for k in r))
        run.log_artifact(items_path, type="dataset", name="qa-heldout", table_columns=columns)  # versioned by content
        report = evaluate(system, items, answer, pages, judge, **extra)
        log_report(run, report)
    return report


# --- teacher-backed systems ---------------------------------------------


def teacher_system(model_id: str, with_context: bool, max_tokens: int = 256, temperature: float | None = 0.0,
                   workers: int = 32):
    """(complete, answer) on the deployed teacher.Teacher: complete(messages)
    -> teacher.Completion (text, ttft_s, ...), answer(batch) -> texts, run in
    parallel (the container batches them). with_context=False is the base
    model, which is also the judge (temperature 0, a few tokens). Every call
    goes to the one Teacher service pool of model_id."""
    from concurrent.futures import ThreadPoolExecutor

    from cartridges.cartridge.teacher import teacher_cls

    teacher = teacher_cls("Teacher", model_id)

    def complete(messages: Messages):
        return teacher.generate.remote(messages, max_tokens=max_tokens, with_context=with_context,
                                       temperature=temperature)[0]

    def answer(batch: list[Messages]) -> list[str]:
        with ThreadPoolExecutor(workers) as pool:
            return [o.text for o in pool.map(complete, batch)]

    return complete, answer


def build(model_id: str = "Qwen/Qwen3.5-4B", out_name: str = "qa_heldout_dev.jsonl",
          out_dir: str = "data/journals/output/2026_10_13", eval_dir: str = "data/cartridge/eval",
          max_items: int = 150, n_unanswerable: int = 30) -> dict:
    """Build an eval set with the deployed teacher (timeline in context;
    pairs are sampled at the teacher's default temperature). 4B builds the dev
    set; the final proof uses qa_heldout.jsonl built with 27B, frozen by its
    sha256 before any 27B training run."""
    _check_budget(model_id, hours=0.5)
    _, write = teacher_system(model_id, with_context=True, max_tokens=2048, temperature=None)
    _, answer = teacher_system(model_id, with_context=True)
    path = Path(eval_dir) / out_name
    quality = build_eval_set(Path(out_dir), Path(eval_dir) / "heldout_pages.json", write, answer, path,
                             max_items, n_unanswerable)
    return quality | {"path": str(path), "sha256": file_sha256(path), "model_id": model_id}


def _check_budget(model_id: str, hours: float) -> None:
    """tracking.check_budget for one Teacher container for `hours` (startup
    and the 5 min scaledown included)."""
    from cartridges.cartridge import tracking
    from cartridges.cartridge.teacher import GPU_FOR_MODEL

    tracking.check_budget(hours * tracking.GPU_RATES_USD_PER_HOUR[GPU_FOR_MODEL[model_id]])


def file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eval_teacher_systems(group: str, model_id: str = "Qwen/Qwen3.5-4B",
                         items_path: str = "data/cartridge/eval/qa_heldout.jsonl",
                         out_dir: str = "data/journals/output/2026_10_13", n_ttft: int = 5) -> dict:
    """The base and teacher (ICL) eval runs in `group`, judged by the base
    model at temperature 0 without the timeline. TTFT is the median of n_ttft
    sequential requests (the batched answers queue, so theirs is not), with
    the context cached; ttft_cold_s is the cold prefill of the context."""
    import statistics

    from cartridges.cartridge.teacher import teacher_cls

    _check_budget(model_id, hours=0.75)
    judge_complete, _ = teacher_system(model_id, with_context=False, max_tokens=8)
    questions = [i["question"] for i in read_jsonl(Path(items_path))[:n_ttft]]
    reports = {}
    for system, with_context in (("base", False), ("teacher", True)):
        complete, answer = teacher_system(model_id, with_context=with_context)
        ttft = statistics.median(complete(question_messages(q)).ttft_s for q in questions)
        info = teacher_cls("Teacher", model_id).info.remote() if with_context else {}
        report = run_eval(system, answer, group, model_id, items_path=Path(items_path), out_dir=Path(out_dir),
                          judge=lambda m: judge_complete(m).text, config={"judge": model_id},
                          ttft_s=ttft, ttft_cold_s=info.get("prefill_s", ttft), cache_mem_gb=info.get("cache_gb", 0.0))
        reports[system] = {k: v for k, v in report.items() if k != "rows"}
    return reports


def main() -> None:
    import fire

    fire.Fire({"build": build, "eval-teacher": eval_teacher_systems})


if __name__ == "__main__":
    main()
