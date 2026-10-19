"""Scraped psychology and personality test items (data/corpus/psych_tests) as
origin "psych_test" tasks: framework_lens questions that ask the assistant to
answer one test item the way the user would.

Whole instruments, about 10% stratified by category, are held out as eval, so
eval measures tests the student has never seen. These tasks are not real user
questions and never become exemplars.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from cartridges.cartridge.tasks.sampler import split as largest_remainder
from cartridges.cartridge.tasks.types import GenRun, Task

EVAL_FRAC = 0.1


def heldout_instruments(instruments: list[dict], frac: float = EVAL_FRAC, seed: int = 0) -> set[str]:
    """round(frac * n) instrument ids, spread over categories by size
    (largest remainder), uniformly within a category."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for ins in instruments:
        by_category[ins["category"]].append(ins["instrument_id"])
    total = round(frac * len(instruments))
    counts = largest_remainder(total, {c: len(ids) for c, ids in sorted(by_category.items())})
    rng = random.Random(seed)
    return {i for c, k in counts.items() for i in rng.sample(sorted(by_category[c]), k)}


def load(directory: Path, frac: float = EVAL_FRAC, seed: int = 0) -> tuple[GenRun, list[Task], set[str]]:
    """(the gen_runs row, every task, the held-out instrument ids)."""
    directory = Path(directory)
    run = json.loads((directory / "gen_run.json").read_text())
    run["config"] = {**run.get("config", {}), "origin": "psych_test", "eval_frac": frac, "seed": seed}
    items = {r["task_id"]: r for r in map(json.loads, (directory / "items.jsonl").read_text().splitlines()) if r}
    instruments = [json.loads(line) for line in (directory / "instruments.jsonl").read_text().splitlines() if line]
    held = heldout_instruments(instruments, frac, seed)
    tasks = []
    for line in (directory / "tasks.jsonl").read_text().splitlines():
        row = {k: v for k, v in json.loads(line).items() if k not in ("raw_reply", "created_at")}
        item = items[row["task_id"]]
        tasks.append(Task(**{
            **row,
            "origin": "psych_test",
            "split": "eval" if item["instrument_id"] in held else "train",
            "metadata": {k: item[k] for k in ("instrument_id", "item_no", "license", "category")},
        }))
    return GenRun(**run), tasks, held
