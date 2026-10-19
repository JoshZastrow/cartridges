import json
from collections import Counter

from cartridges.cartridge.tasks.psych import heldout_instruments, load
from cartridges.cartridge.tasks.validate import task_id, text_hash


def write_corpus(d, n_instruments=20, items=3):
    tasks, items_rows, instruments = [], [], []
    for i in range(n_instruments):
        category = "big_five" if i < 10 else "attachment" if i < 16 else "dark_traits"
        instruments.append({"instrument_id": f"ins{i}", "category": category})
        for j in range(items):
            text = f"Answer item {j} of test {i} the way I would."
            h = text_hash(text)
            tasks.append({"task_id": str(task_id(h)), "text": text, "text_hash": h, "kind": "framework_lens",
                          "style": None, "framework": f"fw{i % 4}", "role": None, "origin": "human_template",
                          "split": "train", "status": "validated", "reject_reason": None, "source_citations": [],
                          "ref_citations": [], "ref_evidence": [], "grounded": None, "parent_ids": [],
                          "raw_reply": None, "run_id": "psych-run", "created_at": "2026-10-16T00:00:00+00:00"})
            items_rows.append({"task_id": str(task_id(h)), "instrument_id": f"ins{i}", "item_no": j + 1,
                               "license": "public_domain", "category": category})
    (d / "tasks.jsonl").write_text("\n".join(map(json.dumps, tasks)) + "\n")
    (d / "items.jsonl").write_text("\n".join(map(json.dumps, items_rows)) + "\n")
    (d / "instruments.jsonl").write_text("\n".join(map(json.dumps, instruments)) + "\n")
    (d / "gen_run.json").write_text(json.dumps({"run_id": "psych-run", "component": "seed", "model": None,
                                                "prompt_version": "psych-tests-v1", "config": {"origin": "human_template"}}))
    return instruments


def test_heldout_instruments_are_stratified_and_deterministic(tmp_path):
    instruments = write_corpus(tmp_path)
    held = heldout_instruments(instruments, frac=0.1, seed=0)
    assert len(held) == 2 and held == heldout_instruments(instruments, frac=0.1, seed=0)
    cats = Counter(next(i["category"] for i in instruments if i["instrument_id"] == h) for h in held)
    assert cats == {"big_five": 1, "attachment": 1}  # by category size, largest remainder


def test_load_sets_origin_split_and_metadata(tmp_path):
    write_corpus(tmp_path)
    run, tasks, held = load(tmp_path)
    assert run.run_id == "psych-run" and run.config["origin"] == "psych_test"
    assert len(tasks) == 60 and {t.origin for t in tasks} == {"psych_test"}
    for t in tasks:
        assert (t.split == "eval") == (t.metadata["instrument_id"] in held)  # whole instruments held out
        assert set(t.metadata) == {"instrument_id", "item_no", "license", "category"}
    assert Counter(t.split for t in tasks) == {"train": 54, "eval": 6}
