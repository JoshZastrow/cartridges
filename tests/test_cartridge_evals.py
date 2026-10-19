import json

import pytest

from cartridges.cartridge.evals import (
    ABSTAIN,
    build_eval_set,
    build_heldout,
    build_unanswerable,
    cache_bytes,
    citation_metrics,
    compare,
    datagen_quality,
    evaluate,
    grounded_reward,
    heldout_pages,
    heldout_split,
    heuristic_correct,
    load_pages,
    normalize_dates,
    parse_citations,
    parse_pairs,
    run_eval,
    token_f1,
    write_split,
)
from cartridges.datagen.qa import Passage

MOOD = "M: Passionate-9, warm-6, content-8."
PAGES = {"A:p.2": "S: Shopping for a couch.\n\n" + MOOD, "A:p.3": "S: Walked to the lake."}
ITEM = {
    "question": "How content did I feel the evening I shopped for a couch in Nov 2021?",
    "answer": "8",
    "answer_type": "number",
    "citations": ["A:p.2"],
    "evidence": "Passionate-9, warm-6, content-8",
    "passage_id": "A:p.2",
    "date": "2021-11-20",
}
NOT_IN = {"question": "What car did I buy in 2023?", "answer": ABSTAIN, "answer_type": "unanswerable",
          "citations": [], "evidence": ""}


def passage(n, date):
    return Passage("A", ((n, date, f"page {n} text " * 20),))


def test_heldout_split_spreads_over_the_timeline_and_round_trips(tmp_path):
    ps = [passage(n, f"2022-{1 + n % 12:02d}-01") for n in range(1, 41)]
    held = heldout_split(ps, frac=0.2, seed=3)
    assert len(held) == 8 and held == heldout_split(ps, frac=0.2, seed=3)
    gaps = {ps.index(b) - ps.index(a) for a, b in zip(held, held[1:])}
    assert gaps == {5}
    write_split(tmp_path / "split.json", held, 0.2, 3)
    assert heldout_pages(tmp_path / "split.json") == {c for p in held for c in p.citations}


def test_load_pages_reads_every_page(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    (d / "transcript.md").write_text("## p. 1 - date: 2021-11-19\n\nhello\n---\n## p. 2 - date: 2021-11-19 (cont.)\n\nworld\n")
    assert load_pages(tmp_path) == {"A:p.1": "hello", "A:p.2": "world"}


def test_token_f1_and_heuristic():
    assert token_f1("8 [A:p.2]", "8") == 1.0
    assert token_f1("The couch, a lamp", "couch lamp") == 1.0
    assert token_f1("blue", "red") == 0.0
    assert heuristic_correct("You felt content-8 that night [A:p.2]", "8")
    assert not heuristic_correct("28", "8")


@pytest.mark.parametrize(
    "text",
    ["2021-11-20", "11/20/2021", "11/20/21", "Nov 20, 2021", "November 20th 2021", "Nov. 20, 2021",
     "20 November 2021", "the 20th of November, 2021"],
)
def test_dates_in_any_form_match(text):
    assert token_f1(text, "2021-11-20") == 1.0
    assert heuristic_correct(f"It was {text} [A:p.2]", "2021-11-20")


def test_normalize_dates_partial_and_non_dates():
    assert normalize_dates("nov 20").split() == ["11", "20"]
    assert normalize_dates("in november 2021.").split() == ["in", "2021", "11", "."]
    assert normalize_dates("2021-11-05").split() == normalize_dates("nov 5, 2021").split() == ["2021", "11", "5"]
    assert normalize_dates("you may go in march") == "you may go in march"  # no number next to it
    assert not heuristic_correct("2021-11-21", "2021-11-20")
    assert grounded_reward("Nov 20, 2021 [A:p.2]", {**ITEM, "answer": "2021-11-20", "answer_type": "date"},
                           PAGES) == pytest.approx(1.0)


def test_angle_citations_parse_and_placeholders_are_invalid():
    assert parse_citations("8 [<A:p.2>] and [A:p.3]") == ["A:p.2", "A:p.3"]
    assert citation_metrics("8 [<A:p.2>]", ITEM, PAGES) == {"n_cited": 1, "citation_valid": True, "citation_hit": True}
    assert parse_citations("see [<journal>:p.127]") == ["<journal>:p.127"]
    assert citation_metrics("8 [A:p.2] [<journal>:p.127]", ITEM, PAGES)["citation_valid"] is False
    assert token_f1("8 [<A:p.2>]", "8") == 1.0  # angle citations are stripped before F1 too


def test_citations():
    assert parse_citations("8 [A:p.2] [Auto biography:p.12] [A:p.2]") == ["A:p.2", "Auto biography:p.12"]
    assert citation_metrics("8 [A:p.2]", ITEM, PAGES) == {"n_cited": 1, "citation_valid": True, "citation_hit": True}
    # a page that exists but lacks the evidence; a page that does not exist
    assert citation_metrics("8 [A:p.3]", ITEM, PAGES)["citation_hit"] is False
    assert citation_metrics("8 [A:p.99]", ITEM, PAGES)["citation_valid"] is False
    # not a reference page, but it contains the evidence
    other = {**ITEM, "citations": ["A:p.7"]}
    assert citation_metrics("8 [A:p.2]", other, PAGES)["citation_hit"] is True


@pytest.mark.parametrize(
    "completion, item, reward",
    [
        ("8 [A:p.2]", ITEM, 1.0),
        ("<think>hmm</think>8", ITEM, 0.8),  # right, uncited
        ("5 [A:p.3]", ITEM, 0.0),
        (ABSTAIN, ITEM, 0.0),
        (ABSTAIN, NOT_IN, 1.0),
        ("A red Honda [A:p.2]", NOT_IN, 0.0),
    ],
)
def test_grounded_reward(completion, item, reward):
    assert grounded_reward(completion, item, PAGES) == pytest.approx(reward)


def test_judge_decides_correctness():
    seen = []

    def judge(messages):
        seen.append(messages[0]["content"])
        return "<think>dates match</think> CORRECT"

    assert grounded_reward("November 20th [A:p.2]", {**ITEM, "answer": "2021-11-20", "answer_type": "date"},
                           PAGES, judge) == pytest.approx(0.96)  # judge: correct; F1 0.8 (no year)
    assert "Reference answer: 2021-11-20" in seen[0] and "Candidate answer: November 20th" in seen[0]
    assert grounded_reward("8 [A:p.2]", ITEM, PAGES, lambda m: "INCORRECT") == pytest.approx(0.4)


def test_evaluate_and_compare():
    def answers(correct):
        def answer(batch):
            assert batch[0][0]["role"] == "system"
            return [("8 [A:p.2]" if correct else "3") if "couch" in m[1]["content"] else ABSTAIN for m in batch]
        return answer

    items = [ITEM, NOT_IN]
    reports = {name: evaluate(name, items, answers(ok), PAGES, ttft_s=0.1)
               for name, ok in (("base", False), ("teacher", True), ("cartridge", True))}
    t = reports["teacher"]
    assert (t["accuracy"], t["citation_validity"], t["abstention_rate"], t["ttft_s"]) == (1.0, 1.0, 1.0, 0.1)
    assert t["rows"][0]["completion"] == "8 [A:p.2]" and t["accuracy_by_type"] == {"number": 1.0}
    assert reports["base"]["accuracy"] == 0.0 and reports["base"]["citation_rate"] == 0.0
    assert compare(reports) == {"cartridge_over_teacher": 1.0, "cartridge_minus_base": 1.0, "success": True}


GOOD1 = {k: ITEM[k] for k in ("question", "answer", "answer_type", "citations", "evidence")}
GOOD2 = {**GOOD1, "question": "Where did I walk on Dec 1 2021?", "answer": "the lake",
         "answer_type": "short_text", "citations": ["A:p.3"], "evidence": "Walked to the lake."}


def fake_teacher(messages):
    """Pair-writing replies by passage; a question list otherwise."""
    prompt = messages[0]["content"]
    if "[A:p.2]" in prompt:
        pairs = [GOOD1, {**GOOD1, "evidence": "made up"}, {**GOOD1, "question": "Second?"}, GOOD1]
        return "<think>ok</think>```json\n" + json.dumps({"pairs": pairs}) + "\n```"
    if "[A:p.3]" in prompt:
        return json.dumps({"pairs": [GOOD2]})
    return json.dumps({"questions": ["What car did I buy?", "Where did I walk on Dec 1?"]})


def fake_answer(batch):
    """Pair-writing prompts (no system turn) as fake_teacher; questions otherwise."""
    return [fake_teacher(m) if m[0]["role"] == "user" else ABSTAIN if "car" in m[1]["content"] else "the lake [A:p.3]"
            for m in batch]


def test_parse_pairs():
    assert parse_pairs(fake_teacher([{"content": "[A:p.2]"}]))[0] == GOOD1
    assert parse_pairs("nope") == [] and parse_pairs('{"pairs": 3}') == []


def test_build_heldout_keeps_verified_pairs_round_robin():
    p1 = Passage("A", ((2, "2021-11-20", PAGES["A:p.2"]),))
    p2 = Passage("A", ((3, "2021-12-01", PAGES["A:p.3"] + " " + "x" * 300),))
    items, n_written = build_heldout([p1, p2], fake_answer, max_items=10)
    assert [i["question"] for i in items] == [GOOD1["question"], GOOD2["question"], "Second?"]
    assert items[1]["passage_id"] == "A:p.3" and items[1]["date"] == "2021-12-01"
    q = datagen_quality(n_written, items, [p1, p2])
    assert q["verify_pass_rate"] == 0.6 and q["months_covered"] == 1.0


def write_journal(out):
    d = out / "A"
    d.mkdir(parents=True)
    pages = [(2, "2021-11-20", PAGES["A:p.2"] + " " + "y" * 300), (3, "2021-12-01", PAGES["A:p.3"] + " " + "x" * 300)]
    (d / "transcript.md").write_text("\n---\n".join(f"## p. {n} - date: {label}\n\n{text}\n" for n, label, text in pages))


def test_build_eval_set_writes_heldout_and_unanswerable(tmp_path):
    write_journal(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"passage_ids": ["A:p.3"], "pages": ["A:p.3"]}))
    quality = build_eval_set(tmp_path, split, fake_answer, fake_answer, tmp_path / "qa.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "qa.jsonl").read_text().splitlines()]
    assert [r["answer_type"] for r in rows] == ["short_text", "unanswerable"]
    assert quality["n_verified"] == 1 and quality["n_unanswerable"] == 1


def test_run_eval_logs_without_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    write_journal(tmp_path)
    items = tmp_path / "qa.jsonl"
    items.write_text(json.dumps({**GOOD2, "passage_id": "A:p.3", "date": "2021-12-01"}) + "\n" + json.dumps(NOT_IN) + "\n")
    report = run_eval("teacher", fake_answer, "exp-20261013-test", "Qwen/Qwen3.5-4B",
                      items_path=items, out_dir=tmp_path, gpu="cpu", ttft_s=0.5)
    assert report["accuracy"] == 1.0 and report["citation_hit"] == 1.0 and report["ttft_s"] == 0.5
    assert report["abstention_rate"] == 1.0  # NOT_IN lacks passage_id/date: the W&B table must still build


def test_build_unanswerable_keeps_questions_the_teacher_abstains_on():
    out = build_unanswerable(fake_answer, fake_answer, n=5)
    assert [o["question"] for o in out] == ["What car did I buy?"] and out[0]["answer_type"] == "unanswerable"
    assert build_unanswerable(lambda batch: ["not json"] * len(batch), fake_answer) == []


def test_cache_bytes():
    assert cache_bytes(1) == 65536  # 2 (K,V) x 16 layers x 4 heads x 256 dims x 2 bytes
