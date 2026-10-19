"""Side-by-side judge (tasks/judge.py) with a fake judge model: no GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cartridges.cartridge.tasks import judge as jg


def item(i: int, origin: str = "chat", kind: str = "recall", x: str = "X answer", y: str = "Y answer") -> jg.Item:
    return jg.Item(task_id=f"t{i}", origin=origin, kind=kind, question=f"Q{i}?", x=x, y=y)


def verdict(winner="A", errors="none", reason="r"):
    return jg.Verdict(winner=winner, factual_errors=errors, reason=reason)


def test_rubric_v0_loads_formats_and_hashes():
    r = jg.load_rubric("v0")
    msg = jg.judge_messages(r, "Why?", "first", "second")[0]["content"]
    assert "Why?" in msg and msg.index("first") < msg.index("second") and '"winner"' in msg
    assert len(r.hash) == 12 and r.hash != jg.Rubric(version="v0", text=r.text + " ").hash
    assert jg.judge_config(r, "Qwen/Qwen3.8-27B", "cartridge", "teacher") == {
        "judge_model": "Qwen/Qwen3.8-27B", "judge_prompt_version": "v0", "judge_prompt_hash": r.hash,
        "x": "cartridge", "y": "teacher"}


def test_rubric_is_pluggable(tmp_path):
    (tmp_path / "judge-v1.md").write_text("Q {question} A {answer_a} B {answer_b}")
    r = jg.load_rubric("v1", root=tmp_path)
    assert jg.judge_messages(r, "q", "a", "b")[0]["content"] == "Q q A a B b" and r.version == "v1"


@pytest.mark.parametrize("text, ok", [
    ('{"winner": "A", "factual_errors": "none", "reason": "x"}', True),
    ('<think>hm</think>```json\n{"winner": "tie", "factual_errors": "both"}\n```', True),
    ('{"winner": "C", "factual_errors": "none"}', False),
    ("not json", False),
    (None, False),
])
def test_parse_verdict(text, ok):
    assert (jg.parse_verdict(text) is not None) is ok


@pytest.mark.parametrize("v_xy, v_yx, winner, consistent, x_err, y_err", [
    (verdict("A"), verdict("B"), "x", True, False, False),  # X wins in both orders
    (verdict("B"), verdict("A"), "y", True, False, False),
    (verdict("A"), verdict("A"), "tie", False, False, False),  # position bias: always A -> tie, inconsistent
    (verdict("tie"), verdict("tie"), "tie", True, False, False),
    (verdict("A", "A"), verdict("B", "B"), "x", True, True, False),  # X flagged in both orders
    (verdict("A", "A"), verdict("B", "none"), "x", True, False, False),  # flagged once: not charged
    (verdict("A", "both"), verdict("B", "A"), "x", True, False, True),
])
def test_combine_maps_both_orders(v_xy, v_yx, winner, consistent, x_err, y_err):
    o = jg.combine(item(0), v_xy, v_yx)
    assert (o.winner, o.consistent, o.x_error, o.y_error, o.parsed) == (winner, consistent, x_err, y_err, True)


def test_combine_unparsed_is_a_tie_and_flagged():
    o = jg.combine(item(0), None, verdict())
    assert (o.winner, o.parsed) == ("tie", False)


def fake_judge(prefer: str = "x", position_bias: bool = False):
    """Answers JSON; prefers the answer containing `prefer`'s text, or always A."""
    calls = []

    def generate(batch, **kw):
        calls.append(kw)
        out = []
        for m in batch:
            text = m[0]["content"]
            a = text.split("Answer A:")[1].split("Answer B:")[0]
            if position_bias:
                w = "A"
            else:
                w = "A" if f"{prefer.upper()} answer" in a else "B"
            out.append(SimpleNamespace(text=json.dumps({"winner": w, "factual_errors": "none", "reason": "because"})))
        return out

    generate.calls = calls
    return generate


def test_judge_pairs_one_batch_both_orders_with_context():
    gen = fake_judge("x")
    outcomes = jg.judge_pairs([item(i) for i in range(3)], gen, jg.load_rubric())
    assert [o.winner for o in outcomes] == ["x"] * 3 and all(o.consistent for o in outcomes)
    (kw,) = gen.calls  # a single batch: cascade attention on the shared journal prefix
    assert kw["with_context"] is True and kw["temperature"] == 0.0 and kw["json_schema"]["title"] == "Verdict"


def test_position_bias_shows_as_inconsistency():
    outcomes = jg.judge_pairs([item(i) for i in range(4)], fake_judge(position_bias=True), jg.load_rubric())
    s = jg.summarize(outcomes)["all"]
    assert s["tie"] == 1.0 and s["inconsistency"] == 1.0


def test_failed_judge_calls_are_counted():
    def gen(batch, **kw):
        return [None] * len(batch)

    s = jg.summarize(jg.judge_pairs([item(0)], gen, jg.load_rubric()))["all"]
    assert s["parse_failures"] == 1 and s["tie"] == 1.0


def outcome(i, winner, origin="chat", kind="recall", x_error=False, y_error=False):
    return jg.Outcome(task_id=f"t{i}", origin=origin, kind=kind, winner=winner, consistent=True,
                      x_error=x_error, y_error=y_error, parsed=True)


def test_summarize_strata_rates_and_non_inferiority():
    outs = ([outcome(i, "x") for i in range(3)] + [outcome(3, "tie"), outcome(4, "y", origin="human")]
            + [outcome(5, "y", origin="human", kind="self_synthesis", x_error=True)])
    s = jg.summarize(outs, seed=1)
    assert set(s) == {"all", "origin:chat", "origin:human", "kind:recall", "kind:self_synthesis"}
    chat, human = s["origin:chat"], s["origin:human"]
    assert (chat["n"], chat["win"], chat["tie"], chat["non_loss"], chat["non_inferior"]) == (4, 0.75, 0.25, 1.0, True)
    assert human["non_loss"] == 0.0 and human["non_inferior"] is False
    lo, hi = s["all"]["non_loss_ci"]
    assert 0.0 <= lo <= s["all"]["non_loss"] <= hi <= 1.0
    assert s["all"]["x_error_rate"] == pytest.approx(1 / 6) and s["all"]["non_inferior"] is False  # more X errors


def test_bootstrap_ci_is_deterministic_and_degenerate_on_constants():
    assert jg.bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert jg.bootstrap_ci([0.0, 1.0] * 10, seed=3) == jg.bootstrap_ci([0.0, 1.0] * 10, seed=3)


def test_calibration_agreement_with_human_annotations():
    annotations = [
        {"task_id": "t0", "kind": "recall", "question": "Q0?", "journal_answer": "J", "no_journal_answer": "N",
         "preference": 2},
        {"task_id": "t1", "kind": "recall", "question": "Q1?", "journal_answer": "J", "no_journal_answer": "N",
         "preference": -1},
        {"task_id": "t2", "kind": "recall", "question": "Q2?", "journal_answer": "J", "no_journal_answer": "N",
         "preference": 0},
        {"task_id": "t3", "kind": "recall", "question": "Q3?", "journal_answer": "J", "no_journal_answer": "N",
         "preference": None},  # not annotated: skipped
    ]
    items = jg.calibration_items(annotations)
    assert (items[0].x, items[0].y) == ("J", "N")
    outs = [outcome(0, "x"), outcome(1, "x"), outcome(2, "tie"), outcome(3, "x")]
    a = jg.agreement(outs, annotations)
    assert a["n"] == 3 and a["accuracy"] == pytest.approx(2 / 3) and a["kappa"] is not None
    assert jg.agreement(outs, [])["n"] == 0


def test_log_comparison_writes_table_and_strata_summary():
    logged, summary = {}, {}
    run = SimpleNamespace(log=logged.update, run=SimpleNamespace(summary=summary))
    outs = [outcome(0, "x"), outcome(1, "y", origin="human")]
    jg.log_comparison(run, outs, jg.summarize(outs))
    assert "judge/items" in logged and summary["judge/origin:chat/non_loss"] == 1.0
    assert summary["judge/all/n"] == 2


def test_stratified_subset_even_across_sources_and_kinds():
    items = ([item(i, "psych_test", ["a", "b"][i % 2]) for i in range(100)] + [item(100 + i, "chat") for i in range(30)]
             + [item(200 + i, "human") for i in range(7)])
    sub = jg.stratified_subset(items, 60, seed=0)
    from collections import Counter

    c = Counter(i.origin for i in sub)
    assert c == {"human": 7, "chat": 27, "psych_test": 26}  # human gives all 7; the other 53 split evenly
    psych = Counter(i.kind for i in sub if i.origin == "psych_test")
    assert abs(psych["a"] - psych["b"]) <= 1
    assert [i.task_id for i in sub] == [i.task_id for i in jg.stratified_subset(list(reversed(items)), 60, seed=0)]


def test_citation_summary():
    pages = {"A:p.1", "A:p.2"}
    answers = ["x [A:p.1]", "y [A:p.9]", "z", "w [<A:p.2>]"]
    s = jg.citation_summary(answers, ["chat", "chat", "human", "human"], pages)
    assert s["all"] == {"n": 4, "citation_rate": 0.75, "citation_validity": 0.5}
    assert s["origin:chat"]["citation_validity"] == 0.5 and s["origin:human"]["citation_rate"] == 0.5


def test_side_by_side_rows_and_prefixed_logging():
    tasks = [SimpleNamespace(task_id="t0", origin="chat", kind="recall", text="Where did I go?")]
    answers = {"teacher": {"t0": {"text": "Tahoe [A:p.1]", "finish_reason": "stop", "n_tokens": 5}},
               "cartridge": {"t0": {"text": "Tahoe [A:p.9]", "finish_reason": "length", "n_tokens": 7}},
               "base": {"t0": {"text": "I don't know.", "finish_reason": "stop", "n_tokens": 4}}}
    cart = jg.combine(item(0), verdict("tie", "A", "close"), verdict("tie", "B", "close too"))
    base = jg.combine(item(0), verdict("B", "A", "base guesses"), verdict("A", "B", "teacher grounded"))
    rows = jg.side_by_side_rows(tasks, answers, {"cartridge": {"t0": cart}, "base": {"t0": base}}, {"A:p.1"})
    (r,) = rows
    assert (r["teacher_citation_valid"], r["cartridge_citation_valid"], r["base_n_citations"]) == (True, False, 0)
    assert r["cartridge_truncated"] is True and r["question"] == "Where did I go?"
    assert (r["cartridge_vs_teacher_winner"], r["cartridge_vs_teacher_x_error"]) == ("tie", True)
    assert (r["base_vs_teacher_winner"], r["base_first_reason"], r["base_swapped_reason"]) == (
        "y", "base guesses", "teacher grounded")

    logged, summary = {}, {}
    run = SimpleNamespace(log=logged.update, run=SimpleNamespace(summary=summary))
    jg.log_comparison(run, [cart], jg.summarize([cart] * 3), prefix="judge/base_vs_teacher")
    assert "judge/base_vs_teacher/items" in logged and "judge/base_vs_teacher/all/non_loss_ci_low" in summary


def test_eval_judge_streams_cartridge_answers_and_resumes():
    from cartridges.cli import _collect, _unanswered

    tasks = [SimpleNamespace(text=f"Q{i}?") for i in range(3)]
    ids = ["t0", "t1", "t2"]
    saved = {"t0": {"task_id": "t0", "cartridge": {"text": "old", "finish_reason": "stop", "n_tokens": 1}}}
    todo = _unanswered(ids, tasks, saved, "cartridge")
    assert todo == [{"task_id": "t1", "text": "Q1?"}, {"task_id": "t2", "text": "Q2?"}]  # only unfinished tasks

    writes = []

    def stream():
        for t in todo:
            yield {"task_id": t["task_id"], "text": "A", "finish_reason": "stop", "n_tokens": 2, "extra": 1}
            if t["task_id"] == "t1":
                raise ConnectionError("dropped")

    with pytest.raises(ConnectionError):
        _collect(stream(), saved, "cartridge", lambda: writes.append(set(saved)))
    assert writes == [{"t0", "t1"}]  # t1 written before the failure
    assert saved["t1"]["cartridge"] == {"text": "A", "finish_reason": "stop", "n_tokens": 2}
    assert _unanswered(ids, tasks, saved, "cartridge") == [{"task_id": "t2", "text": "Q2?"}]  # resume asks for t2 only
