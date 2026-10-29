"""shared.judge with a fake judge model: no GPU."""

from types import SimpleNamespace

import pytest

from shared import judge as jg


@pytest.fixture
def rubric(tmp_path):
    path = tmp_path / "voice-v1.md"
    path.write_text("Context: {context}\nQ: {question}\nAnswer A: {answer_a}\nAnswer B: {answer_b}\nJSON.")
    return jg.load_rubric(path)


def test_load_rubric_from_path(rubric):
    assert rubric.version == "voice-v1" and len(rubric.hash) == 12
    msg = jg.judge_messages(rubric, "Why?", "first", "second", context="passage")[0]["content"]
    assert "passage" in msg and msg.index("first") < msg.index("second")


@pytest.mark.parametrize("text, ok", [
    ('{"winner": "A", "factual_errors": "none", "reason": "x"}', True),
    ('<think>hm</think>```json\n{"winner": "tie", "factual_errors": "both"}\n```', True),
    ('{"winner": "C", "factual_errors": "none"}', False),
    ("not json", False),
    (None, False),
])
def test_parse_verdict(text, ok):
    assert (jg.parse_verdict(text) is not None) is ok


def test_combine_needs_both_orders_to_agree():
    item = jg.Item(task_id="t", origin="o", kind="k", question="q", x="x", y="y")
    v = lambda w, e="none": jg.Verdict(winner=w, factual_errors=e)  # noqa: E731
    assert jg.combine(item, v("A"), v("B")).winner == "x"
    assert jg.combine(item, v("A"), v("A")).winner == "tie"  # position bias
    o = jg.combine(item, v("A", "A"), v("B", "none"))
    assert (o.x_error, o.y_error) == (False, False)  # flagged once: not charged


def test_judge_end_to_end_with_fake_model(rubric):
    calls = []

    def generate(batch, **kw):
        calls.append(kw)
        out = []
        for m in batch:
            a = m[0]["content"].split("Answer A:")[1].split("Answer B:")[0]
            out.append(SimpleNamespace(text=f'{{"winner": "{"A" if "good" in a else "B"}", "factual_errors": "none"}}'))
        return out

    tasks = [jg.Task(task_id=f"t{i}", origin="story", kind="recall", question=f"Q{i}", context="p") for i in range(4)]
    outcomes, summary, items = jg.judge(["good"] * 3 + ["bad"], tasks, ["ok"] * 3 + ["good"], generate, rubric)
    assert [o.winner for o in outcomes] == ["x", "x", "x", "y"]
    assert summary["all"]["win"] == 0.75 and summary["origin:story"]["n"] == 4
    assert items[0].context == "p" and calls[0]["with_context"] is False and calls[0]["temperature"] == 0.0
