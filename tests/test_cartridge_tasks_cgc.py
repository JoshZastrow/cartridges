"""Context gap closed (tasks/cgc.py): metric, segmentation and report, no GPU."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from cartridges.cartridge.tasks import cgc


def st(i, origin="chat", kind="recall", n=10, base=20.0, cart=15.0, teacher=10.0):
    return cgc.ScoredTask(task_id=f"t{i}", origin=origin, kind=kind, n_tokens=n, nll_base=base, nll_cart=cart,
                          nll_teacher=teacher)


def test_to_scored_sums_logprobs_on_the_common_prefix():
    t = cgc.to_scored("t", "chat", "recall", [-1.0, -1.0, -1.0], [-0.5, -0.5], [-0.1, -0.1, -0.1])
    assert (t.n_tokens, t.nll_base, t.nll_cart, t.nll_teacher) == (2, 2.0, 1.0, pytest.approx(0.2))


@pytest.mark.parametrize("cart, want", [(10.0, 1.0), (20.0, 0.0), (15.0, 0.5), (25.0, -0.5)])
def test_cgc_endpoints(cart, want):
    assert cgc.cgc_stats([st(i, cart=cart) for i in range(3)])["cgc"] == pytest.approx(want)


def test_cgc_is_token_weighted_not_task_averaged():
    # one long task closes the whole gap, one short task closes none
    tasks = [st(0, n=90, base=90, cart=45, teacher=45), st(1, n=10, base=10, cart=10, teacher=5)]
    assert cgc.cgc_stats(tasks)["cgc"] == pytest.approx((100 - 55) / (100 - 50))


def test_no_cgc_when_teacher_does_not_beat_base():
    s = cgc.cgc_stats([st(0, base=10, teacher=12)])
    assert s["cgc"] is None and s["cgc_ci"] is None


def test_perplexities_and_ci():
    s = cgc.cgc_stats([st(i, cart=12 + i % 3) for i in range(20)], seed=1)
    assert s["ppl_base"] == pytest.approx(math.exp(2.0)) and s["ppl_teacher"] == pytest.approx(math.exp(1.0))
    lo, hi = s["cgc_ci"]
    assert lo <= s["cgc"] <= hi and s["small"] is False
    assert s == cgc.cgc_stats([st(i, cart=12 + i % 3) for i in range(20)], seed=1)  # deterministic


def test_chosen_logprobs_from_topk():
    assert cgc.chosen_logprobs([5, 7], [[5, 1], [2, 3]], [[-0.1, -2.0], [-0.5, -1.0]]) == [-0.1, None]


def test_segment_source_first_then_kind_and_small_flags():
    tasks = ([st(i, "chat", "recall") for i in range(12)] + [st(20, "human_template", "self_synthesis"),
             st(21, "human", "recall"), st(22, "synthetic", "recall"), st(23, "psych_test", "framework_lens")])
    seg = cgc.segment(tasks)
    assert list(seg) == ["all", "chat", "synthetic", "psych_test", "human"]
    assert set(seg["human"]) == {"all", "kind:recall", "kind:self_synthesis"} and seg["human"]["all"]["n"] == 2
    assert seg["chat"]["all"]["small"] is False and seg["human"]["all"]["small"] is True


def test_report_markdown_and_flat_summary():
    seg = cgc.segment([st(i, "chat") for i in range(3)] + [st(9, "synthetic", "build_artifact", cart=20)])
    md = cgc.report_markdown(seg, {"cartridge": "cartridge-abc:v3", "judge": "none"})
    assert "**Headline CGC:" in md and "## chat by kind" in md and "⚠ small" in md and "cartridge-abc:v3" in md
    flat = cgc.flat_summary(seg)
    assert flat["eval/cgc"] == pytest.approx(seg["all"]["cgc"]) and flat["eval/synthetic/build_artifact/cgc"] == 0.0
    assert "eval/chat/cgc" in flat and "eval/chat/recall/n" in flat


def test_log_report_writes_artifact_table_and_summary(tmp_path):
    logged, summary, artifacts = {}, {}, []
    run = SimpleNamespace(log=logged.update, run=SimpleNamespace(summary=summary),
                          log_artifact=lambda path, type, name: artifacts.append((path, type, name)))
    tasks = [st(i) for i in range(3)]
    path = cgc.log_report(run, tasks, cgc.segment(tasks), {"k": "v"}, tmp_path / "cgc.md")
    assert path.read_text().startswith("# Cartridge eval") and artifacts == [(path, "eval-report", "cgc-report")]
    assert "eval/cgc_tasks" in logged and summary["eval/cgc"] == pytest.approx(0.5)
    by_kind = logged["eval/by_kind"]
    assert logged["eval/by_source"].columns[:3] == ["source", "n", "tokens"] and by_kind.columns[:3] == ["source", "kind", "n"]
    assert [r[:2] for r in by_kind.data] == [["chat", "recall"]] and logged["eval/by_source"].data[0][0] == "all"


def test_assemble_from_scorer_outputs():
    tasks = [SimpleNamespace(task_id=f"t{i}", origin="chat", kind="recall") for i in range(3)]
    rows = [{"task_id": f"t{i}", "prompt_ids": [1], "completion_ids": [5, 6]} for i in range(3)]
    trainer_out = {"base": [{"id": f"l{i}", "task_id": f"t{i}", "logprobs": [-1.0, -1.0]} for i in range(3)],
                   "cartridge": [{"id": f"l{i}", "task_id": f"t{i}", "logprobs": [-0.55, -0.55]} for i in range(2)]}
    base, cart = cgc.by_task(trainer_out["base"]), cgc.by_task(trainer_out["cartridge"])  # t2: no cartridge score
    teacher = cgc.keyed(rows, [[-0.1, -0.1]] * 3)
    out = cgc.assemble(tasks, base, cart, teacher)
    assert [t.task_id for t in out] == ["t0", "t1"] and cgc.cgc_stats(out)["cgc"] == pytest.approx(0.5)


def test_crosscheck_hf_teacher_against_label_topk():
    rows = [{"completion_ids": [5, 7, 9], "topk_ids": [[5, 1], [7, 2], [1, 2]],
             "topk_logprobs": [[-0.1, -3.0], [-0.2, -3.0], [-0.5, -0.9]]}]
    c = cgc.crosscheck(rows, [[-0.15, -0.2, -4.0]])
    assert c["tokens"] == 3 and c["in_label_topk"] == pytest.approx(2 / 3)
    assert c["mean_abs_diff"] == pytest.approx(0.025)


class FakeScorer:
    def __init__(self, fail_on: int | None = None):
        self.calls, self.cancelled, self.fail_on = [], [], fail_on
        self.info = SimpleNamespace(remote=lambda: {"container_id": "ta-1"})
        self.score_rows = SimpleNamespace(spawn=self.spawn)

    def spawn(self, ids, with_context):
        n = len(self.calls)
        self.calls.append((len(ids), with_context))

        def get(timeout):
            if n == self.fail_on:
                raise KeyboardInterrupt
            return [[-0.5] * len(r["completion_ids"]) for r in ids]

        return SimpleNamespace(get=get, cancel=lambda: self.cancelled.append(n))


def test_scorer_session_chunks_and_stops_only_its_container():
    scorer, stopped = FakeScorer(), []
    rows = [{"task_id": f"t{i}", "prompt_ids": [1], "completion_ids": [5, 6]} for i in range(5)]
    with cgc.scorer_session("m", chunk=2, make=lambda name, model: scorer, stop=lambda container_ids: stopped.append(container_ids)) as score:
        out = score(rows, True)
    assert len(out) == 5 and [c[0] for c in scorer.calls] == [2, 2, 1] and stopped == [["ta-1"]]


def test_scorer_session_cancels_on_interrupt_and_still_stops():
    scorer, stopped = FakeScorer(fail_on=1), []
    rows = [{"task_id": f"t{i}", "prompt_ids": [1], "completion_ids": [5]} for i in range(4)]
    with pytest.raises(KeyboardInterrupt):
        with cgc.scorer_session("m", chunk=2, make=lambda n, m: scorer, stop=lambda container_ids: stopped.append(container_ids)) as score:
            score(rows, False)
    assert scorer.cancelled == [1] and stopped == [["ta-1"]]


def test_final_cgc_runs_trainer_then_teacher_with_context_only():
    import contextlib

    events = []
    labels = [SimpleNamespace(task_id=f"t{i}", label_id=f"l{i}", status="validated", targets_uri="u",
                              model_dump=lambda mode, i=i: {"label_id": f"l{i}", "task_id": f"t{i}"}) for i in range(3)]
    tasks = [SimpleNamespace(task_id=f"t{i}", origin="chat", kind="recall") for i in range(3)]

    def read_labeled(valid):
        return [SimpleNamespace(prompt_ids=[1], completion_ids=[5, 6], topk_ids=[[5], [6]],
                                topk_logprobs=[[-0.1], [-0.1]]) for _ in valid]

    def score_cart(cartridge, label_dicts, model_id):
        events.append("trainer")
        return {"base": [{"task_id": d["task_id"], "logprobs": [-1.0, -1.0]} for d in label_dicts],
                "cartridge": [{"task_id": d["task_id"], "logprobs": [-0.55, -0.55]} for d in label_dicts]}

    @contextlib.contextmanager
    def session(model_id):
        events.append("scorer-open")

        def score_ref(rows, with_context):
            events.append(("teacher", with_context))
            return [[-0.1, -0.1] for _ in rows]

        yield score_ref
        events.append("scorer-stop")

    scored, seg, check = cgc.final_cgc("cartridge-x:latest", "m", labels, tasks, read_labeled=read_labeled,
                                       score_cart=score_cart, session=session)
    assert events == ["trainer", "scorer-open", ("teacher", True), "scorer-stop"]  # no base pass on the scorer
    assert seg["all"]["cgc"] == pytest.approx(0.5) and check["in_label_topk"] == 1.0


def test_token_types_in_context():
    pieces = ["You", " met", " Allie", " on", " 11", "/", "20", " [", "Auto", " biography", ":p", ".", "2", "]", ".",
              " She", " left", "."]
    t = cgc.token_types(pieces)
    assert t[:4] == ["other", "other", "name", "other"]
    assert t[4:7] == ["number", "number", "number"]
    assert set(t[7:14]) == {"citation"}
    assert t[15] == "other"  # "She" starts a sentence: not a name


def test_top1_and_type_diagnostics_flow_into_stats_report_and_tables():
    t = cgc.to_scored("t0", "chat", "recall", [-2.0, -2.0, -1.0, -1.0], [-1.0, -2.0, -0.5, -1.0],
                      [-0.5, -0.5, -0.5, -0.5], completion_ids=[1, 2, 3, 4], base_argmax=[1, 9, 9, 9],
                      cart_argmax=[1, 2, 3, 9], types=["name", "name", "number", "other"])
    assert (t.n_top1, t.top1_base, t.top1_cart) == (4, 1, 3)
    assert t.by_type["name"] == (2, 4.0, 3.0, 1.0)
    st = cgc.cgc_stats([t])
    assert st["top1_cart"] == 0.75 and st["top1_base"] == 0.25
    assert st["cgc_name"] == pytest.approx((4 - 3) / (4 - 1)) and st["nll_cart_number"] == 0.5
    seg = cgc.segment([t])
    md = cgc.report_markdown(seg, {})
    assert "top-1 agreement" in md and "| all | name | 2 |" in md
    assert cgc.flat_summary(seg)["eval/chat/recall/top1_cart"] == 0.75


def test_assemble_passes_argmax_and_types():
    tasks = [SimpleNamespace(task_id="t0", origin="chat", kind="recall")]
    base = {"t0": {"logprobs": [-1.0, -1.0], "argmax": [5, 0]}}
    cart = {"t0": {"logprobs": [-0.5, -0.5], "argmax": [5, 6]}}
    out = cgc.assemble(tasks, base, cart, {"t0": [-0.1, -0.1]}, {"t0": {"completion_ids": [5, 6]}},
                       {"t0": ["name", "other"]})
    assert (out[0].top1_cart, out[0].top1_base) == (2, 1) and set(out[0].by_type) == {"name", "other"}


def test_token_types_split_name_and_date_separators():
    pieces = ["We", " met", " All", "ie", " on", " 11", "/", "20", " in", " Tahoe", "."]
    assert cgc.token_types(pieces) == ["other", "other", "name", "name", "other", "number", "number", "number",
                                       "other", "name", "other"]


def test_empty_eval_fails_loudly_before_gpu_and_before_logging(tmp_path):
    calls = []

    def score_cart(*a):
        calls.append("gpu")
        return {"base": [], "cartridge": []}

    with pytest.raises(ValueError, match="nothing to score"):
        cgc.final_cgc("c", "m", [], [], read_labeled=lambda v: [], score_cart=score_cart, session=None)
    assert calls == []  # no GPU call for an empty label set
    with pytest.raises(ValueError, match="empty CGC eval"):
        cgc.log_report(SimpleNamespace(), [], cgc.segment([]), {}, tmp_path / "x.md")


def test_final_cgc_raises_when_nothing_joins():
    import contextlib

    labels = [SimpleNamespace(task_id="t0", label_id="l0", status="validated", targets_uri="u",
                              model_dump=lambda mode: {"label_id": "l0", "task_id": "t0"})]
    tasks = [SimpleNamespace(task_id="t0", origin="chat", kind="recall")]

    @contextlib.contextmanager
    def session(model_id):
        yield lambda rows, with_context: [[-0.1] for _ in rows]

    with pytest.raises(ValueError, match="no task scored by all three"):
        cgc.final_cgc("c", "m", labels, tasks,
                      read_labeled=lambda v: [SimpleNamespace(prompt_ids=[1], completion_ids=[5], topk_ids=[[5]],
                                                              topk_logprobs=[[-0.1]])],
                      score_cart=lambda *a: {"base": [], "cartridge": []}, session=session)


def test_cli_id_list_accepts_fires_tuple():
    from cartridges.cli import _id_list

    assert _id_list(("x8m2098y", "qdmsgn67")) == ["x8m2098y", "qdmsgn67"]
    assert _id_list("x8m2098y,qdmsgn67") == ["x8m2098y", "qdmsgn67"] == _id_list(" x8m2098y , qdmsgn67 ")
