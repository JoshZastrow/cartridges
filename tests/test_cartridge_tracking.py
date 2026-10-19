import datetime
import importlib.util
import subprocess
from types import SimpleNamespace

import pytest

from cartridges.cartridge import tracking
from cartridges.training.perf_metrics import GenerationTiming


@pytest.fixture(autouse=True)
def wandb_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_group_name_format():
    assert tracking.group_name("Smoke Test!", datetime.date(2026, 10, 13)) == "exp-20261013-smoke-test"
    with pytest.raises(ValueError):
        tracking.group_name("!!!")


def test_config_hash_ignores_key_order_and_changes_with_values():
    assert tracking.config_hash({"a": 1, "b": 2}) == tracking.config_hash({"b": 2, "a": 1})
    assert tracking.config_hash({"a": 1}) != tracking.config_hash({"a": 2})
    assert len(tracking.config_hash(None)) == 8


def test_run_name_follows_contract():
    h = tracking.config_hash({"lr": 1e-3})
    assert tracking.run_name("distill", "Qwen/Qwen3.5-4B", 512, {"lr": 1e-3}) == f"distill-qwen3.5-4b-512tok-{h}"
    assert tracking.run_name("grpo", "Qwen/Qwen3.8-27B", 2048, None, role="gen").startswith("grpo-gen-qwen3.8-27b-")
    named = tracking.run_name("eval", "Qwen/Qwen3.5-4B", 0, None, role="teacher", group="exp-20261014-reflective-qs")
    assert named == f"reflective-qs-eval-teacher-qwen3.5-4b-0tok-{tracking.config_hash(None)}"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("A100-80GB", ("A100-80GB", 1)),
        ("NVIDIA A100-SXM4-80GB", ("A100-80GB", 1)),
        ("NVIDIA A100-SXM4-40GB", ("A100-40GB", 1)),
        ("h100:2", ("H100", 2)),
        ("NVIDIA H100 80GB HBM3", ("H100", 1)),
        ("H200", ("H200", 1)),
        ("NVIDIA H200", ("H200", 1)),
        ("L4", ("L4", 1)),
        ("NVIDIA L4", ("L4", 1)),
        ("NVIDIA L40S", ("L40S", 1)),
        ("A10G", ("A10G", 1)),
        ("NVIDIA A10G", ("A10G", 1)),
        (None, ("CPU", 1)),
    ],
)
def test_normalize_gpu(name, expected):
    assert tracking.normalize_gpu(name) == expected


def test_normalize_gpu_unknown_raises():
    with pytest.raises(ValueError):
        tracking.normalize_gpu("TPU-v5")


def test_git_info_falls_back_to_env_outside_a_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "_REPO_ROOT", tmp_path)  # not a git repo
    monkeypatch.setenv("CARTRIDGES_GIT_SHA", "abc123")
    monkeypatch.setenv("CARTRIDGES_GIT_DIRTY", "1")
    assert tracking.git_info() == {"git_sha": "abc123", "git_dirty": True}


def test_modal_env_round_trips_git_info():
    env = tracking.modal_env()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert env["CARTRIDGES_GIT_SHA"] == sha
    assert env["CARTRIDGES_GIT_DIRTY"] in {"0", "1"}


def test_modal_info_empty_locally(monkeypatch):
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    assert tracking.modal_info() == {}


def test_init_run_rejects_bad_job_type_and_group():
    group = tracking.group_name("t")
    with pytest.raises(ValueError):
        tracking.init_run("train", "Qwen/Qwen3.5-4B", group)
    with pytest.raises(ValueError):
        tracking.init_run("distill", "Qwen/Qwen3.5-4B", "my-group")


def test_run_tracks_cost_from_gpu_rate_and_wall_clock():
    clock = FakeClock()
    group = tracking.group_name("t")
    with tracking.init_run(
        "distill", "Qwen/Qwen3.5-4B", group, {"lr": 1e-3}, p=512, gpu="H100:2", clock=clock
    ) as run:
        assert "-distill-qwen3.5-4b-512tok-" in run.name
        assert run.run.group == group
        meta = run.run.config["_meta"]
        assert meta["gpu"] == "H100" and meta["n_gpus"] == 2 and meta["p"] == 512
        assert "git_sha" in meta and "git_dirty" in meta
        clock.t = 1800.0  # half an hour on two H100s
        run.log({"train/kl": 0.5})
        assert run.run.summary["cost/gpu_hours"] == pytest.approx(1.0)
        assert run.run.summary["cost/usd"] == pytest.approx(3.95)
    assert run.run.summary["cost/wall_clock_s"] == pytest.approx(1800.0)


def test_rate_override():
    clock = FakeClock()
    run = tracking.init_run("eval", "Qwen/Qwen3.5-4B", tracking.group_name("t"), rate_usd_per_hour=10.0, clock=clock)
    clock.t = 360.0
    assert run.cost()["cost/usd"] == pytest.approx(1.0)
    run.finish()


def test_timer_and_generation_metrics():
    with tracking.init_run("teacher", "Qwen/Qwen3.5-4B", tracking.group_name("t")) as run:
        with run.timer("prefill", tokens=1000) as t:
            sum(range(10_000))
        assert t.seconds > 0
        assert run.run.summary["time/prefill_s"] == pytest.approx(t.seconds)
        assert run.run.summary["throughput/prefill_tok_s"] == pytest.approx(1000 / t.seconds)

        run.log_generation(GenerationTiming(t_start=0.0, t_first=0.25, t_end=1.25, n_output_tokens=11))
        assert run.run.summary["gen/ttft_s"] == pytest.approx(0.25)
        assert run.run.summary["gen/decode_tok_s"] == pytest.approx(10.0)
        assert run.log_gpu_memory() == {}  # no CUDA on the test machine


def test_artifacts_are_noops_when_disabled(tmp_path):
    f = tmp_path / "cartridge.safetensors"
    f.write_bytes(b"x")
    with tracking.init_run("distill", "Qwen/Qwen3.5-4B", tracking.group_name("t")) as run:
        assert run.log_artifact(f, type="cartridge") is None
        assert run.use_artifact(f) == f
        with pytest.raises(FileNotFoundError):
            run.use_artifact("cartridge-abc:latest")
    assert tracking.fetch_artifact(f) == f


def test_qualify_artifact_ref():
    assert tracking._qualify("cartridge-abc") == f"{tracking.ENTITY}/{tracking.PROJECT}/cartridge-abc:latest"
    assert tracking._qualify("e/p/cartridge-abc:v3") == "e/p/cartridge-abc:v3"


def _fake_run(name, job_type, usd, group="exp-20261013-t", system=None, start="2026-10-13T00:00:00Z", **summary):
    config = {"_meta": {"gpu": "A100-80GB"}}
    if system:
        config["system"] = system
    return SimpleNamespace(
        name=name, job_type=job_type, state="finished", group=group, url=f"https://wandb.ai/{name}",
        summary={"cost/usd": usd, **summary}, config=config, created_at=start,
    )


class FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, path, filters):
        assert path == f"{tracking.ENTITY}/{tracking.PROJECT}"
        return [r for r in self._runs if all(getattr(r, k) == v for k, v in filters.items())]


RUNS = [
    _fake_run("distill-a", "distill", 1.5),
    _fake_run("grpo-a-gen", "grpo", 2.0),
    _fake_run("grpo-a-train", "grpo", 2.5),
    _fake_run("eval-base", "eval", 0.25, system="base", **{"eval/accuracy": 0.2}),
    _fake_run("eval-cart", "eval", 0.25, system="cartridge", **{"eval/accuracy": 0.8, "eval/token_f1": 0.6}),
    _fake_run("eval-ask", "eval", 0.1, **{"eval/accuracy": 1.0}),  # cartridge-ask: no config.system
    _fake_run("other", "distill", 40.0, group="exp-20261012-old"),
]


def test_group_costs_rolls_up_by_job_type():
    costs = tracking.group_costs("exp-20261013-t", api=FakeApi(RUNS))
    assert costs["total_usd"] == pytest.approx(6.6)
    assert costs["by_job_type"] == pytest.approx({"distill": 1.5, "grpo": 4.5, "eval": 0.6})
    assert len(costs["runs"]) == 6


def test_run_without_cost_summary_falls_back_to_runtime():
    # died in a Modal @enter before its first log(): no cost/* keys, only W&B's _runtime
    r = SimpleNamespace(summary={"_runtime": 1800.0}, config={"_meta": {"n_gpus": 2, "rate_usd_per_hour": 0.8}})
    assert tracking._run_cost(r) == pytest.approx((0.8, 1.0))
    assert tracking._run_cost(SimpleNamespace(summary={}, config={})) == (0.0, 0.0)


def test_budget_status_across_groups():
    status = tracking.budget_status(api=FakeApi(RUNS))
    assert status["spent_usd"] == pytest.approx(46.6)
    assert status["remaining_usd"] == pytest.approx(3.4)
    assert status["by_group"]["exp-20261012-old"] == 40.0


def test_group_summary_markdown_has_stages_and_eval_comparison():
    md = tracking.group_summary_markdown("exp-20261013-t", api=FakeApi(RUNS))
    assert "$6.60 direct + $0.00 shared Teacher service" in md
    assert "| eval-ask | 1 |" not in md  # ask runs are listed with cost but not in the eval table
    assert md.index("| distill |") < md.index("| grpo |") < md.index("| eval |")
    assert "| cartridge | 0.8 | 0.6 | - | - | - |" in md
    assert "other" not in md


@pytest.mark.skipif(importlib.util.find_spec("wandb_workspaces") is None, reason="needs wandb-workspaces")
def test_create_group_report_builds(monkeypatch):
    import wandb_workspaces.reports.v2 as wr

    saved = []
    monkeypatch.setattr(wr.Report, "save", lambda self: saved.append(self))
    monkeypatch.setattr(wr.Report, "url", "https://wandb.ai/report", raising=False)
    assert tracking.create_group_report("exp-20261013-t") == "https://wandb.ai/report"
    assert saved and saved[0].title.endswith("exp-20261013-t")


def test_check_budget_guards_launches():
    api = FakeApi(RUNS)  # 46.6 spent
    assert tracking.check_budget(3.0, api=api)["spent_usd"] == pytest.approx(46.6)
    with pytest.raises(tracking.BudgetExceeded):
        tracking.check_budget(4.0, api=api)


@pytest.mark.parametrize("project_usd,alerts", [(39.0, 0), (40.0, 1)])
def test_finish_alerts_at_80_percent_of_budget(monkeypatch, project_usd, alerts):
    sent = []
    fake = SimpleNamespace(
        id="r1", name="distill-x", group="exp-20261013-t", summary={},
        settings=SimpleNamespace(mode="online"),
        alert=lambda **kw: sent.append(kw), finish=lambda exit_code: None,
    )
    monkeypatch.setattr(tracking, "group_costs", lambda group: {"total_usd": 1.0})
    monkeypatch.setattr(tracking, "budget_status", lambda: {"spent_usd": project_usd})
    tracking.TrackedRun(run=fake, gpu="CPU", n_gpus=1, rate_usd_per_hour=0.05).finish()
    assert len(sent) == alerts
    assert fake.summary["cost/project_usd"] == project_usd


def test_use_artifact_lineage_only():
    used = []
    fake = SimpleNamespace(settings=SimpleNamespace(mode="online"), use_artifact=lambda ref, type: used.append((ref, type)))
    run = tracking.TrackedRun(run=fake, gpu="CPU", n_gpus=1, rate_usd_per_hour=0.05)
    assert run.use_artifact("selfstudy-abc", type="dataset", download=False) is None
    assert used == [(f"{tracking.ENTITY}/{tracking.PROJECT}/selfstudy-abc:latest", "dataset")]
    with tracking.init_run("distill", "Qwen/Qwen3.5-4B", tracking.group_name("t")) as disabled:
        assert disabled.use_artifact("selfstudy-abc", download=False) is None


def test_jsonl_table_has_requested_columns(tmp_path):
    f = tmp_path / "qa.jsonl"
    f.write_text('{"question": "q1", "citations": ["J:p.1"]}\n{"question": "q2"}\n')
    table = tracking.jsonl_table(f, ["question", "citations"])
    assert table.columns == ["question", "citations"]
    assert table.data == [["q1", '["J:p.1"]'], ["q2", None]]


def test_teacher_service_cost_split_by_time_window():
    # One service container, 00:00-01:00, $4. Experiment A runs 00:00-00:30, B 00:15-00:45.
    # 00:00-15 A alone ($1), 15-30 A and B share ($0.5 each), 30-45 B alone ($1), 45-60 idle ($1).
    runs = [
        _fake_run("teacher-gen", "teacher", 4.0, group="exp-20261015-teacher-service",
                  start="2026-10-15T00:00:00Z", **{"cost/wall_clock_s": 3600}),
        _fake_run("a-datagen", "datagen", 0.0, group="exp-20261015-a", start="2026-10-15T00:00:00Z",
                  **{"cost/wall_clock_s": 1800}),
        _fake_run("b-eval", "eval", 0.0, group="exp-20261015-b", start="2026-10-15T00:15:00Z",
                  **{"cost/wall_clock_s": 1800}),
    ]
    out = tracking.teacher_attribution(api=FakeApi(runs))
    assert out["service_usd"] == pytest.approx(4.0)
    assert out["by_group"] == pytest.approx({"exp-20261015-a": 1.5, "exp-20261015-b": 1.5})
    assert out["unattributed_usd"] == pytest.approx(1.0)
    costs = tracking.group_costs("exp-20261015-a", api=FakeApi(runs))
    assert costs["teacher_service_usd"] == pytest.approx(1.5)
    assert costs["total_with_teacher_usd"] == pytest.approx(1.5)


def test_phase_groups_and_stage_costs():
    groups = tracking.phase_groups("20261016")
    assert groups["sweep-27b"] == "exp-20261016-sweep-27b" and len(groups) == 5
    runs = [
        _fake_run("sweep-a", "distill", 2.0, group="exp-20261016-sweep-27b"),
        _fake_run("sweep-b", "distill", 3.0, group="exp-20261016-sweep-27b"),
    ]
    md = tracking.stage_cost_markdown(groups, api=FakeApi(runs))
    assert "| sweep-27b | `exp-20261016-sweep-27b` | 2 | 5.00 | 0.00 | 5.00 |" in md
    assert "**5.00**" in md


@pytest.mark.skipif(importlib.util.find_spec("wandb_workspaces") is None, reason="needs wandb-workspaces")
def test_build_phase_report():
    report = tracking.build_phase_report(tracking.phase_groups("20261016"), api=FakeApi([]))
    titles = [getattr(b, "text", "") for b in report.blocks]
    assert "LR sweep" in titles and "Cost per stage" in titles


def test_teacher_client_run_counts_as_experiment_activity():
    # A CPU client logged as job_type "teacher" (e.g. 27B labeling) still claims the service time.
    runs = [
        _fake_run("svc", "teacher", 2.0, group="exp-20261016-teacher-service", start="2026-10-16T00:00:00Z",
                  **{"cost/wall_clock_s": 600}),
        _fake_run("label", "teacher", 0.0, group="exp-20261016-label-27b", start="2026-10-16T00:00:00Z",
                  **{"cost/wall_clock_s": 600}),
    ]
    assert tracking.teacher_attribution(api=FakeApi(runs))["by_group"] == pytest.approx({"exp-20261016-label-27b": 2.0})
