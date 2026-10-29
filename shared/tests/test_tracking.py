"""shared.tracking with W&B disabled: naming, cost, artifacts."""

import datetime
import sys
import types
from pathlib import Path

import pytest

from shared import tracking


@pytest.fixture(autouse=True)
def no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


def test_group_name_and_config_hash():
    assert tracking.group_name("Smoke Run!", datetime.date(2026, 10, 29)) == "exp-20261029-smoke-run"
    assert tracking.config_hash({"a": 1, "b": 2}) == tracking.config_hash({"b": 2, "a": 1})


@pytest.mark.parametrize("name, out", [
    (None, ("CPU", 1)), ("H200", ("H200", 1)), ("h100:2", ("H100", 2)), ("NVIDIA A100-SXM4-80GB", ("A100-80GB", 1)),
])
def test_normalize_gpu(name, out):
    assert tracking.normalize_gpu(name) == out


def test_start_run_cost_and_disabled_artifacts(tmp_path):
    with pytest.raises(ValueError):
        tracking.start_run("voice", "not-a-group")
    clock = iter([0.0, 1800.0, 1800.0, 1800.0])
    run = tracking.start_run("voice", tracking.group_name("t"), {"x": 1}, name="fake/stock/sft", gpu="H200",
                             clock=lambda: next(clock))
    assert run.cost() == {"cost/wall_clock_s": 1800.0, "cost/gpu_hours": 0.5, "cost/usd": pytest.approx(2.27)}
    assert tracking.log_artifact(run, tmp_path, type="voice-cartridge") is None
    assert run.use_artifact(tmp_path) == tmp_path
    run.finish()


def test_imports_from_a_shallow_path_as_in_a_modal_container(monkeypatch):
    """On Modal the module is /root/shared/tracking.py (3 parents, no .git); importing it must not raise."""
    module = types.ModuleType("tracking_shallow")
    module.__file__ = "/root/shared/tracking.py"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(Path(tracking.__file__).read_text(), module.__file__, "exec"), module.__dict__)
    assert isinstance(module._REPO_ROOT, Path)
