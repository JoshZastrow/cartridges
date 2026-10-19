import logging
import threading

import pytest

from cartridges.cartridge.tasks.qgen import probe, qgen_eval, topup_mix


class FakeCall:
    """A modal FunctionCall: get(timeout) raises TimeoutError until `ready`."""

    def __init__(self, ready_after=None, error=None):
        self.ready, self.error, self.cancelled = threading.Event(), error, False
        if ready_after is not None:
            threading.Timer(ready_after, self.ready.set).start()

    def get(self, timeout):
        if self.error:
            raise self.error
        if not self.ready.wait(timeout):
            raise TimeoutError()
        return "OK"

    def cancel(self):
        self.cancelled = True


def test_probe_logs_while_waiting_and_returns(caplog):
    call = FakeCall(ready_after=0.25)
    with caplog.at_level(logging.INFO):
        assert probe(lambda: call, timeout_s=5, log_every_s=0.05) == "OK"
    assert any("teacher probe: waiting" in r.message for r in caplog.records) and not call.cancelled


def test_probe_cancels_the_call_on_timeout_or_error():
    call = FakeCall()
    with pytest.raises(TimeoutError):
        probe(lambda: call, timeout_s=0.1, log_every_s=0.03)
    assert call.cancelled

    call = FakeCall(error=ConnectionError("no capacity"))
    with pytest.raises(ConnectionError):
        probe(lambda: call, timeout_s=1, log_every_s=0.1)
    assert call.cancelled


def test_topup_mix_weights_the_short_kinds():
    assert topup_mix({"recall": 0, "situation_help": 0}) is None
    assert topup_mix({"recall": 3, "situation_help": 1, "role_request": 0}) == {"recall": 0.75, "situation_help": 0.25}


def fakes(calls, short):
    def synthesize(n, seed, mix):
        calls.append(("synthesize", n, seed, mix))
        return {"run_id": f"r{seed}"}

    return dict(
        spawn_probe=lambda: calls.append("probe") or FakeCall(ready_after=0),
        seed=lambda: calls.append("seed") or {"inserted_templates": 15},
        synthesize=synthesize,
        shortfall=lambda run_ids: calls.append(("shortfall", list(run_ids))) or short,
        pairs=lambda run_ids: calls.append(("pairs", list(run_ids))) or {"pairs": 100},
        stop=lambda: calls.append("stop"),
    )


def test_qgen_eval_without_topup():
    calls = []
    report = qgen_eval(**fakes(calls, {"recall": 0}))
    assert calls == ["probe", "seed", ("synthesize", 150, 0, None), ("shortfall", ["r0"]), ("pairs", ["r0"]), "stop"]
    assert report["pairs"] == {"pairs": 100} and "topup" not in report


def test_qgen_eval_tops_up_short_kinds():
    calls = []
    qgen_eval(**fakes(calls, {"recall": 2, "situation_help": 0}))
    assert ("synthesize", 50, 1, {"recall": 1.0}) in calls
    assert ("pairs", ["r0", "r1"]) in calls and calls[-1] == "stop"


def test_qgen_eval_stops_the_teacher_when_a_step_fails():
    calls = []
    steps = fakes(calls, {})

    def broken(n, seed, mix):
        raise RuntimeError("container gone")

    with pytest.raises(RuntimeError):
        qgen_eval(**{**steps, "synthesize": broken})
    assert calls == ["probe", "seed", "stop"]

    calls.clear()
    with pytest.raises(TimeoutError):
        qgen_eval(**{**steps, "spawn_probe": FakeCall}, probe_timeout_s=0.05, probe_log_every_s=0.02)
    assert calls == ["stop"]


def test_qgen_eval_classify_then_topup_existing_runs():
    calls = []
    steps = fakes(calls, {"recall": 3})
    report = qgen_eval(**{**steps, "seed": None}, classify=lambda: calls.append("classify") or {"cache_misses": 9},
                       run_ids=["old"], n=0, topup_n=20)
    assert calls == ["probe", "classify", ("shortfall", ["old"]), ("synthesize", 20, 1, {"recall": 1.0}),
                     ("pairs", ["old", "r1"]), "stop"]
    assert report["classify"] == {"cache_misses": 9}
