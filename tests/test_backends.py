"""The gateway's backend calls against fake Modal handles, and the
CartridgeServer's pure parts, on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import modal
import pytest

from cartridges.cartridge import serve_cartridge, teacher
from cartridges.serve import backends

MESSAGES = [{"role": "user", "content": "What was the author listening to on the morning of 2021-11-19?"}]


class FakeCall:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.cancelled, self.timeout = result, error, False, None

    def get(self, timeout=None):
        self.timeout = timeout
        if self.error:
            raise self.error
        return self.result

    def cancel(self):
        self.cancelled = True


def _handle(call, spawned):
    def spawn(messages, **kw):
        spawned.append((messages, kw))
        return call

    return SimpleNamespace(generate=SimpleNamespace(spawn=spawn))


def test_teacher_generate_maps_completion_and_passes_the_options(monkeypatch):
    completion = teacher.Completion(text="Cat Stevens [Auto biography:p.1].", prompt_ids=[1] * 40,
                                    completion_ids=[2] * 9, finish_reason="stop")
    call, spawned = FakeCall(result=[completion]), []
    monkeypatch.setattr(backends, "teacher_handle", lambda model_id: _handle(call, spawned))

    out = backends.teacher_generate("Qwen/Qwen3.5-4B", True, MESSAGES, 128, 0.2, timeout_s=90)
    assert out == backends.Generation(text=completion.text, finish_reason="stop", prompt_tokens=40, completion_tokens=9)
    assert spawned == [(MESSAGES, {"max_tokens": 128, "temperature": 0.2, "with_context": True})]
    assert call.timeout == 90 and not call.cancelled
    with pytest.raises(Exception):  # frozen
        out.text = "x"


def test_cartridge_generate_builds_generation_from_the_dict(monkeypatch):
    result = {"text": "Cat Stevens", "finish_reason": "length", "prompt_tokens": 30, "completion_tokens": 128}
    call, spawned = FakeCall(result=result), []
    monkeypatch.setattr(backends, "cartridge_handle", lambda cartridge: _handle(call, spawned))

    out = backends.cartridge_generate("/vol/cartridges/x", MESSAGES, 128, None)
    assert out == backends.Generation(**result)
    assert spawned == [(MESSAGES, {"max_tokens": 128, "temperature": None})]
    assert call.timeout == 1500


@pytest.mark.parametrize("error", [
    modal.exception.RemoteError("container died"),
    modal.exception.FunctionTimeoutError("30 min"),
    modal.exception.ConnectionError("grpc"),
    modal.exception.InternalFailure("retriable"),
])
def test_transport_failures_become_transient_errors(monkeypatch, error):
    call = FakeCall(error=error)
    monkeypatch.setattr(backends, "teacher_handle", lambda model_id: _handle(call, []))
    with pytest.raises(backends.TransientBackendError) as info:
        backends.teacher_generate("Qwen/Qwen3.5-4B", True, MESSAGES, 8, None)
    assert info.value.__cause__ is error
    assert not call.cancelled  # the call already failed; nothing to cancel


def test_get_timeout_cancels_the_call_and_is_transient(monkeypatch):
    call = FakeCall(error=TimeoutError())  # what FunctionCall.get(timeout=...) raises
    monkeypatch.setattr(backends, "cartridge_handle", lambda cartridge: _handle(call, []))
    with pytest.raises(backends.TransientBackendError):
        backends.cartridge_generate("/vol/cartridges/x", MESSAGES, 8, None, timeout_s=1)
    assert call.cancelled


@pytest.mark.parametrize("error", [
    modal.exception.InvalidError("bad parameter"),
    modal.exception.NotFoundError("no such app"),
    ValueError("re-raised from the container"),
])
def test_input_errors_propagate_unchanged(monkeypatch, error):
    call = FakeCall(error=error)
    monkeypatch.setattr(backends, "teacher_handle", lambda model_id: _handle(call, []))
    with pytest.raises(type(error)):
        backends.teacher_generate("Qwen/Qwen3.5-4B", False, MESSAGES, 8, None)


def test_handles_are_cached_per_model(monkeypatch):
    made = []
    monkeypatch.setattr(teacher, "teacher_cls", lambda name, model_id: made.append(model_id) or object())
    backends.teacher_handle.cache_clear()
    a, b = backends.teacher_handle("Qwen/Qwen3.5-4B"), backends.teacher_handle("Qwen/Qwen3.5-4B")
    backends.teacher_handle("Qwen/Qwen3.8-27B")
    assert a is b and made == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.8-27B"]
    backends.teacher_handle.cache_clear()


# ---------------------------------------------------------------- serve_cartridge


def test_finish_stops_at_the_first_eos():
    assert serve_cartridge.finish([5, 6, 7, 151645, 151645, 151645], [151645, 151643]) == ([5, 6, 7], "stop")
    assert serve_cartridge.finish([151643], [151645, 151643]) == ([], "stop")
    assert serve_cartridge.finish([5, 6, 7], [151645]) == ([5, 6, 7], "length")
    assert serve_cartridge.finish([], [151645]) == ([], "length")


def test_cartridge_cls_makes_one_pool_per_path(monkeypatch):
    made = []

    class FakeCls:
        def with_options(self, **options):
            made.append(("options", options))
            return self

        def __call__(self, **params):
            made.append(("params", params))
            return params

    monkeypatch.setattr(modal.Cls, "from_name", staticmethod(lambda app, name: made.append((app, name)) or FakeCls()))
    assert serve_cartridge.cartridge_cls("/vol/cartridges/x") == {"cartridge": "/vol/cartridges/x"}
    assert made == [(serve_cartridge.APP_NAME, "CartridgeServer"), ("params", {"cartridge": "/vol/cartridges/x"})]
    serve_cartridge.cartridge_cls(gpu="A100-80GB")
    assert made[-2:] == [("options", {"gpu": "A100-80GB"}), ("params", {"cartridge": serve_cartridge.SMOKE_CARTRIDGE})]
