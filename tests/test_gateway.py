import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from cartridges.cartridge.tasks.label import CITATION_REMINDER
from cartridges.serve.chat_models import CHAT_MODELS
from cartridges.serve.gateway import ECHO_PREFIX, TransientBackendError, build_app

AUTH = {"Authorization": "Bearer key"}


class FakeGeneration(BaseModel, frozen=True):
    text: str
    finish_reason: str | None = "stop"
    prompt_tokens: int = 10
    completion_tokens: int = 3


def make_client(failures: list[Exception] | None = None, calls: list | None = None) -> TestClient:
    """A gateway whose teacher raises each exception in `failures` first, then answers."""
    failures = list(failures or [])

    def teacher_generate(model_id, with_context, messages, max_tokens, temperature):
        if calls is not None:
            calls.append((model_id, with_context, messages, max_tokens, temperature))
        if failures:
            raise failures.pop(0)
        return FakeGeneration(text="<think>hmm</think>The answer [p. 3].")

    def cartridge_generate(cartridge, messages, max_tokens, temperature):
        return FakeGeneration(text=f"cartridge {cartridge}")

    return TestClient(build_app(teacher_generate, cartridge_generate, api_key="key", backoff_s=0))


def chat(model: str, content: str = "hi", **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


def test_models_lists_the_registry():
    response = make_client().get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == list(CHAT_MODELS)
    assert body["data"][0] == {"id": "journal-teacher-27b", "object": "model", "owned_by": "cartridges"}


def test_non_stream_reply_strips_think_and_reports_usage():
    calls: list = []
    body = {
        "model": "journal-teacher-4b",
        "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "When did I move?"}],
        "max_tokens": 64,
        "temperature": 0.2,
    }
    response = make_client(calls=calls).post("/v1/chat/completions", json=body, headers=AUTH)
    assert response.status_code == 200
    reply = response.json()
    assert reply["object"] == "chat.completion" and reply["model"] == "journal-teacher-4b"
    assert reply["choices"][0]["message"] == {"role": "assistant", "content": "The answer [p. 3]."}
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    # the empty system turn is dropped; the model's default system turn opens the rest
    reminded = [{"role": "system", "content": CITATION_REMINDER}, {"role": "user", "content": "When did I move?"}]
    assert calls == [("Qwen/Qwen3.5-4B", True, reminded, 64, 0.2)]


def test_cartridge_model_routes_to_cartridge_backend():
    response = make_client().post("/v1/chat/completions", json=chat("journal-cartridge-4b-smoke"), headers=AUTH)
    assert response.json()["choices"][0]["message"]["content"] == "cartridge /vol/cartridges/exp-20261013-smoke/grpo"


def test_mock_echo_echoes_the_last_user_turn():
    body = chat("mock-echo", "hello")
    body["messages"].append({"role": "assistant", "content": "echo: hello"})
    body["messages"].append({"role": "user", "content": "second turn"})
    response = make_client().post("/v1/chat/completions", json=body, headers=AUTH)
    assert response.json()["choices"][0]["message"]["content"] == ECHO_PREFIX + "second turn"


def test_stream_framing_ends_with_done():
    with make_client().stream("POST", "/v1/chat/completions", json=chat("mock-echo", "stream me", stream=True), headers=AUTH) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == ECHO_PREFIX + "stream me"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["usage"]["completion_tokens"] == 3
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_retry_succeeds_after_one_transient_error():
    calls: list = []
    client = make_client(failures=[TransientBackendError("connection reset")], calls=calls)
    response = client.post("/v1/chat/completions", json=chat("journal-teacher-4b"), headers=AUTH)
    assert response.status_code == 200
    assert len(calls) == 2


def test_transient_errors_give_502_after_two_retries():
    calls: list = []
    failures = [TransientBackendError("down")] * 3
    response = make_client(failures=failures, calls=calls).post("/v1/chat/completions", json=chat("journal-teacher-4b"), headers=AUTH)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == 502 and "down" in response.json()["error"]["message"]
    assert len(calls) == 3


def test_no_retry_on_non_transient_error():
    calls: list = []
    response = make_client(failures=[ValueError("bad messages")], calls=calls).post(
        "/v1/chat/completions", json=chat("journal-teacher-4b"), headers=AUTH
    )
    assert response.status_code == 502
    assert len(calls) == 1


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_bad_key_is_401(headers):
    client = make_client()
    assert client.get("/v1/models", headers=headers).status_code == 401
    response = client.post("/v1/chat/completions", json=chat("mock-echo"), headers=headers)
    assert response.status_code == 401
    assert response.json()["error"] == {"message": "invalid API key", "type": "authentication_error", "code": 401}


def test_unknown_model_is_404():
    response = make_client().post("/v1/chat/completions", json=chat("gpt-9"), headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "gpt-9" in response.json()["error"]["message"]


def live_client(calls: list, fail_after: int | None = None) -> TestClient:
    """A gateway with streaming backends that yield two deltas and a final event
    (or raise after `fail_after` deltas)."""

    def events(*args, **kwargs):
        calls.append((args, kwargs))
        for i, (text, tokens) in enumerate([("Hel", 1), ("lo.", 2)]):
            if i == fail_after:
                raise RuntimeError("container died")
            yield {"text": text, "tokens": tokens}
        yield {"finish_reason": "stop", "prompt_tokens": 7, "completion_tokens": 3}

    def unused(*args, **kwargs):
        raise AssertionError("stream requests must not use the blocking backends")

    return TestClient(build_app(unused, unused, api_key="key", teacher_stream=events, cartridge_stream=events))


def stream_lines(client: TestClient, model: str) -> list[str]:
    with client.stream("POST", "/v1/chat/completions", json=chat(model, stream=True), headers=AUTH) as response:
        assert response.status_code == 200
        return [line for line in response.iter_lines() if line]


def test_live_stream_sends_each_event_with_its_token_count():
    calls: list = []
    lines = stream_lines(live_client(calls), "journal-teacher-27b")
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert [(c["choices"][0]["delta"]["content"], c["tokens"]) for c in chunks[1:3]] == [("Hel", 1), ("lo.", 2)]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    reminded = [{"role": "system", "content": CITATION_REMINDER}, {"role": "user", "content": "hi"}]
    assert calls == [(("Qwen/Qwen3.8-27B", True, reminded, 1024, None), {})]


def test_live_stream_passes_the_cartridge_gpu():
    calls: list = []
    stream_lines(live_client(calls), "journal-cartridge-27b-am")
    (args, kwargs), = calls
    assert args[0] == CHAT_MODELS["journal-cartridge-27b-am"].cartridge and kwargs == {"gpu": "H200"}


def test_live_stream_failure_is_an_error_event_then_done():
    lines = stream_lines(live_client([], fail_after=1), "journal-cartridge-27b-am")
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert events[1]["choices"][0]["delta"]["content"] == "Hel"
    assert events[-1]["error"]["code"] == 502 and "container died" in events[-1]["error"]["message"]


def vllm_client(calls: list, fail_after: int | None = None, n: int = 5) -> TestClient:
    """A gateway whose cartridge-vllm backend streams `n` one-token deltas and a
    final event (or raises after `fail_after` deltas), and answers non-streamed."""

    def stream(model_id, cartridge, messages, max_tokens, temperature):
        calls.append(("stream", model_id, cartridge, max_tokens, temperature))
        for i in range(n):
            if i == fail_after:
                raise RuntimeError("no vLLM export")
            yield {"text": f"w{i} ", "tokens": 1}
        yield {"finish_reason": "length", "prompt_tokens": 12, "completion_tokens": n}

    def generate(model_id, cartridge, messages, max_tokens, temperature):
        calls.append(("generate", model_id, cartridge, max_tokens, temperature))
        return FakeGeneration(text="<think></think>vLLM answer", completion_tokens=2)

    def unused(*args, **kwargs):
        raise AssertionError("cartridge-vllm models must not use the other backends")

    return TestClient(build_app(unused, unused, api_key="key", teacher_stream=unused, cartridge_stream=unused,
                                cartridge_vllm_generate=generate, cartridge_vllm_stream=stream))


def test_cartridge_vllm_streams_each_delta_then_usage_then_done():
    calls: list = []
    lines = stream_lines(vllm_client(calls), "journal-cartridge-4b-smoke-vllm")
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    content = [c for c in chunks[1:] if c["choices"][0]["delta"].get("content")]
    assert [c["choices"][0]["delta"]["content"] for c in content] == [f"w{i} " for i in range(5)]
    assert sum(c["tokens"] for c in content) == chunks[-1]["usage"]["completion_tokens"] == 5
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    model = CHAT_MODELS["journal-cartridge-4b-smoke-vllm"]
    assert calls == [("stream", "Qwen/Qwen3.5-4B", model.cartridge, 1024, None)]


def test_cartridge_vllm_stream_failure_is_an_error_event_then_done():
    lines = stream_lines(vllm_client([], fail_after=2), "journal-cartridge-27b-am-vllm")
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert [e["choices"][0]["delta"]["content"] for e in events[1:3]] == ["w0 ", "w1 "]
    assert events[-1]["error"]["code"] == 502 and "no vLLM export" in events[-1]["error"]["message"]


def test_cartridge_vllm_non_stream_routes_to_the_engine_of_the_base_model():
    calls: list = []
    response = vllm_client(calls).post("/v1/chat/completions", json=chat("journal-cartridge-27b-am-vllm"), headers=AUTH)
    assert response.json()["choices"][0]["message"]["content"] == "vLLM answer"
    assert calls == [("generate", "Qwen/Qwen3.8-27B", "/vol/cartridges/exp-20261027-am-stage2/onpolicy-best", 1024, None)]


def test_journal_models_get_the_training_system_turn_when_the_client_sends_none():
    # Teacher labels, cartridge training and evals all open with the citation
    # reminder; without it the 27B cartridge answers every question alike.
    seen: list = []

    def generate(*args, **kwargs):
        seen.append(args[-3])  # messages
        return FakeGeneration(text="ok")

    client = TestClient(build_app(generate, generate, api_key="key", cartridge_vllm_generate=generate))
    own = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q"}]
    for model in ("journal-cartridge-27b-am-vllm", "journal-cartridge-27b-am", "journal-teacher-27b"):
        client.post("/v1/chat/completions", json=chat(model, "q"), headers=AUTH)
        client.post("/v1/chat/completions", json={"model": model, "messages": own}, headers=AUTH)
    client.post("/v1/chat/completions", json=chat("qwen3.5-4b-base", "q"), headers=AUTH)
    reminded = [{"role": "system", "content": CITATION_REMINDER}, {"role": "user", "content": "q"}]
    assert seen == [reminded, own] * 3 + [[{"role": "user", "content": "q"}]]
