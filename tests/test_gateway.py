import json

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

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
    # the empty system turn is dropped, the rest passes through
    assert calls == [("Qwen/Qwen3.5-4B", True, [{"role": "user", "content": "When did I move?"}], 64, 0.2)]


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
