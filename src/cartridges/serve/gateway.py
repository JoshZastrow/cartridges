"""OpenAI-compatible gateway in front of the Modal backends, for LibreChat.

    GET  /v1/models            the registry (serve/chat_models.py)
    POST /v1/chat/completions  stream or not; OpenAI wire format

The backends (serve/backends.py) block until the answer is done; the gateway
runs them in a thread and, for `stream: true`, chunks the finished text into
`chat.completion.chunk` events. Transport-class failures
(TransientBackendError) are retried twice with a short backoff; anything
else is not retried. Errors are OpenAI-style `{error: {message, type, code}}`.

Run with: uv run cartridges gateway_serve --port 8008
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from pydantic import BaseModel

from cartridges.cartridge.evals import strip_think
from cartridges.serve.chat_models import CHAT_MODELS, ChatModel

DEFAULT_API_KEY = "cartridges-local"
CALL_TIMEOUT_S = 25 * 60  # one Modal call, cold start included
MAX_RETRIES = 2
STREAM_CHUNKS = 4
ECHO_PREFIX = "echo: "


class TransientBackendError(Exception):
    """A retryable failure between the gateway and Modal (transport, RemoteError).
    backends.py raises this; validation errors must not be mapped to it."""


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 1024
    temperature: float | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: dict[str, str]
    finish_reason: str | None


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def text_of(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content or ""


def clean_messages(messages: list[ChatMessage]) -> list[dict]:
    """Pass the client's messages through; drop empty system turns."""
    return [
        m.model_dump(mode="json")
        for m in messages
        if not (m.role == "system" and not text_of(m.content).strip())
    ]


def mock_generate(messages: list[dict]) -> dict:
    last_user = next((text_of(m["content"]) for m in reversed(messages) if m["role"] == "user"), "")
    text = ECHO_PREFIX + last_user
    prompt_tokens = sum(len(text_of(m["content"]).split()) for m in messages)
    return {"text": text, "finish_reason": "stop", "prompt_tokens": prompt_tokens, "completion_tokens": len(text.split())}


def split_text(text: str, n: int = STREAM_CHUNKS) -> list[str]:
    if not text:
        return [""]
    size = -(-len(text) // n)
    return [text[i : i + size] for i in range(0, len(text), size)]


def build_app(
    teacher_generate: Callable,
    cartridge_generate: Callable,
    api_key: str = DEFAULT_API_KEY,
    models: dict[str, ChatModel] = CHAT_MODELS,
    backoff_s: float = 1.0,
    call_timeout_s: float = CALL_TIMEOUT_S,
):
    """The public API. `teacher_generate(model_id, with_context, messages,
    max_tokens, temperature)` and `cartridge_generate(cartridge, messages,
    max_tokens, temperature)` block and return a backends.Generation (fakes in
    tests)."""
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    def check_key(authorization: str = Header(default="")) -> None:
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    web = FastAPI(title="cartridges gateway")

    @web.exception_handler(HTTPException)
    async def openai_error(request: Request, exc: HTTPException) -> JSONResponse:
        types = {401: "authentication_error", 404: "invalid_request_error", 504: "timeout_error"}
        error = {"message": str(exc.detail), "type": types.get(exc.status_code, "api_error"), "code": exc.status_code}
        return JSONResponse({"error": error}, status_code=exc.status_code)

    def call_backend(model: ChatModel, messages: list[dict], request: ChatRequest) -> dict:
        if model.backend == "teacher":
            return teacher_generate(model.model_id, model.with_context, messages, request.max_tokens, request.temperature)
        return cartridge_generate(model.cartridge, messages, request.max_tokens, request.temperature)

    async def generate(model: ChatModel, request: ChatRequest) -> tuple[str, str | None, Usage]:
        messages = clean_messages(request.messages)
        if model.backend == "mock":
            result = mock_generate(messages)
        else:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(call_backend, model, messages, request), timeout=call_timeout_s
                    )
                    break
                except TransientBackendError as e:
                    if attempt == MAX_RETRIES:
                        raise HTTPException(status_code=502, detail=f"backend failed after {attempt + 1} attempts: {e}")
                    await asyncio.sleep(backoff_s * 2**attempt)
                except TimeoutError:
                    raise HTTPException(status_code=504, detail=f"backend call exceeded {call_timeout_s:.0f}s")
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"backend error: {e}")
        get = result.get if isinstance(result, dict) else lambda k: getattr(result, k)
        usage = Usage(
            prompt_tokens=get("prompt_tokens") or 0,
            completion_tokens=get("completion_tokens") or 0,
            total_tokens=(get("prompt_tokens") or 0) + (get("completion_tokens") or 0),
        )
        return strip_think(get("text") or ""), get("finish_reason"), usage

    def sse(request_id: str, created: int, model: str, delta: dict, finish_reason: str | None, **extra) -> str:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def stream(request_id: str, created: int, model: str, text: str, finish_reason: str | None, usage: Usage) -> AsyncIterator[str]:
        yield sse(request_id, created, model, {"role": "assistant", "content": ""}, None)
        for piece in split_text(text):
            yield sse(request_id, created, model, {"content": piece}, None)
        yield sse(request_id, created, model, {}, finish_reason or "stop", usage=usage.model_dump())
        yield "data: [DONE]\n\n"

    @web.get("/v1/models", dependencies=[Depends(check_key)])
    async def list_models() -> dict:
        return {"object": "list", "data": [{"id": m.id, "object": "model", "owned_by": "cartridges"} for m in models.values()]}

    @web.post("/v1/chat/completions", dependencies=[Depends(check_key)])
    async def chat_completions(request: ChatRequest):
        model = models.get(request.model)
        if model is None:
            raise HTTPException(status_code=404, detail=f"model not found: {request.model}")
        request_id, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        text, finish_reason, usage = await generate(model, request)
        if request.stream:
            return StreamingResponse(
                stream(request_id, created, model.id, text, finish_reason, usage), media_type="text/event-stream"
            )
        return ChatResponse(
            id=request_id,
            created=created,
            model=model.id,
            choices=[Choice(message={"role": "assistant", "content": text}, finish_reason=finish_reason or "stop")],
            usage=usage,
        )

    return web
