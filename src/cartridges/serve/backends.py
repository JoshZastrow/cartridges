"""What the gateway calls to generate: the Teacher (vLLM, journal context in
the prompt) and the CartridgeServer (HF, base model + cartridge), both deployed
Modal classes. Each call blocks until the answer is back (the gateway runs it
in a thread) and returns a `Generation`.

Calls go through `.spawn()` + `.get(timeout=...)`, so a cold start (minutes)
is bounded: on timeout the call is cancelled, since Modal would otherwise keep
it queued and start a container for it later. Failures of the transport or the
container (not of the request itself) become the gateway's
`TransientBackendError` (the original exception is `__cause__`), the one class
it retries.
"""

from __future__ import annotations

import functools

import modal
from pydantic import BaseModel, ConfigDict

from cartridges.cartridge import teacher
from cartridges.cartridge.serve_cartridge import cartridge_cls
from cartridges.serve.gateway import TransientBackendError  # the class the gateway's retry loop catches

# Retryable: the modal client's transport and server-side failures, a
# container that died or timed out (FunctionTimeoutError is a modal
# TimeoutError), an expired output, and the builtin TimeoutError that
# FunctionCall.get(timeout=...) raises. Not here: InvalidError, NotFoundError,
# AuthError, RequestSizeError, (De)SerializationError and the user exceptions
# a call re-raises, which repeat on retry.
TRANSIENT_ERRORS = (
    TimeoutError,
    modal.exception.TimeoutError,
    modal.exception.RemoteError,
    modal.exception.ConnectionError,
    modal.exception.InternalFailure,
    modal.exception.ServiceError,
)


class Generation(BaseModel):
    """One finished answer. prompt_tokens counts the conversation the client
    sent (the student's prompt_ids), not the Teacher's context block."""

    model_config = ConfigDict(frozen=True)

    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


def from_completion(completion: teacher.Completion) -> Generation:
    return Generation(
        text=completion.text, finish_reason=completion.finish_reason,
        prompt_tokens=len(completion.prompt_ids), completion_tokens=len(completion.completion_ids),
    )


def await_call(call, timeout_s: float):
    """`call.get(timeout_s)`, mapping transport failures to TransientBackendError
    and cancelling the call when the wait runs out."""
    try:
        return call.get(timeout=timeout_s)
    except TRANSIENT_ERRORS as e:
        if isinstance(e, TimeoutError):
            call.cancel()
        raise TransientBackendError(f"{type(e).__name__}: {e}") from e


@functools.lru_cache(maxsize=None)
def teacher_handle(model_id: str):
    """The deployed Teacher for `model_id`; cached so each request does not
    re-hydrate the class handle."""
    return teacher.teacher_cls("Teacher", model_id)


@functools.lru_cache(maxsize=None)
def cartridge_handle(cartridge: str, gpu: str | None = None):
    return cartridge_cls(cartridge, **({"gpu": gpu} if gpu else {}))


def teacher_generate(
    model_id: str, with_context: bool, messages: list[dict], max_tokens: int, temperature: float | None,
    timeout_s: float = 1500,
) -> Generation:
    call = teacher_handle(model_id).generate.spawn(
        messages, max_tokens=max_tokens, temperature=temperature, with_context=with_context
    )
    return from_completion(await_call(call, timeout_s)[0])


def cartridge_generate(
    cartridge: str, messages: list[dict], max_tokens: int, temperature: float | None, timeout_s: float = 1500,
    gpu: str | None = None,
) -> Generation:
    call = cartridge_handle(cartridge, gpu).generate.spawn(messages, max_tokens=max_tokens, temperature=temperature)
    return Generation(**await_call(call, timeout_s))


def teacher_stream(model_id: str, with_context: bool, messages: list[dict], max_tokens: int, temperature: float | None):
    """Teacher.generate_stream's events: {text, tokens} deltas, then
    {finish_reason, prompt_tokens, completion_tokens}. Not retried."""
    return teacher_handle(model_id).generate_stream.remote_gen(
        messages, max_tokens=max_tokens, with_context=with_context, temperature=temperature
    )


def cartridge_stream(
    cartridge: str, messages: list[dict], max_tokens: int, temperature: float | None, gpu: str | None = None
):
    """CartridgeServer.generate_stream's events, in teacher_stream's format."""
    return cartridge_handle(cartridge, gpu).generate_stream.remote_gen(
        messages, max_tokens=max_tokens, temperature=temperature
    )
