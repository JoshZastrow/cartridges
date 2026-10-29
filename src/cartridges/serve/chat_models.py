"""The chat model registry: the ids LibreChat offers and where each one runs.

Picking a model id in LibreChat is the hot swap; a new cartridge is one row.
See docs/plans/librechat.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

SMOKE_CARTRIDGE = "/vol/cartridges/exp-20261013-smoke/grpo"


class ChatModel(BaseModel, frozen=True):
    id: str
    backend: Literal["teacher", "cartridge", "mock"]
    model_id: str | None = None  # teacher backend
    with_context: bool = False  # teacher backend: journal context block in the prompt
    cartridge: str | None = None  # cartridge backend: volume path
    gpu: str | None = None  # cartridge backend: CartridgeServer GPU (None: its L4 default)


CHAT_MODELS: dict[str, ChatModel] = {
    m.id: m
    for m in [
        ChatModel(id="journal-teacher-27b", backend="teacher", model_id="Qwen/Qwen3.8-27B", with_context=True),
        ChatModel(id="journal-teacher-4b", backend="teacher", model_id="Qwen/Qwen3.5-4B", with_context=True),
        ChatModel(id="qwen3.5-4b-base", backend="teacher", model_id="Qwen/Qwen3.5-4B", with_context=False),
        ChatModel(id="journal-cartridge-4b-smoke", backend="cartridge", cartridge=SMOKE_CARTRIDGE),
        ChatModel(id="journal-cartridge-27b-am", backend="cartridge", gpu="H200",
                  cartridge="/vol/cartridges/exp-20261027-am-stage2/onpolicy-best"),  # attention-matched init, stage 2
        ChatModel(id="mock-echo", backend="mock"),
    ]
}
