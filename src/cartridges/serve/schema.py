"""Request/response schema for `serve.inference`.

Loosely mirrors the OpenAI chat-completions multimodal shape (a message with
text/image content blocks in, text out) rather than DeepSeek-OCR-2's own
custom `model.infer(prompt=..., image_file=...)` signature -- so callers
don't need to know this specific model's calling convention.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from cartridges.training.config import BaseModelId, ModelSource, TrainConfig


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    image_base64: str  # raw PNG bytes, base64-encoded (no data: URI prefix)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent]


class ServeConfig(BaseModel):
    model_id: BaseModelId = BaseModelId.DEEPSEEK_OCR_2
    base_model_dir: str = "/cache/deepseek_ocr"
    model_source: ModelSource = ModelSource.BASE
    hf_repo_id: str = TrainConfig().hf_repo_id
    # required when model_source == ARTIFACT: a revision (git tag/branch/
    # commit sha) in hf_repo_id, e.g. "run-<wandb_run_id>" or "date-...".
    hf_revision: str | None = None

    base_size: int = 1024
    # See training/config.py's TrainConfig.image_size comment: the vision
    # encoder only supports 768 or 1024, not the reference notebook's 640.
    image_size: int = 768
    crop_mode: bool = True

    # See training/config.py's TrainConfig.max_new_tokens comment: infer()
    # hardcodes 8192 with no other reliable early-stop guarantee.
    max_new_tokens: int = 2560


class GenerateRequest(BaseModel):
    serve_config: ServeConfig = ServeConfig()
    messages: list[ChatMessage]


class GenerateResponse(BaseModel):
    text: str
    model_id: str
    model_source: str
    hf_revision: str | None
    stage_timings_s: dict[str, float]
    total_s: float
