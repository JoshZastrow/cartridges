"""Modal-hosted inference serving for the OCR/transcription model.

A separate Modal app from `training/` (own app name, own schema/API shape),
but it reuses training/finetune.py's `image` and `_load_base_model` directly
rather than re-declaring a lighter image. That was tried first and doesn't
work for this model: `unsloth/DeepSeek-OCR-2`'s trust_remote_code modeling
file imports `transformers.DeepseekV2Model`, which only exists once
`unsloth` has patched `transformers` at import time (or on a much newer
transformers than plain inference otherwise needs) -- so serving genuinely
needs the same heavy stack training does, not a trimmed-down one.

Run with:
uv run modal run src/cartridges/serve/inference.py
"""

from __future__ import annotations

import contextlib
import logging
import time

import modal

from cartridges.serve.schema import GenerateRequest, GenerateResponse, ImageContent
from cartridges.training.config import ModelSource, TrainConfig
from cartridges.training.finetune import _capped_generate, _load_base_model
from cartridges.training.finetune import image as training_image

MINUTES = 60

# Own Modal app ("cartridges-serve", not "cartridges-training") -- but the
# *same* image spec as training, since this model's trust_remote_code file
# needs unsloth's transformers patches regardless of whether you're training
# or just running inference (see module docstring).
app = modal.App("cartridges-serve", image=training_image)
hf_secret = modal.Secret.from_name("HF_TOKEN")  # adapters live in a private HF Hub repo
model_cache = modal.Volume.from_name("cartridges-training-cache", create_if_missing=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cartridges.serve")


@contextlib.contextmanager
def _stage(name: str, timings: dict[str, float] | None = None, **fields):
    """Like training/finetune.py's `_stage`, but also records elapsed time
    into `timings` (by stage name) so it can travel back in the response,
    not just sit in server-side logs."""
    context = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("[stage start] %s %s", name, context)
    start = time.monotonic()
    try:
        yield
    except Exception:
        logger.exception("[stage FAILED] %s after %.1fs", name, time.monotonic() - start)
        raise
    else:
        elapsed = time.monotonic() - start
        logger.info("[stage done] %s in %.1fs", name, elapsed)
        if timings is not None:
            timings[name] = elapsed


FREE_OCR_PROMPT = "<image>\nFree OCR. "


def _load_model(config, timings: dict[str, float]):
    train_config = TrainConfig(model_id=config.model_id, base_model_dir=config.base_model_dir)
    # _load_base_model logs its own timing under the "cartridges.training"
    # logger (no timings dict there), so wrap it here too to get it into the
    # response's stage_timings_s, not just server-side logs.
    with _stage("load_base_model", timings, base_model_dir=config.base_model_dir):
        model, tokenizer = _load_base_model(train_config)  # unsloth's FastVisionModel.from_pretrained

    if config.model_source == ModelSource.ARTIFACT:
        if not config.hf_revision:
            raise ValueError("hf_revision is required when model_source='artifact'")
        from peft import PeftModel

        with _stage(
            "attach_adapter", timings, hf_repo_id=config.hf_repo_id, hf_revision=config.hf_revision
        ):
            model = PeftModel.from_pretrained(model, config.hf_repo_id, revision=config.hf_revision)

    return model, tokenizer


def _extract(request: GenerateRequest) -> tuple[str, str]:
    """Pull (prompt_text, image_base64) out of the first user message."""
    for message in request.messages:
        prompt_text = None
        image_base64 = None
        for block in message.content:
            if isinstance(block, ImageContent):
                image_base64 = block.image_base64
            else:
                prompt_text = block.text
        if image_base64 is not None:
            return prompt_text or FREE_OCR_PROMPT, image_base64
    raise ValueError("no image content block found in request.messages")


@app.function(
    # L4 has plenty of VRAM for this model but far too little memory
    # bandwidth for autoregressive generation (bandwidth-bound, not VRAM-
    # bound) -- see training/finetune.py's identical note. A100 it is.
    gpu="A100",
    timeout=15 * MINUTES,
    volumes={"/cache": model_cache},
    secrets=[hf_secret],
)
def generate(request: GenerateRequest) -> GenerateResponse:
    """One-shot generate: load model, decode the request's image, run
    `model.infer`, return text + a stage-by-stage timing breakdown.

    Deliberately not a warm `@app.cls` service yet (that's the natural next
    step once this is more than a smoke test) -- every call here re-loads
    the model from scratch on purpose, so "how long does loading take" is
    always part of what gets measured, not hidden behind a warm container.
    """
    import base64
    import tempfile
    from pathlib import Path

    from unsloth import FastVisionModel

    config = request.serve_config
    timings: dict[str, float] = {}
    total_start = time.monotonic()

    prompt, image_base64 = _extract(request)
    model, tokenizer = _load_model(config, timings)
    FastVisionModel.for_inference(model)

    with _stage("decode_image", timings):
        image_bytes = base64.b64decode(image_base64)

    with _stage("inference", timings, prompt=prompt, max_new_tokens=config.max_new_tokens):
        with tempfile.TemporaryDirectory() as tmpdir, _capped_generate(model, config.max_new_tokens):
            image_path = Path(tmpdir) / "image.png"
            image_path.write_bytes(image_bytes)
            text = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=tmpdir,
                base_size=config.base_size,
                image_size=config.image_size,
                crop_mode=config.crop_mode,
                save_results=False,
                test_compress=False,
                # without this, infer() only streams tokens to stdout and
                # returns None.
                eval_mode=True,
            ).strip()

    total = time.monotonic() - total_start
    logger.info("generate done: total=%.1fs stages=%s", total, timings)
    return GenerateResponse(
        text=text,
        model_id=config.model_id.value,
        model_source=config.model_source.value,
        hf_revision=config.hf_revision,
        stage_timings_s=timings,
        total_s=total,
    )
