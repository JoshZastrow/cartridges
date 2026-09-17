"""PDF rasterization and OCR engines.

The OCR processor can use either local PaddleOCR or a Modal-hosted OCR model.
Callers should ask the factory for an engine instead of importing a concrete
implementation.

Deploy the model engine with:
uv run modal deploy src/cartridges/processors/ocr.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from cartridges.storage.cache import get_cached, hash_image_bytes, put_cached

if TYPE_CHECKING:
    from PIL import Image

try:
    import modal
except ModuleNotFoundError:
    class _MissingModal:
        class Image:
            @staticmethod
            def debian_slim(*args, **kwargs):
                del args, kwargs
                return _MissingModal.Image()

            def pip_install(self, *args, **kwargs):
                del args, kwargs
                return self

        class App:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def cls(self, *args, **kwargs):
                del args, kwargs

                def decorator(cls):
                    return cls

                return decorator

        class Volume:
            @staticmethod
            def from_name(*args, **kwargs):
                del args, kwargs
                return None

        class Cls:
            @staticmethod
            def from_name(*args, **kwargs):
                del args, kwargs
                raise RuntimeError(
                    "modal is required for model OCR; install project dependencies first"
                )

        @staticmethod
        def enter(*args, **kwargs):
            del args, kwargs

            def decorator(func):
                return func

            return decorator

        @staticmethod
        def method(*args, **kwargs):
            del args, kwargs

            def decorator(func):
                return func

            return decorator

    modal = _MissingModal()

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
QWEN_OCR_MODEL_NAME = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
PADDLE_OCR_MODEL_ID = "paddleocr:PP-OCRv5_server_det+PP-OCRv5_server_rec"
MINUTES = 60
OCR_ENGINE_KINDS = Literal["paddleocr", "model"]

# "Free OCR" = raw text only, no layout/grounding markup; that is what we want
# for a plain transcription.
FREE_OCR_PROMPT = "<image>\nFree OCR. "
QWEN_OCR_PROMPT = (
    "Transcribe all text in this image exactly as written, verbatim. "
    "Output only the transcribed text, nothing else."
)


class OcrEngine(Protocol):
    """Common interface for OCR engines."""

    def transcribe(
        self,
        image: Image.Image,
        *,
        source_path: str | None = None,
        source_mtime: float | None = None,
    ) -> str:
        """Return recognized text for a single page image."""


def pdf_page_to_image(pdf_path: Path, page_number: int, dpi: int = 400) -> Image.Image:
    import fitz  # pymupdf
    from PIL import Image

    doc = fitz.open(pdf_path)
    try:
        if not (1 <= page_number <= len(doc)):
            raise ValueError(f"page_number must be 1..{len(doc)}")
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        page = doc[page_number - 1]
        return Image.open(io.BytesIO(page.get_pixmap(matrix=matrix).tobytes("png")))
    finally:
        doc.close()


def pdf_to_images(pdf_path: Path, dpi: int = 400) -> list[Image.Image]:
    return [
        pdf_page_to_image(pdf_path, page_number, dpi=dpi)
        for page_number in range(1, pdf_page_count(pdf_path) + 1)
    ]


def pdf_page_count(pdf_path: Path) -> int:
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


class PaddleOcrEngine:
    """PaddleOCR-backed local OCR engine."""

    model_name = PADDLE_OCR_MODEL_ID

    def __init__(self) -> None:
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
            )
        return self._engine

    def transcribe(
        self,
        image: Image.Image,
        *,
        source_path: str | None = None,
        source_mtime: float | None = None,
    ) -> str:
        import numpy as np

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        page_hash = hash_image_bytes(image_bytes)

        cached = get_cached(page_hash, self.model_name)
        if cached is not None:
            return cached

        results = self._get_engine().predict(np.array(image.convert("RGB")))
        lines: list[str] = []
        for res in results:
            lines.extend(res["rec_texts"])
        transcription = "\n".join(lines)
        put_cached(page_hash, self.model_name, transcription, source_path, source_mtime)
        return transcription


class ModelOcrEngine:
    """Modal-hosted OCR model engine."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name

    def transcribe(
        self,
        image: Image.Image,
        *,
        source_path: str | None = None,
        source_mtime: float | None = None,
    ) -> str:
        class_name = TRANSCRIBER_CLASSES[self.model_name]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        page_hash = hash_image_bytes(image_bytes)

        cached = get_cached(page_hash, self.model_name)
        if cached is not None:
            return cached

        transcriber = modal.Cls.from_name("cartridges-vlm", class_name)
        transcription = transcriber().transcribe.remote(image_bytes)
        put_cached(page_hash, self.model_name, transcription, source_path, source_mtime)
        return transcription


_paddle_engine: PaddleOcrEngine | None = None
_model_engines: dict[str, ModelOcrEngine] = {}


def create_ocr_engine(
    kind: OCR_ENGINE_KINDS = "paddleocr",
    *,
    model_name: str | None = None,
) -> OcrEngine:
    """Create or reuse an OCR engine by kind."""
    global _paddle_engine

    if kind == "paddleocr":
        if _paddle_engine is None:
            _paddle_engine = PaddleOcrEngine()
        return _paddle_engine

    if kind == "model":
        resolved_model_name = model_name or MODEL_NAME
        if resolved_model_name not in _model_engines:
            _model_engines[resolved_model_name] = ModelOcrEngine(resolved_model_name)
        return _model_engines[resolved_model_name]

    raise ValueError(f"unknown OCR engine kind: {kind!r}")


def transcribe_image(
    image: Image.Image,
    *,
    engine: OCR_ENGINE_KINDS = "paddleocr",
    model_name: str | None = None,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> str:
    ocr_engine = create_ocr_engine(engine, model_name=model_name)
    return ocr_engine.transcribe(
        image,
        source_path=source_path,
        source_mtime=source_mtime,
    )


def ocr_image(image: Image.Image) -> str:
    """Compatibility helper for PaddleOCR transcription."""
    return transcribe_image(image, engine="paddleocr")


def model_transcribe_image(
    image: Image.Image,
    model_name: str | None = None,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> str:
    """Compatibility helper for model-backed OCR transcription."""
    return transcribe_image(
        image,
        engine="model",
        model_name=model_name,
        source_path=source_path,
        source_mtime=source_mtime,
    )


vlm_transcribe_image = model_transcribe_image

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.6.0",
    "torchvision==0.21.0",
    "transformers==4.46.3",
    "tokenizers==0.20.3",
    "einops",
    "addict",
    "easydict",
    "accelerate",
    "pillow",
)

app = modal.App("cartridges-vlm", image=image)

# Model registry: model_name -> Modal class name serving it.
TRANSCRIBER_CLASSES = {
    MODEL_NAME: "Transcriber",
    QWEN_OCR_MODEL_NAME: "QwenOCRTranscriber",
}

model_cache = modal.Volume.from_name("cartridges-vlm-cache", create_if_missing=True)


@app.cls(
    gpu="L4",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/cache": model_cache},
)
class Transcriber:
    @modal.enter()
    def load(self):
        import warnings

        import torch  # pyright: ignore[reportMissingImports]
        import transformers
        from transformers import AutoModel, AutoTokenizer

        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore", category=UserWarning)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, cache_dir="/cache"
        )
        # tokenizer.eos_token_id is None for this tokenizer. Resolve the
        # actual end-of-sequence token id so infer()'s internal generate()
        # call does not fall back to an unset pad/eos id.
        eos_id = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = eos_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = eos_id

        self.model = (
            AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir="/cache",
            )
            .eval()
            .cuda()
            .to(torch.bfloat16)
        )

    @modal.method()
    def transcribe(self, image_bytes: bytes) -> str:
        import contextlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.png"
            image_path.write_bytes(image_bytes)
            # model.infer() prints internal debug shapes via plain print();
            # silence stdout for the duration of the call.
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.model.infer(
                    self.tokenizer,
                    prompt=FREE_OCR_PROMPT,
                    image_file=str(image_path),
                    output_path=tmpdir,
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    eval_mode=True,
                )
            return result.strip()


@app.cls(
    gpu="L4",
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    volumes={"/cache": model_cache},
)
class QwenOCRTranscriber:
    @modal.enter()
    def load(self):
        import torch  # pyright: ignore[reportMissingImports]
        import transformers
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        transformers.logging.set_verbosity_error()

        self.processor = AutoProcessor.from_pretrained(
            QWEN_OCR_MODEL_NAME, cache_dir="/cache"
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_OCR_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir="/cache",
        )

    @modal.method()
    def transcribe(self, image_bytes: bytes) -> str:
        from PIL import Image

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QWEN_OCR_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_image], padding=True, return_tensors="pt"
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
