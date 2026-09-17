"""PDF rasterization + OCR, page by page.

Uses PyMuPDF to rasterize PDF pages to images (no external binary deps) and
PaddleOCR to recognize text on each page image.
"""

import io
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from PIL import Image

_ocr_engine = None


def pdf_to_images(pdf_path: Path, dpi: int = 400) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    try:
        return [
            Image.open(io.BytesIO(page.get_pixmap(matrix=matrix).tobytes("png")))
            for page in doc
        ]
    finally:
        doc.close()


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )
    return _ocr_engine


def ocr_image(image: Image.Image) -> str:
    results = _get_engine().predict(np.array(image.convert("RGB")))
    lines: list[str] = []
    for res in results:
        lines.extend(res["rec_texts"])
    return "\n".join(lines)


def vlm_transcribe_image(
    image: Image.Image,
    model_name: str | None = None,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> str:
    """Transcribe a page image via a Modal-hosted VLM (default: DeepSeek-OCR-2).

    Handles handwriting much better than ocr_image (PaddleOCR) at the cost
    of occasional hallucination on illegible strokes. Requires the
    cartridges-vlm Modal app to be deployed (see vlm_app.py).

    Cached in Postgres, keyed on (page image hash, model_id) — identical
    page + same model is a cache hit; a changed source page or a swapped
    model_id (e.g. after fine-tuning) is a cache miss. source_path/
    source_mtime are stored alongside the cache row for auditability, not
    as part of the cache key.
    """
    import modal

    from cartridges.datagen.cache import get_cached, hash_image_bytes, put_cached
    from cartridges.datagen.vlm_app import MODEL_NAME, TRANSCRIBER_CLASSES

    model_name = model_name or MODEL_NAME
    class_name = TRANSCRIBER_CLASSES[model_name]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    page_hash = hash_image_bytes(image_bytes)

    cached = get_cached(page_hash, model_name)
    if cached is not None:
        return cached

    transcriber = modal.Cls.from_name("cartridges-vlm", class_name)
    transcription = transcriber().transcribe.remote(image_bytes)
    put_cached(page_hash, model_name, transcription, source_path, source_mtime)
    return transcription
