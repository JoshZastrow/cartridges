"""Compatibility wrapper for the old datagen OCR module."""

from cartridges.processors.ocr import (
    MODEL_NAME,
    OCR_ENGINE_KINDS,
    PADDLE_OCR_MODEL_ID,
    QWEN_OCR_MODEL_NAME,
    OcrEngine,
    create_ocr_engine,
    model_transcribe_image,
    ocr_image,
    pdf_to_images,
    transcribe_image,
    vlm_transcribe_image,
)

__all__ = [
    "MODEL_NAME",
    "OCR_ENGINE_KINDS",
    "PADDLE_OCR_MODEL_ID",
    "QWEN_OCR_MODEL_NAME",
    "OcrEngine",
    "create_ocr_engine",
    "model_transcribe_image",
    "ocr_image",
    "pdf_to_images",
    "transcribe_image",
    "vlm_transcribe_image",
]
