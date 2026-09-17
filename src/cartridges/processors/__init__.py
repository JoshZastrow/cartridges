"""Reusable processors for data generation and other workflows."""

from cartridges.processors.ocr import (
    MODEL_NAME,
    OCR_ENGINE_KINDS,
    PADDLE_OCR_MODEL_ID,
    QWEN_OCR_MODEL_NAME,
    OcrEngine,
    create_ocr_engine,
    ocr_image,
    pdf_to_images,
    transcribe_image,
)

__all__ = [
    "MODEL_NAME",
    "OCR_ENGINE_KINDS",
    "PADDLE_OCR_MODEL_ID",
    "QWEN_OCR_MODEL_NAME",
    "OcrEngine",
    "create_ocr_engine",
    "ocr_image",
    "pdf_to_images",
    "transcribe_image",
]
