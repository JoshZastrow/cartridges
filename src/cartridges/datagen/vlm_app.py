"""Compatibility wrapper for the old Modal OCR app module."""

from cartridges.processors.ocr import (
    MODEL_NAME,
    QWEN_OCR_MODEL_NAME,
    TRANSCRIBER_CLASSES,
    QwenOCRTranscriber,
    Transcriber,
    app,
)

__all__ = [
    "MODEL_NAME",
    "QWEN_OCR_MODEL_NAME",
    "TRANSCRIBER_CLASSES",
    "QwenOCRTranscriber",
    "Transcriber",
    "app",
]
