"""Command-line interface for cartridges."""

import asyncio
import logging
from pathlib import Path

import fire

from cartridges.logging_config import configure_logging

logger = logging.getLogger(__name__)


class Cli:
    """cartridges command-line interface."""

    def prepare_data(self, input_path: str, output_path: str, model: str = "") -> None:
        """Convert a source document into a golden-format markdown transcription.

        model: optional model name to use (e.g. 'prithivMLmods/Qwen2-VL-OCR-2B-Instruct').
        Defaults to deepseek-ai/DeepSeek-OCR-2.
        """
        from cartridges.datagen.agent.pipeline import prepare_data

        asyncio.run(prepare_data(Path(input_path), Path(output_path), model))

    def pdf_to_image(self, pdf_path: str, page_number: int, output_path: str, dpi: int = 400) -> None:
        """Rasterize a single PDF page to a PNG, without running OCR/transcription."""
        from cartridges.datagen.ocr import pdf_to_images

        pages = pdf_to_images(Path(pdf_path), dpi=dpi)
        if not (1 <= page_number <= len(pages)):
            raise ValueError(f"page_number must be 1..{len(pages)}")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pages[page_number - 1].save(output)
        logger.info("wrote %s", output)


def main() -> None:
    configure_logging()
    fire.Fire(Cli)
