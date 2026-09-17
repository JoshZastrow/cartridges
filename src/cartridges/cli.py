"""Command-line interface for cartridges."""

import logging
from pathlib import Path

import fire

from cartridges.logging.config import configure_logging

logger = logging.getLogger(__name__)


class Cli:
    """cartridges command-line interface."""

    def prepare_data(
        self,
        input_path: str,
        output_path: str,
        model: str = "",
        use_agent: bool = False,
        engine: str = "model",
        dpi: int = 400,
    ) -> None:
        """Convert a source document into a golden-format markdown transcription.

        By default this runs the local Prefect flow. Pass use_agent=True to run
        the Anthropic agent workflow instead.

        model: optional model name to use (e.g. 'prithivMLmods/Qwen2-VL-OCR-2B-Instruct').
        For the non-agent Prefect path, engine='model' uses the cached Modal
        model OCR engine; engine='paddleocr' uses local PaddleOCR.
        """
        from cartridges.datagen.flows import prepare_data_flow

        prepare_data_flow(
            input_path=Path(input_path),
            output_path=Path(output_path),
            engine=engine,
            model=model,
            use_agent=use_agent,
            dpi=dpi,
        )

    def pdf_to_image(
        self, pdf_path: str, page_number: int, output_path: str, dpi: int = 400
    ) -> None:
        """Rasterize a single PDF page to a PNG, without running OCR/transcription."""
        from cartridges.processors.ocr import pdf_to_images

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
