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

    def ocr(
        self,
        image_path: str,
        page_number: int = 1,
        hf_revision: str = "",
        dpi: int = 400,
        max_new_tokens: int = 1024,
    ) -> None:
        """Transcribe a single image (or one page of a PDF) via the hosted model.

        image_path: a .png/.jpg file, or a .pdf (page_number selects which page,
        1-indexed). hf_revision: optional Hugging Face Hub revision (e.g.
        "run-<wandb_run_id>" or "date-2026-09-22") in the fine-tuned adapter
        repo (see TrainConfig.hf_repo_id) to use instead of the base model.
        Prints the transcribed text to stdout.
        """
        import base64
        import io

        from cartridges.serve.inference import app, generate
        from cartridges.serve.schema import ChatMessage, GenerateRequest, ImageContent, ServeConfig
        from cartridges.training.config import ModelSource

        path = Path(image_path)
        if path.suffix.lower() == ".pdf":
            from cartridges.processors.ocr import pdf_to_images

            pages = pdf_to_images(path, dpi=dpi)
            if not (1 <= page_number <= len(pages)):
                raise ValueError(f"page_number must be 1..{len(pages)}")
            buf = io.BytesIO()
            pages[page_number - 1].save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = path.read_bytes()

        request = GenerateRequest(
            serve_config=ServeConfig(
                model_source=ModelSource.ARTIFACT if hf_revision else ModelSource.BASE,
                hf_revision=hf_revision or None,
                max_new_tokens=max_new_tokens,
            ),
            messages=[
                ChatMessage(content=[ImageContent(image_base64=base64.b64encode(image_bytes).decode())])
            ],
        )

        with app.run():
            response = generate.remote(request)

        print(response.text)
        logger.info(
            "ocr done: model_source=%s hf_revision=%s total_s=%.1f",
            response.model_source, response.hf_revision, response.total_s,
        )


def main() -> None:
    configure_logging()
    fire.Fire(Cli)
