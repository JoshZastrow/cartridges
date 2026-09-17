"""Prefect flows for data generation workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from prefect import flow, task

from cartridges.processors.ocr import OCR_ENGINE_KINDS


@task
def convert_document(
    input_path: Path,
    *,
    engine: OCR_ENGINE_KINDS = "paddleocr",
    model: str = "",
) -> str:
    from cartridges.datagen.loaders import convert_file

    return convert_file(
        input_path,
        engine=engine,
        model_name=model or None,
    )


@task
def write_markdown(output_path: Path, markdown: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)


@task
def run_agent_prepare_data(input_path: Path, output_path: Path, model: str = "") -> None:
    from cartridges.datagen.agent.pipeline import prepare_data

    asyncio.run(prepare_data(input_path, output_path, model))


@flow(name="prepare-data")
def prepare_data_flow(
    input_path: Path | str,
    output_path: Path | str,
    *,
    engine: OCR_ENGINE_KINDS = "paddleocr",
    model: str = "",
    use_agent: bool = False,
) -> None:
    """Convert a source document into markdown.

    Default path is a local Prefect workflow. Set use_agent=True to delegate to
    the Anthropic agent workflow.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if use_agent:
        run_agent_prepare_data(input_path, output_path, model)
        return

    markdown = convert_document(input_path, engine=engine, model=model)
    write_markdown(output_path, markdown)
