"""Prefect flows for data generation workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from prefect import flow, task

from cartridges.processors.ocr import OCR_ENGINE_KINDS


@task
def ensure_transcription_cache(engine: OCR_ENGINE_KINDS) -> None:
    del engine
    from cartridges.storage.cache import init_cache

    init_cache()


@task
def read_existing_sections(output_path: Path) -> list[str]:
    from cartridges.datagen.loaders import split_markdown_pages

    if not output_path.exists():
        return []
    return split_markdown_pages(output_path.read_text())


@task
def count_pdf_pages(input_path: Path) -> int:
    from cartridges.datagen.loaders import pdf_page_count

    return pdf_page_count(input_path)


@task
def convert_document(
    input_path: Path,
    *,
    engine: OCR_ENGINE_KINDS = "model",
    model: str = "",
    start_page: int = 1,
) -> str:
    from cartridges.datagen.loaders import convert_file

    return convert_file(
        input_path,
        engine=engine,
        model_name=model or None,
        start_page=start_page,
    )


@task
def merge_markdown_sections(existing_sections: list[str], new_markdown: str) -> str:
    from cartridges.datagen.loaders import split_markdown_pages

    sections = existing_sections + split_markdown_pages(new_markdown)
    return "\n---\n".join(section.rstrip() for section in sections) + "\n"


@task
def assert_section_count(markdown: str, expected_count: int) -> None:
    from cartridges.datagen.loaders import split_markdown_pages

    actual_count = len(split_markdown_pages(markdown))
    if actual_count != expected_count:
        raise ValueError(
            f"expected {expected_count} markdown page sections, found {actual_count}"
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
    engine: OCR_ENGINE_KINDS = "model",
    model: str = "",
    use_agent: bool = False,
) -> None:
    """Convert a source document into markdown.

    Default path is a local Prefect workflow. Set use_agent=True to delegate to
    the Anthropic agent workflow.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    ensure_transcription_cache(engine)

    if use_agent:
        run_agent_prepare_data(input_path, output_path, model)
        return

    if input_path.suffix.lower() == ".pdf":
        existing_sections = read_existing_sections(output_path)
        total_pages = count_pdf_pages(input_path)
        if len(existing_sections) > total_pages:
            raise ValueError(
                f"{output_path} has {len(existing_sections)} sections, "
                f"but {input_path} has only {total_pages} pages"
            )
        new_markdown = convert_document(
            input_path,
            engine=engine,
            model=model,
            start_page=len(existing_sections) + 1,
        )
        markdown = merge_markdown_sections(existing_sections, new_markdown)
        assert_section_count(markdown, total_pages)
    else:
        markdown = convert_document(input_path, engine=engine, model=model)
    write_markdown(output_path, markdown)
