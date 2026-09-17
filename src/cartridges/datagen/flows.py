"""Prefect flows for data generation workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

from prefect import flow, get_run_logger, task

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
def transcribe_pdf_page(
    input_path: Path,
    page_number: int,
    total_pages: int,
    *,
    engine: OCR_ENGINE_KINDS = "model",
    model: str = "",
    dpi: int = 400,
) -> str:
    from cartridges.datagen.loaders import pdf_page_to_markdown

    logger = get_run_logger()
    logger.info("Transcribing page %s/%s with %s", page_number, total_pages, engine)
    section = pdf_page_to_markdown(
        input_path,
        page_number,
        engine=engine,
        model_name=model or None,
        dpi=dpi,
    )
    logger.info("Finished page %s/%s", page_number, total_pages)
    return section


@task
def convert_document(
    input_path: Path,
    *,
    engine: OCR_ENGINE_KINDS = "model",
    model: str = "",
    start_page: int = 1,
    dpi: int = 400,
) -> str:
    from cartridges.datagen.loaders import convert_file

    return convert_file(
        input_path,
        engine=engine,
        model_name=model or None,
        start_page=start_page,
        dpi=dpi,
    )


@task
def merge_markdown_sections(existing_sections: list[str], new_markdown: str) -> str:
    from cartridges.datagen.loaders import split_markdown_pages

    sections = existing_sections + split_markdown_pages(new_markdown)
    return "\n---\n".join(section.rstrip() for section in sections) + "\n"


@task
def append_markdown_section(output_path: Path, section: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n---\n" if output_path.exists() and output_path.read_text().strip() else ""
    with output_path.open("a") as file:
        file.write(separator)
        file.write(section.rstrip())
        file.write("\n")


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
    dpi: int = 250,
) -> None:
    """Convert a source document into markdown.

    Default path is a local Prefect workflow. Set use_agent=True to delegate to
    the Anthropic agent workflow.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    logger = get_run_logger()

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
        logger.info(
            "%s has %s/%s page sections before processing",
            output_path,
            len(existing_sections),
            total_pages,
        )
        for page_number in range(len(existing_sections) + 1, total_pages + 1):
            section = transcribe_pdf_page(
                input_path,
                page_number,
                total_pages,
                engine=engine,
                model=model,
                dpi=dpi,
            )
            append_markdown_section(output_path, section)
            checkpoint_sections = read_existing_sections(output_path)
            if len(checkpoint_sections) != page_number:
                raise ValueError(
                    f"expected {page_number} sections after page {page_number}, "
                    f"found {len(checkpoint_sections)}"
                )
        final_sections = read_existing_sections(output_path)
        if len(final_sections) != total_pages:
            raise ValueError(
                f"expected {total_pages} markdown page sections, "
                f"found {len(final_sections)}"
            )
        logger.info("Finished %s with %s/%s sections", output_path, total_pages, total_pages)
        return
    else:
        markdown = convert_document(input_path, engine=engine, model=model, dpi=dpi)
    write_markdown(output_path, markdown)
