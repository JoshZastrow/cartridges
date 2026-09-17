"""Per-format document -> markdown conversion.

Bodies are stubs: no conversion library has been chosen yet (markitdown vs.
unstructured vs. custom). Fill these in once that's decided.
"""

from pathlib import Path
import re

from cartridges.processors.ocr import (
    OCR_ENGINE_KINDS,
    pdf_page_count as _pdf_page_count,
    pdf_page_to_image,
    pdf_to_images,
    transcribe_image,
)


def split_markdown_pages(markdown: str) -> list[str]:
    return [
        section.strip()
        for section in re.split(r"(?m)^\s*---\s*$", markdown)
        if section.strip()
    ]


def sanitize_page_text(text: str) -> str:
    return re.sub(r"(?m)^\s*---\s*$", "- - -", text.strip())


def format_pdf_page(page_number: int, text: str, date: str = "unknown") -> str:
    return f"## p. {page_number} - date: {date}\n\n{sanitize_page_text(text)}\n"


def pdf_page_count(path: Path) -> int:
    return _pdf_page_count(path)


def pdf_page_to_markdown(
    path: Path,
    page_number: int,
    *,
    engine: OCR_ENGINE_KINDS = "model",
    model_name: str | None = None,
    dpi: int = 400,
) -> str:
    image = pdf_page_to_image(path, page_number, dpi=dpi)
    text = transcribe_image(
        image,
        engine=engine,
        model_name=model_name,
        source_path=str(path),
        source_mtime=path.stat().st_mtime,
    )
    return format_pdf_page(page_number, text)


def pdf_to_markdown(
    path: Path,
    *,
    engine: OCR_ENGINE_KINDS = "model",
    model_name: str | None = None,
    start_page: int = 1,
    dpi: int = 400,
) -> str:
    if start_page < 1:
        raise ValueError("start_page must be 1 or greater")

    pages = [
        format_pdf_page(
            page_number,
            transcribe_image(
                image,
                engine=engine,
                model_name=model_name,
                source_path=str(path),
                source_mtime=path.stat().st_mtime,
            ),
        )
        for page_number, image in enumerate(pdf_to_images(path, dpi=dpi), start=1)
        if page_number >= start_page
    ]
    return "\n---\n".join(pages)


def docx_to_markdown(path: Path) -> str:
    raise NotImplementedError


def html_to_markdown(path: Path) -> str:
    raise NotImplementedError


def txt_to_markdown(path: Path) -> str:
    return path.read_text()


def image_to_markdown(path: Path) -> str:
    raise NotImplementedError


LOADERS = {
    ".pdf": pdf_to_markdown,
    ".docx": docx_to_markdown,
    ".html": html_to_markdown,
    ".htm": html_to_markdown,
    ".txt": txt_to_markdown,
    ".png": image_to_markdown,
    ".jpg": image_to_markdown,
    ".jpeg": image_to_markdown,
}


def convert_file(
    path: Path,
    *,
    engine: OCR_ENGINE_KINDS = "model",
    model_name: str | None = None,
    start_page: int = 1,
    dpi: int = 250,
) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"no loader registered for {path.suffix!r}")
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(
            path,
            engine=engine,
            model_name=model_name,
            start_page=start_page,
            dpi=dpi,
        )
    return loader(path)
