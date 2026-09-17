"""Per-format document -> markdown conversion.

Bodies are stubs: no conversion library has been chosen yet (markitdown vs.
unstructured vs. custom). Fill these in once that's decided.
"""

from pathlib import Path

from cartridges.processors.ocr import OCR_ENGINE_KINDS, pdf_to_images, transcribe_image


def pdf_to_markdown(
    path: Path,
    *,
    engine: OCR_ENGINE_KINDS = "paddleocr",
    model_name: str | None = None,
) -> str:
    pages = [
        transcribe_image(
            image,
            engine=engine,
            model_name=model_name,
            source_path=str(path),
            source_mtime=path.stat().st_mtime,
        )
        for image in pdf_to_images(path)
    ]
    return "\n\n---\n\n".join(pages)


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
    engine: OCR_ENGINE_KINDS = "paddleocr",
    model_name: str | None = None,
) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"no loader registered for {path.suffix!r}")
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path, engine=engine, model_name=model_name)
    return loader(path)
