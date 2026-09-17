"""Agent-facing tools for the prepare-data skill: page-granular PDF OCR + file write."""

import logging
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from cartridges.datagen.ocr import pdf_to_images, vlm_transcribe_image

logger = logging.getLogger(__name__)

_PAGE_CACHE: dict[str, list] = {}
_LAST_DATE: dict[str, str] = {}
_INITIALIZED: set[str] = set()


def _pages(pdf_path: str) -> list:
    if pdf_path not in _PAGE_CACHE:
        _PAGE_CACHE[pdf_path] = pdf_to_images(Path(pdf_path))
    return _PAGE_CACHE[pdf_path]


@tool("list_pdf_pages", "Return the number of pages in a PDF file.", {"pdf_path": str})
async def list_pdf_pages(args):
    return {"content": [{"type": "text", "text": str(len(_pages(args["pdf_path"])))}]}


@tool(
    "pdf_page_to_image",
    "Rasterize one page (1-indexed) of a PDF to a PNG file and return the output path.",
    {"pdf_path": str, "page_number": int, "output_path": str},
)
async def pdf_page_to_image(args):
    pages = _pages(args["pdf_path"])
    page_number = args["page_number"]
    if not (1 <= page_number <= len(pages)):
        return {
            "content": [{"type": "text", "text": f"page_number must be 1..{len(pages)}"}],
            "is_error": True,
        }
    output_path = Path(args["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages[page_number - 1].save(output_path)
    return {"content": [{"type": "text", "text": f"wrote {output_path}"}]}


@tool(
    "ocr_pdf_page",
    "Run OCR on one page (1-indexed) of a PDF and return the raw recognized text. "
    "Pass model='' to use the default model, or an explicit model name (e.g. "
    "'prithivMLmods/Qwen2-VL-OCR-2B-Instruct') to use a different deployed model.",
    {"pdf_path": str, "page_number": int, "model": str},
)
async def ocr_pdf_page(args):
    pdf_path = args["pdf_path"]
    pages = _pages(pdf_path)
    page_number = args["page_number"]
    if not (1 <= page_number <= len(pages)):
        return {
            "content": [{"type": "text", "text": f"page_number must be 1..{len(pages)}"}],
            "is_error": True,
        }
    text = vlm_transcribe_image(
        pages[page_number - 1],
        model_name=args["model"].strip() or None,
        source_path=pdf_path,
        source_mtime=Path(pdf_path).stat().st_mtime,
    )
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "append_page",
    "Append one transcribed page to the output markdown file, in page order. "
    "Formats it as '## p. N - date: D' followed by the text, with pages "
    "separated by a '---' line. Pass date='' if no date is visible on this "
    "page — the most recently seen date for this output file is carried "
    "forward automatically (or 'unknown' if none seen yet). Call once per "
    "page, in increasing page order; the first call (re)creates the file.",
    {"output_path": str, "page_number": int, "date": str, "text": str},
)
async def append_page(args):
    output_path = args["output_path"]
    date = args["date"].strip()
    if date:
        _LAST_DATE[output_path] = date
    else:
        date = _LAST_DATE.get(output_path, "unknown")

    block = f"## p. {args['page_number']} - date: {date}\n\n{args['text']}\n"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_path not in _INITIALIZED:
        path.write_text(block)
        _INITIALIZED.add(output_path)
    else:
        with path.open("a") as f:
            f.write("\n---\n" + block)

    total = len(next(iter(_PAGE_CACHE.values()), []))
    total_str = str(total) if total else "?"
    logger.info(
        "%s: page %s/%s appended (date: %s)",
        output_path,
        args["page_number"],
        total_str,
        date,
    )

    return {
        "content": [
            {"type": "text", "text": f"appended page {args['page_number']} (date: {date})"}
        ]
    }


prepare_data_server = create_sdk_mcp_server(
    name="prepare-data",
    version="1.0.0",
    tools=[list_pdf_pages, pdf_page_to_image, ocr_pdf_page, append_page],
)
