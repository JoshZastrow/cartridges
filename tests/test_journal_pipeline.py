from pathlib import Path

import pytest

from cartridges.datagen.loaders import (
    format_pdf_page,
    pdf_page_count,
    split_markdown_pages,
)


MORNING_PAGES_PDF = Path("data/journals/inputs/Morning Pages.pdf")
MORNING_PAGES_OUTPUT = Path("data/journals/output/Morning Pages.md")


def test_morning_pages_output_has_one_section_per_pdf_page():
    if not MORNING_PAGES_PDF.exists() or not MORNING_PAGES_OUTPUT.exists():
        pytest.skip("Morning Pages fixture files are not present")

    expected_pages = pdf_page_count(MORNING_PAGES_PDF)
    output_sections = split_markdown_pages(MORNING_PAGES_OUTPUT.read_text())

    assert len(output_sections) == expected_pages


def test_format_pdf_page_escapes_standalone_delimiter_lines():
    page = format_pdf_page(1, "before\n---\nafter")

    assert "- - -" in page
    assert len(split_markdown_pages(page)) == 1
