from pathlib import Path

import fitz

from app.services.pdf_parser import parse_pdf


def test_parse_pdf_chunks_and_equations(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Sample Paper\nAbstract\nThis paper studies classification.\n3.1 Introduction\nWe classify text.\n4.2 Methodology\nx = W y + b\n",
    )
    document.save(pdf_path)
    document.close()

    parsed = parse_pdf(pdf_path)
    assert parsed.title == "Sample Paper"
    assert parsed.sections
    assert parsed.chunk_count == len(parsed.sections)
    assert any(section.name == "introduction" for section in parsed.sections)
    assert any(section.name == "methodology" for section in parsed.sections)
    assert any("=" in equation for equation in parsed.equations)
