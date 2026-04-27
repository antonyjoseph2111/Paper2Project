from __future__ import annotations

import re
from pathlib import Path

import fitz

from app.models.schemas import ParsedPaper, SectionChunk


SECTION_NAMES = [
    "abstract",
    "introduction",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "experiments",
    "results",
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"references\s.*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_sections(text: str) -> list[SectionChunk]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    chunks: list[SectionChunk] = []
    current_name = "full_text"
    current_lines: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered in SECTION_NAMES or any(lowered.startswith(f"{prefix}.") for prefix in SECTION_NAMES):
            if current_lines:
                chunks.append(SectionChunk(name=current_name, content=_clean_text(" ".join(current_lines))))
            current_name = lowered
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append(SectionChunk(name=current_name, content=_clean_text(" ".join(current_lines))))
    return chunks


def parse_pdf(pdf_path: Path) -> ParsedPaper:
    document = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in document]
    full_text = "\n".join(pages)
    sections = _extract_sections(full_text)

    title = ""
    first_page_lines = [line.strip() for line in pages[0].splitlines() if line.strip()] if pages else []
    if first_page_lines:
        title = first_page_lines[0]

    abstract = next((s.content for s in sections if "abstract" in s.name), "")
    introduction = next((s.content for s in sections if "introduction" in s.name), "")
    methodology = next((s.content for s in sections if s.name in {"method", "methods", "methodology", "approach"}), "")
    model_description = next((s.content for s in sections if "model" in s.name), methodology)

    equations = re.findall(r"[^.]*=[^.]*", full_text)
    keywords = sorted({word.lower() for word in re.findall(r"\b(transformer|attention|classification|regression|language|vision)\b", full_text, flags=re.IGNORECASE)})

    return ParsedPaper(
        title=title,
        problem=abstract[:400],
        abstract=_clean_text(abstract),
        introduction=_clean_text(introduction),
        methodology_text=_clean_text(methodology),
        model_description=_clean_text(model_description),
        equations=[_clean_text(eq) for eq in equations[:20]],
        keywords=keywords,
        sections=sections,
    )
