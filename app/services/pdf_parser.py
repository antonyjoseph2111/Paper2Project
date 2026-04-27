from __future__ import annotations

import re
from pathlib import Path

import fitz

from app.core.config import settings
from app.models.schemas import ParsedPaper, SectionChunk
from app.services.source_enrichment import collect_tex_text, parse_grobid_tei


SECTION_PATTERNS = {
    "abstract": re.compile(r"^(abstract)\b", re.IGNORECASE),
    "introduction": re.compile(r"^(\d+\.?\s*)?introduction\b", re.IGNORECASE),
    "methodology": re.compile(r"^(\d+\.?\s*)?(method|methods|methodology|approach)\b", re.IGNORECASE),
    "model": re.compile(r"^(\d+\.?\s*)?(model|architecture)\b", re.IGNORECASE),
    "experiments": re.compile(r"^(\d+\.?\s*)?(experiments?|evaluation|results)\b", re.IGNORECASE),
}


def _clean_text(text: str) -> str:
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?is)\nreferences\b.*$", "", text)
    return text.strip()


def _detect_section_name(line: str) -> str | None:
    stripped = line.strip()
    for name, pattern in SECTION_PATTERNS.items():
        if pattern.match(stripped):
            return name
    subsection_patterns = {
        "model": re.compile(r"^(\d+(?:\.\d+)+\s+)?(model|architecture|network)\b", re.IGNORECASE),
        "methodology": re.compile(r"^(\d+(?:\.\d+)+\s+)?(training|optimization|method|approach)\b", re.IGNORECASE),
        "experiments": re.compile(r"^(\d+(?:\.\d+)+\s+)?(evaluation|results|ablation|experiment)\b", re.IGNORECASE),
    }
    for name, pattern in subsection_patterns.items():
        if pattern.match(stripped):
            return name
    return None


def _chunk_text(name: str, text: str, page_start: int, page_end: int) -> list[SectionChunk]:
    chunks: list[SectionChunk] = []
    clean = _clean_text(text)
    if not clean:
        return chunks
    max_chars = settings.max_section_chunk_chars
    start = 0
    index = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            split = clean.rfind(" ", start, end)
            if split > start + (max_chars // 2):
                end = split
        chunk_text = clean[start:end].strip()
        chunks.append(
            SectionChunk(
                name=name,
                content=chunk_text,
                chunk_id=f"{name}_{index}",
                source_page_start=page_start,
                source_page_end=page_end,
            )
        )
        index += 1
        start = end
    return chunks


def _extract_sections(pages: list[str]) -> list[SectionChunk]:
    raw_sections: list[tuple[str, list[str], int, int]] = []
    current_name = "front_matter"
    current_lines: list[str] = []
    current_page_start = 1
    current_page_end = 1

    for page_number, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            detected = _detect_section_name(stripped)
            if detected:
                if current_lines:
                    raw_sections.append((current_name, current_lines[:], current_page_start, current_page_end))
                current_name = detected
                current_lines = []
                current_page_start = page_number
            else:
                current_lines.append(stripped)
            current_page_end = page_number
    if current_lines:
        raw_sections.append((current_name, current_lines[:], current_page_start, current_page_end))

    chunks: list[SectionChunk] = []
    for name, lines, page_start, page_end in raw_sections:
        chunks.extend(_chunk_text(name, "\n".join(lines), page_start, page_end))
    return chunks


def _extract_equations(text: str) -> list[str]:
    equations: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        if len(candidate) < 8 or len(candidate) > 200:
            continue
        has_math = any(symbol in candidate for symbol in ["=", "+", "-", "*", "/", "^", "sum", "softmax", "argmax"])
        has_alpha = bool(re.search(r"[A-Za-z]", candidate))
        has_mathish_structure = bool(re.search(r"[A-Za-z0-9_]+\s*=\s*.+", candidate))
        if has_math and has_alpha and has_mathish_structure:
            equations.append(candidate)
    return list(dict.fromkeys(equations))[:20]


def _extract_keywords(text: str) -> list[str]:
    vocab = [
        "transformer",
        "attention",
        "classification",
        "regression",
        "vision",
        "language",
        "reinforcement learning",
        "segmentation",
        "generation",
        "diffusion",
        "cnn",
        "gan",
    ]
    lowered = text.lower()
    return [keyword for keyword in vocab if keyword in lowered]


def parse_pdf(pdf_path: Path, grobid_tei_path: Path | None = None, arxiv_source_dir: Path | None = None) -> ParsedPaper:
    document = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in document]
    enriched_title = ""
    enriched_abstract = ""
    enriched_body = ""
    if grobid_tei_path and grobid_tei_path.exists():
        tei_data = parse_grobid_tei(grobid_tei_path)
        enriched_title = tei_data.get("title", "")
        enriched_abstract = tei_data.get("abstract", "")
        enriched_body = tei_data.get("body", "")
    tex_text = collect_tex_text(arxiv_source_dir) if arxiv_source_dir and arxiv_source_dir.exists() else ""
    full_text = _clean_text("\n".join(filter(None, pages + [enriched_body, tex_text])))
    sections = _extract_sections(pages)
    if enriched_body:
        sections.extend(_chunk_text("grobid_body", enriched_body, 1, len(pages) or 1))
    if tex_text:
        sections.extend(_chunk_text("latex_source", tex_text, 1, len(pages) or 1))

    first_page_lines = [line.strip() for line in pages[0].splitlines() if line.strip()] if pages else []
    title = enriched_title or (first_page_lines[0] if first_page_lines else pdf_path.stem)

    def join_named(name: str) -> str:
        return _clean_text(" ".join(section.content for section in sections if section.name == name))

    abstract = join_named("abstract")
    introduction = join_named("introduction")
    methodology = join_named("methodology")
    model_description = join_named("model") or methodology
    abstract = enriched_abstract or abstract
    if not abstract and first_page_lines:
        abstract = _clean_text(" ".join(first_page_lines[1:8]))

    figure_captions = []
    for line in full_text.splitlines():
        if re.match(r"^(fig(?:ure)?\.?\s*\d+)", line.strip(), flags=re.IGNORECASE):
            figure_captions.append(line.strip())

    equations = _extract_equations("\n".join(pages))
    keywords = _extract_keywords(full_text)

    source_kind = "pdf"
    if grobid_tei_path and grobid_tei_path.exists() and arxiv_source_dir and arxiv_source_dir.exists():
        source_kind = "pdf+grobid+latex"
    elif grobid_tei_path and grobid_tei_path.exists():
        source_kind = "pdf+grobid"
    elif arxiv_source_dir and arxiv_source_dir.exists():
        source_kind = "pdf+latex"

    return ParsedPaper(
        title=title,
        problem=abstract[:400] if abstract else methodology[:400],
        abstract=abstract,
        introduction=introduction,
        methodology_text=methodology,
        model_description=model_description,
        equations=equations,
        keywords=keywords,
        figures=figure_captions[:15],
        sections=sections,
        chunk_count=len(sections),
        source_kind=source_kind,
    )
