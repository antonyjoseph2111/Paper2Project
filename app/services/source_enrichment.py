from __future__ import annotations

import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from app.core.config import settings


def detect_arxiv_id(filename: str) -> str | None:
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", filename)
    if match:
        return match.group(1)
    return None


def maybe_download_arxiv_source(pdf_path: Path) -> Path | None:
    if not settings.arxiv_source_enabled:
        return None
    arxiv_id = detect_arxiv_id(pdf_path.name)
    if not arxiv_id:
        return None
    output_path = pdf_path.parent / f"{arxiv_id.replace('/', '_')}_source.tar"
    if output_path.exists():
        return output_path
    response = httpx.get(f"https://arxiv.org/e-print/{arxiv_id}", timeout=settings.llm_timeout_seconds)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def extract_arxiv_source(archive_path: Path) -> Path | None:
    extract_dir = archive_path.parent / "arxiv_source"
    if extract_dir.exists():
        return extract_dir
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_dir)
        return extract_dir
    except tarfile.TarError:
        return None


def maybe_fetch_grobid(pdf_path: Path) -> Path | None:
    if not settings.grobid_url:
        return None
    output_path = pdf_path.parent / "grobid_tei.xml"
    if output_path.exists():
        return output_path
    with pdf_path.open("rb") as handle:
        files = {"input": (pdf_path.name, handle, "application/pdf")}
        response = httpx.post(
            settings.grobid_url.rstrip("/") + "/api/processFulltextDocument",
            files=files,
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def parse_grobid_tei(tei_path: Path) -> dict[str, str]:
    namespaces = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei_path.read_text(encoding="utf-8"))
    extracted: dict[str, str] = {}
    title = root.findtext(".//tei:titleStmt/tei:title", default="", namespaces=namespaces)
    if title:
        extracted["title"] = " ".join(title.split())
    abstract_parts = root.findall(".//tei:abstract//tei:p", namespaces)
    if abstract_parts:
        extracted["abstract"] = " ".join(" ".join(part.itertext()).strip() for part in abstract_parts)
    body_sections = []
    for div in root.findall(".//tei:body//tei:div", namespaces):
        head = div.findtext("./tei:head", default="", namespaces=namespaces).strip()
        text = " ".join(" ".join(node.itertext()).strip() for node in div.findall(".//tei:p", namespaces)).strip()
        if head and text:
            body_sections.append(f"{head}\n{text}")
    if body_sections:
        extracted["body"] = "\n\n".join(body_sections)
    return extracted


def collect_tex_text(source_dir: Path) -> str:
    chunks: list[str] = []
    for path in list(source_dir.rglob("*.tex"))[:30]:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)
