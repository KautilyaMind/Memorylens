from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from src.config import settings
from src.document_manifest import document_path, load_document_manifest
from src.metadata import Document, normalize_metadata

LOGGER = logging.getLogger(__name__)
PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s+(?:of|/)\s+\d+)?$", re.IGNORECASE)
BULLET_RE = re.compile(r"^(?:[•●▪■‒–—*-]|\d+[.)]|[A-Za-z][.)])\s+")


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _repeated_margin_lines(pages: list[list[str]]) -> set[str]:
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        nonempty = [line.strip() for line in lines if line.strip()]
        counts.update(set(_line_key(line) for line in nonempty[:2] + nonempty[-2:]))
    threshold = max(2, (len(pages) + 1) // 2)
    return {line for line, count in counts.items() if count >= threshold and len(line) > 2}


def _is_heading(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    return (
        3 <= len(line) <= 120
        and bool(letters)
        and not line.endswith((".", ",", ";", ":"))
        and sum(char.isupper() for char in letters) / len(letters) >= 0.72
    )


def _clean_page(lines: list[str], repeated: set[str]) -> str:
    kept: list[str] = []
    for raw in lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or _line_key(line) in repeated or PAGE_NUMBER_RE.fullmatch(line):
            if kept and kept[-1] != "":
                kept.append("")
            continue
        kept.append(line)

    paragraphs: list[str] = []
    current = ""
    for line in kept:
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        structural = _is_heading(line) or BULLET_RE.match(line) is not None
        if structural:
            if current:
                paragraphs.append(current)
            paragraphs.append(line)
            current = ""
        elif not current:
            current = line
        elif current.endswith("-") and not current.endswith(" -"):
            current = current[:-1] + line
        else:
            current += " " + line
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs).strip()


def extract_pdf_pages(path: Path, entry: dict[str, Any]) -> list[Document]:
    try:
        with pymupdf.open(path) as pdf:
            raw_pages = [
                page.get_text("text", sort=True).replace("\ufffd", "—").splitlines()
                for page in pdf
            ]
    except Exception as exc:
        raise ValueError(f"Could not parse PDF {path}: {exc}") from exc

    repeated = _repeated_margin_lines(raw_pages)
    metadata = normalize_metadata(entry, str(path.resolve()))
    documents: list[Document] = []
    for page_number, lines in enumerate(raw_pages, start=1):
        text = _clean_page(lines, repeated)
        if text:
            documents.append(
                Document(
                    text,
                    {
                        **metadata,
                        "page": page_number,
                        "content_type": "text",
                        "contains_table": False,
                        "table_title": "",
                    },
                )
            )
    if not documents:
        raise ValueError(f"PDF contains no extractable text: {path}")
    return documents


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _table_title(page: Any, bbox: tuple[float, float, float, float], headers: list[str]) -> str:
    candidates: list[tuple[float, str]] = []
    for block in page.get_text("blocks", sort=True):
        if len(block) < 5 or float(block[3]) > float(bbox[1]) + 2:
            continue
        lines = [line.strip() for line in str(block[4]).splitlines() if line.strip()]
        if lines:
            candidates.append((float(block[3]), lines[-1]))
    if candidates:
        candidate = max(candidates, key=lambda item: item[0])[1]
        if 3 <= len(candidate) <= 160:
            return candidate
    return " | ".join(header for header in headers if header)[:160]


def _serialize_table(
    rows: list[list[Any]],
    title: str,
    document_title: str,
    page_number: int,
) -> str:
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return ""
    cleaned = [[_clean_cell(row[index]) if index < len(row) else "" for index in range(width)] for row in rows]
    headers = [cell or f"Column {index + 1}" for index, cell in enumerate(cleaned[0])]
    body = cleaned[1:]
    if not any(any(cell for cell in row) for row in body):
        return ""
    lines = [
        f"Document: {document_title}",
        f"Table: {title or 'Untitled table'}",
        f"Page: {page_number}",
        "Columns: " + " | ".join(headers),
    ]
    for row in body:
        values = [
            f"{headers[index]}: {cell}"
            for index, cell in enumerate(row)
            if cell
        ]
        if values:
            lines.append(" | ".join(values))
    return "\n".join(lines).strip()


def extract_pdf_tables(path: Path, entry: dict[str, Any]) -> list[Document]:
    metadata = normalize_metadata(entry, str(path.resolve()))
    documents: list[Document] = []
    table_index = 0
    try:
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                try:
                    tables = page.find_tables().tables
                except Exception as exc:
                    LOGGER.warning(
                        "Table inspection failed for %s page %d: %s",
                        path.name,
                        page_number,
                        exc,
                    )
                    continue
                for table in tables:
                    try:
                        rows = table.extract()
                        if not rows:
                            continue
                        header_names = [
                            _clean_cell(value) for value in getattr(table.header, "names", [])
                        ]
                        title = _table_title(page, table.bbox, header_names)
                        text = _serialize_table(
                            rows, title, str(entry["title"]), page_number
                        )
                        if len(text) < 40:
                            continue
                        table_index += 1
                        documents.append(
                            Document(
                                text,
                                {
                                    **metadata,
                                    "page": page_number,
                                    "content_type": "table",
                                    "contains_table": True,
                                    "table_title": title,
                                    "table_index": table_index,
                                },
                            )
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "Skipping unreliable table in %s page %d: %s",
                            path.name,
                            page_number,
                            exc,
                        )
    except Exception as exc:
        LOGGER.warning("Could not inspect tables in %s: %s", path, exc)
    return documents


def load_corpus(
    corpus_dir: Path = settings.corpus_dir,
    enable_table_extraction: bool = settings.enable_table_extraction,
) -> list[Document]:
    documents: list[Document] = []
    for entry in load_document_manifest():
        path = document_path(entry, corpus_dir)
        if not path.exists():
            LOGGER.warning("Skipping missing configured PDF: %s", path)
            continue
        LOGGER.info("Loading %s", path.name)
        documents.extend(extract_pdf_pages(path, entry))
        if enable_table_extraction:
            tables = extract_pdf_tables(path, entry)
            LOGGER.info("Extracted %d tables from %s", len(tables), path.name)
            documents.extend(tables)
    return documents
