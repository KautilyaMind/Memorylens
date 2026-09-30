from __future__ import annotations

import re
from collections import defaultdict

from src.metadata import Chunk, Document

PART_NUMBER_RE = re.compile(r"\b(?:MT|MTA|MTFD)[A-Z0-9-]{5,}\b", re.IGNORECASE)
NUMERIC_SPEC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:GB|Gb|TB|MHz|GHz|MT/s|MTPS|MB/s|GB/s|TB/s|V|mV|W|mW|ns|µs|IOPS)\b",
    re.IGNORECASE,
)


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        upper = min(start + size, len(text))
        end = upper
        if upper < len(text):
            candidates = [text.rfind(marker, start + size // 2, upper) for marker in ("\n\n", ". ", "\n", " ")]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] in {"\n\n", ". "} else 1)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return chunks


def _looks_like_heading(paragraph: str) -> bool:
    if "\n" in paragraph or not 3 <= len(paragraph) <= 120:
        return False
    letters = [char for char in paragraph if char.isalpha()]
    return bool(letters) and not paragraph.endswith(".") and (
        sum(char.isupper() for char in letters) / len(letters) >= 0.72
    )


def _pdf_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    for paragraph in (part.strip() for part in text.split("\n\n")):
        if not paragraph:
            continue
        if _looks_like_heading(paragraph):
            if body:
                sections.append((heading, "\n\n".join(body)))
                body = []
            heading = paragraph.title()
        else:
            body.append(paragraph)
    if body:
        sections.append((heading, "\n\n".join(body)))
    return sections or [("", text)]


def _contains_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return sum(bool(re.search(r"\S\s{2,}\S", line)) for line in lines) >= 2


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be in [0, chunk_size)")
    pending: list[tuple[str, dict]] = []
    for document in documents:
        for section, text in _pdf_sections(document.text):
            for piece in _split_text(text, size, overlap):
                pending.append(
                    (piece, {**document.metadata, "section": section, "subsection": ""})
                )

    counters: defaultdict[str, int] = defaultdict(int)
    chunks: list[Chunk] = []
    for text, metadata in pending:
        document_id = str(metadata["document_id"])
        counters[document_id] += 1
        contains_table = _contains_table(text)
        metadata = {
            **metadata,
            "chunk_id": f"{document_id}_CH_{counters[document_id]:04d}",
            "page": int(metadata.get("page", 0) or 0),
            "content_type": "table" if contains_table else "text",
            "contains_numeric_specs": bool(NUMERIC_SPEC_RE.search(text)),
            "contains_part_number": bool(PART_NUMBER_RE.search(text)),
            "contains_table": contains_table,
        }
        chunks.append(Chunk(text, metadata))
    return chunks
