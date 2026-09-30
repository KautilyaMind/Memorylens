from __future__ import annotations

import re
from collections import defaultdict

from src.metadata import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
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


def _markdown_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", "", text)]
    sections: list[tuple[str, str, str]] = []
    headings: dict[int, str] = {}
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(("", "", text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        name = match.group(2).strip()
        headings[level] = name
        for deeper in [key for key in headings if key > level]:
            del headings[deeper]
        section = headings.get(2) or headings.get(1) or name
        subsection = name if level >= 3 else ""
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip()
        if content:
            sections.append((section, subsection, content))
    return sections


def _obvious_pdf_heading(text: str) -> str:
    for line in text.splitlines()[:12]:
        candidate = line.strip()
        if 3 <= len(candidate) <= 100 and not candidate.endswith("."):
            letters = [char for char in candidate if char.isalpha()]
            if letters and sum(char.isupper() for char in letters) / len(letters) > 0.75:
                return candidate.title()
    return ""


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be in [0, chunk_size)")
    pending: list[tuple[str, dict]] = []
    for document in documents:
        if document.metadata.get("source_type") == "webpage":
            for section, subsection, text in _markdown_sections(document.text):
                for piece in _split_text(text, size, overlap):
                    pending.append((piece, {**document.metadata, "section": section, "subsection": subsection}))
        else:
            section = _obvious_pdf_heading(document.text)
            for piece in _split_text(document.text, size, overlap):
                pending.append((piece, {**document.metadata, "section": section, "subsection": ""}))

    counters: defaultdict[str, int] = defaultdict(int)
    chunks: list[Chunk] = []
    for text, metadata in pending:
        document_id = str(metadata["document_id"])
        counters[document_id] += 1
        metadata = {
            **metadata,
            "chunk_id": f"{document_id}_CH_{counters[document_id]:04d}",
            "page": int(metadata.get("page", 0) or 0),
            "content_type": "table" if "|" in text and "---" in text else "text",
            "contains_numeric_specs": bool(NUMERIC_SPEC_RE.search(text)),
            "contains_part_number": bool(PART_NUMBER_RE.search(text)),
            "contains_table": bool("|" in text and "---" in text),
        }
        chunks.append(Chunk(text, metadata))
    return chunks
