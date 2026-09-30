from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUIRED_DOCUMENT_FIELDS = {
    "document_id",
    "title",
    "file_name",
    "category",
    "product_family",
    "technology",
    "document_type",
    "source",
    "source_url",
}


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_metadata(raw: dict[str, Any], file_path: str) -> dict[str, Any]:
    """Return JSON-safe document metadata with a stable field set."""
    return {
        "document_id": str(raw.get("document_id", "")),
        "title": str(raw.get("title", "Untitled")),
        "file_name": str(raw.get("file_name", "")),
        "source_url": str(raw.get("source_url", "")),
        "source": str(raw.get("source", "Micron Technology")),
        "category": str(raw.get("category", "UNKNOWN")).upper(),
        "product_family": str(raw.get("product_family", "")),
        "technology": str(raw.get("technology", "")),
        "document_type": str(raw.get("document_type", "")),
        "file_path": file_path,
    }


def validate_document_entry(document: dict[str, Any]) -> None:
    missing = REQUIRED_DOCUMENT_FIELDS - set(document)
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
    file_name = str(document["file_name"])
    if Path(file_name).name != file_name or not file_name.lower().endswith(".pdf"):
        raise ValueError("file_name must be a plain .pdf file name")
