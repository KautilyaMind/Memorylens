from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REQUIRED_SOURCE_FIELDS = {
    "document_id",
    "title",
    "url",
    "category",
    "product_family",
    "technology",
    "document_type",
    "source_type",
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
        "source_url": str(raw.get("source_url") or raw.get("url") or ""),
        "source": str(raw.get("source", "Micron Technology")),
        "category": str(raw.get("category", "UNKNOWN")).upper(),
        "product_family": str(raw.get("product_family", "")),
        "technology": str(raw.get("technology", "")),
        "document_type": str(raw.get("document_type", "")),
        "source_type": str(raw.get("source_type", "")),
        "retrieved_at": str(raw.get("retrieved_at", "")),
        "local_file": str(raw.get("local_file", file_path)),
        "file_path": file_path,
    }


def validate_source(source: dict[str, Any]) -> None:
    missing = REQUIRED_SOURCE_FIELDS - set(source)
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
    if source["source_type"] not in {"webpage", "pdf"}:
        raise ValueError("source_type must be 'webpage' or 'pdf'")
