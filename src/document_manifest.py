from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.config import settings
from src.metadata import validate_document_entry

ALLOWED_CATEGORIES = {"DRAM", "NAND", "NOR", "TECHNICAL"}


def load_document_manifest(path: Path = settings.documents_file) -> list[dict[str, Any]]:
    try:
        documents = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read document manifest {path}: {exc}") from exc
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents.yaml must contain a non-empty list")

    for entry in documents:
        if not isinstance(entry, dict):
            raise ValueError("Each documents.yaml entry must be a mapping")
        validate_document_entry(entry)
        empty_fields = [field for field, value in entry.items() if not str(value).strip()]
        if empty_fields:
            raise ValueError(f"Empty fields in document entry: {', '.join(empty_fields)}")
        category = str(entry["category"]).upper()
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Unsupported category {category!r}")
        entry["category"] = category

    for field in ("document_id", "file_name"):
        values = [str(entry[field]) for entry in documents]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate {field}: {', '.join(duplicates)}")
    return documents


def document_path(entry: dict[str, Any], corpus_dir: Path = settings.corpus_dir) -> Path:
    return corpus_dir / str(entry["category"]).lower() / str(entry["file_name"])
