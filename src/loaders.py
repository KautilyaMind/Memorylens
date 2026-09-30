from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pymupdf
import yaml

from src.metadata import Document, normalize_metadata

LOGGER = logging.getLogger(__name__)


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"Markdown has no YAML front matter: {path}")
    try:
        _, front_matter, body = raw.split("---", 2)
        metadata = yaml.safe_load(front_matter) or {}
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Malformed front matter in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Front matter must be a mapping: {path}")
    body = body.strip()
    if not body:
        raise ValueError(f"Markdown document is empty: {path}")
    return Document(body, normalize_metadata(metadata, str(path.resolve())))


def load_pdf(path: Path) -> list[Document]:
    sidecar = path.with_suffix(".metadata.yaml")
    if not sidecar.exists():
        raise ValueError(f"PDF metadata sidecar is missing: {sidecar}")
    try:
        raw_metadata = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed PDF metadata sidecar {sidecar}: {exc}") from exc
    metadata = normalize_metadata(raw_metadata, str(path.resolve()))
    pages: list[Document] = []
    try:
        with pymupdf.open(path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text", sort=True).replace("\ufffd", "—").strip()
                if text:
                    page_metadata = {**metadata, "page": page_index + 1}
                    pages.append(Document(text, page_metadata))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF {path}: {exc}") from exc
    if not pages:
        raise ValueError(f"PDF contains no extractable text: {path}")
    return pages


def iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    yield from sorted(
        path for path in corpus_dir.rglob("*") if path.suffix.lower() in {".md", ".pdf"}
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in iter_corpus_files(corpus_dir):
        LOGGER.info("Loading %s", path.name)
        if path.suffix.lower() == ".md":
            documents.append(load_markdown(path))
        else:
            documents.extend(load_pdf(path))
    return documents
