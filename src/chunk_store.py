from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.metadata import Chunk

CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


def chunk_ids(chunks: list[Chunk]) -> list[str]:
    ids = [str(chunk.metadata.get("chunk_id", "")) for chunk in chunks]
    if any(not chunk_id for chunk_id in ids):
        raise ValueError("Every canonical chunk must have a chunk_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical chunk dataset contains duplicate chunk IDs")
    return ids


def chunk_id_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def write_chunks(chunks: list[Chunk], directory: Path) -> dict[str, Any]:
    ids = chunk_ids(chunks)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps({"text": chunk.text, **chunk.metadata}, ensure_ascii=False) + "\n"
            )
    manifest = {"chunk_count": len(chunks), "chunk_id_digest": chunk_id_digest(ids)}
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_chunks(directory: Path) -> list[Chunk]:
    chunks_path = directory / CHUNKS_FILE
    manifest_path = directory / MANIFEST_FILE
    if not chunks_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Canonical chunks not found in {directory}. Run: python scripts/ingest.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                text = str(record.pop("text"))
                chunks.append(Chunk(text=text, metadata=record))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load canonical chunks: {exc}") from exc
    ids = chunk_ids(chunks)
    if manifest.get("chunk_count") != len(chunks):
        raise RuntimeError("Canonical chunk count does not match its manifest")
    if manifest.get("chunk_id_digest") != chunk_id_digest(ids):
        raise RuntimeError("Canonical chunk IDs do not match their manifest")
    return chunks
