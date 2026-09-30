from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.metadata import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "index_manifest.json"


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], directory: Path):
        self.index = index
        self.chunks = chunks
        self.directory = directory

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: list[Chunk], directory: Path) -> "VectorStore":
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding matrix and chunk count do not match")
        if not chunks:
            raise ValueError("Cannot build an index with zero chunks")
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        store = cls(index, chunks, directory)
        store.save()
        return store

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        index_path = directory / INDEX_FILE
        chunks_path = directory / CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {directory}. Run: python scripts/ingest.py"
            )
        try:
            index = faiss.read_index(str(index_path))
            chunks = []
            with chunks_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    chunks.append(Chunk(record["text"], record["metadata"]))
        except Exception as exc:
            raise RuntimeError(f"Could not load vector index: {exc}") from exc
        if index.ntotal != len(chunks):
            raise RuntimeError("Corrupt vector store: FAISS and metadata counts differ")
        return cls(index, chunks, directory)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.directory / INDEX_FILE))
        with (self.directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps({"text": chunk.text, "metadata": chunk.metadata}, ensure_ascii=False) + "\n")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        scores, indices = self.index.search(
            np.ascontiguousarray(query_embedding, dtype="float32"),
            min(top_k, len(self.chunks)),
        )
        results: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append({"rank": rank, "score": float(score), "text": chunk.text, **chunk.metadata})
        return results


def write_index_manifest(directory: Path, model_name: str, chunk_count: int, document_count: int) -> None:
    payload = {
        "embedding_model": model_name,
        "chunk_count": chunk_count,
        "document_count": document_count,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_index_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Index manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
