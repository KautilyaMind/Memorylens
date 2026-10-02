from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.chunk_store import chunk_id_digest, chunk_ids
from src.config import settings
from src.evaluation import BENCHMARK_VERSION, FAISS_INDEX_VERSION, TOKENIZER_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evaluation_manifest(chunks: list[Any], dataset_path: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        metadata = chunk.metadata
        document_id = str(metadata["document_id"])
        if document_id in documents:
            continue
        file_path = Path(str(metadata.get("file_path", "")))
        if not file_path.is_absolute():
            file_path = settings.project_root / file_path
        documents[document_id] = {
            "file_name": str(metadata.get("file_name", "")),
            "sha256": sha256_file(file_path) if file_path.is_file() else "unavailable",
        }
    return {
        "manifest_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_sha256": sha256_file(dataset_path),
        "corpus": {
            "document_manifest": str(settings.documents_file.relative_to(settings.project_root)),
            "document_count": len(documents),
            "documents": dict(sorted(documents.items())),
        },
        "chunks": {
            "count": len(chunks),
            "chunk_id_digest": chunk_id_digest(chunk_ids(chunks)),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "table_extraction": settings.enable_table_extraction,
        },
        "retrieval": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "bm25_tokenizer_version": TOKENIZER_VERSION,
            "faiss_index_version": FAISS_INDEX_VERSION,
            "dense_candidates": settings.dense_candidates,
            "bm25_candidates": settings.bm25_candidates,
            "rrf_k": settings.rrf_k,
            "rerank_candidates": settings.rerank_candidates,
        },
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(
    frozen: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    checks = {
        "benchmark version": (frozen.get("benchmark_version"), current.get("benchmark_version")),
        "benchmark hash": (frozen.get("benchmark_sha256"), current.get("benchmark_sha256")),
        "document hashes": (
            frozen.get("corpus", {}).get("documents"),
            current.get("corpus", {}).get("documents"),
        ),
        "chunk count": (
            frozen.get("chunks", {}).get("count"),
            current.get("chunks", {}).get("count"),
        ),
        "chunk ID digest": (
            frozen.get("chunks", {}).get("chunk_id_digest"),
            current.get("chunks", {}).get("chunk_id_digest"),
        ),
        "chunking configuration": (
            {key: frozen.get("chunks", {}).get(key) for key in ("chunk_size", "chunk_overlap", "table_extraction")},
            {key: current.get("chunks", {}).get(key) for key in ("chunk_size", "chunk_overlap", "table_extraction")},
        ),
        "retrieval configuration": (frozen.get("retrieval"), current.get("retrieval")),
    }
    return [name for name, (expected, actual) in checks.items() if expected != actual]
