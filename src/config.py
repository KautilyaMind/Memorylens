from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    values = (item.strip() for item in os.getenv(name, default).split(","))
    return tuple(dict.fromkeys(item for item in values if item))


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    documents_file: Path = PROJECT_ROOT / "config" / "documents.yaml"
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
    chunks_dir: Path = PROJECT_ROOT / "data" / "chunks"
    bm25_dir: Path = PROJECT_ROOT / "data" / "bm25"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    gemini_fallback_models: tuple[str, ...] = _csv_env(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite",
    )
    gemini_max_retries: int = _int_env("GEMINI_MAX_RETRIES", 1)
    gemini_retry_base_seconds: float = _float_env("GEMINI_RETRY_BASE_SECONDS", 1.0)
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = _int_env("TOP_K", 5)
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid_rerank").strip().lower()
    dense_candidates: int = _int_env("DENSE_CANDIDATES", 20)
    bm25_candidates: int = _int_env("BM25_CANDIDATES", 20)
    rrf_k: int = _int_env("RRF_K", 60)
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_candidates: int = _int_env("RERANK_CANDIDATES", 20)
    rerank_top_k: int = _int_env("RERANK_TOP_K", 5)
    rerank_batch_size: int = _int_env("RERANK_BATCH_SIZE", 8)
    enable_query_analysis: bool = _bool_env("ENABLE_QUERY_ANALYSIS", True)
    enable_table_extraction: bool = _bool_env("ENABLE_TABLE_EXTRACTION", True)
    context_max_chars: int = _int_env("CONTEXT_MAX_CHARS", 16000)
    chunk_size: int = _int_env("CHUNK_SIZE", 1000)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 150)

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError(
                "RETRIEVAL_MODE must be dense, bm25, hybrid, or hybrid_rerank"
            )
        for name, value in (
            ("TOP_K", self.top_k),
            ("DENSE_CANDIDATES", self.dense_candidates),
            ("BM25_CANDIDATES", self.bm25_candidates),
            ("RRF_K", self.rrf_k),
            ("RERANK_CANDIDATES", self.rerank_candidates),
            ("RERANK_TOP_K", self.rerank_top_k),
            ("RERANK_BATCH_SIZE", self.rerank_batch_size),
            ("CONTEXT_MAX_CHARS", self.context_max_chars),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


settings = Settings()
