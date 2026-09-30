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


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    values = (item.strip() for item in os.getenv(name, default).split(","))
    return tuple(dict.fromkeys(item for item in values if item))


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    sources_file: Path = PROJECT_ROOT / "config" / "sources.yaml"
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    vectorstore_dir: Path = PROJECT_ROOT / "data" / "vectorstore"
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
    chunk_size: int = _int_env("CHUNK_SIZE", 1000)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 150)


settings = Settings()
