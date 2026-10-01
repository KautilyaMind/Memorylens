from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunk_store import chunk_id_digest, chunk_ids
from src.metadata import Chunk
from src.retriever_filters import matches_filters

INDEX_FILE = "index.json"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./+-][a-z0-9]+)*", re.IGNORECASE)
GENERIC_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "does",
    "document",
    "documentation",
    "documents",
    "discuss",
    "discusses",
    "find",
    "for",
    "from",
    "in",
    "information",
    "how",
    "important",
    "is",
    "of",
    "on",
    "or",
    "related",
    "say",
    "the",
    "to",
    "what",
    "which",
    "why",
    "with",
}


def technical_tokenize(text: str) -> list[str]:
    """Tokenize without stemming away product codes, units, or numeric specs."""
    output: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        if token in GENERIC_STOP_WORDS:
            continue
        joined_spec = re.fullmatch(
            r"(\d+(?:\.\d+)?)(mt/s|mtps|gt/s|gb/s|tb/s|gbps|tbps)", token
        )
        if joined_spec:
            number, unit = joined_spec.groups()
            output.extend((number, _normalize_unit(unit)))
            continue
        if token in {"mtps", "mt/s", "gt/s", "gb/s", "tb/s", "gbps", "tbps"}:
            output.append(_normalize_unit(token))
            continue
        ddr_speed = re.fullmatch(r"((?:lp)?ddr\d+x?|gddr\d+|hbm\d+e?)-(\d+)", token)
        if ddr_speed:
            output.extend((token, ddr_speed.group(1), ddr_speed.group(2)))
            continue
        output.append(token)
    return output


def _normalize_unit(unit: str) -> str:
    return {
        "mtps": "mt/s",
        "gbps": "gb/s",
        "tbps": "tb/s",
    }.get(unit, unit)


class BM25Store:
    def __init__(self, chunks: list[Chunk], tokenized_corpus: list[list[str]], directory: Path):
        if not chunks or len(chunks) != len(tokenized_corpus):
            raise ValueError("BM25 chunks and tokenized corpus must be non-empty and aligned")
        self.chunks = chunks
        self.tokenized_corpus = tokenized_corpus
        self.directory = directory
        self.index = BM25Okapi(tokenized_corpus)

    @classmethod
    def build(cls, chunks: list[Chunk], directory: Path) -> "BM25Store":
        tokens = [
            technical_tokenize(
                " ".join(
                    (
                        str(chunk.metadata.get("title", "")),
                        str(chunk.metadata.get("section", "")),
                        chunk.text,
                    )
                )
            )
            for chunk in chunks
        ]
        if any(not item for item in tokens):
            raise ValueError("Cannot build BM25 with an empty tokenized chunk")
        store = cls(chunks, tokens, directory)
        store.save()
        return store

    @classmethod
    def load(cls, chunks: list[Chunk], directory: Path) -> "BM25Store":
        path = directory / INDEX_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found in {directory}. Run: python scripts/ingest.py"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tokens = payload["tokenized_corpus"]
            stored_ids = payload["chunk_ids"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load BM25 index: {exc}") from exc
        expected_ids = chunk_ids(chunks)
        if stored_ids != expected_ids:
            raise RuntimeError("BM25 chunk IDs do not match the canonical chunk dataset")
        if payload.get("chunk_id_digest") != chunk_id_digest(expected_ids):
            raise RuntimeError("BM25 chunk ID digest is corrupt")
        if payload.get("chunk_count") != len(chunks):
            raise RuntimeError("BM25 chunk count is corrupt")
        if (
            not isinstance(tokens, list)
            or len(tokens) != len(chunks)
            or any(
                not isinstance(row, list)
                or any(not isinstance(token, str) for token in row)
                for row in tokens
            )
        ):
            raise RuntimeError("BM25 tokenized corpus count is corrupt")
        try:
            return cls(chunks, tokens, directory)
        except Exception as exc:
            raise RuntimeError(f"Could not reconstruct BM25 index: {exc}") from exc

    def save(self) -> None:
        ids = chunk_ids(self.chunks)
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunk_count": len(ids),
            "chunk_ids": ids,
            "chunk_id_digest": chunk_id_digest(ids),
            "tokenized_corpus": self.tokenized_corpus,
        }
        (self.directory / INDEX_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def scores(self, query: str) -> np.ndarray:
        tokens = technical_tokenize(query)
        if not tokens:
            raise ValueError("Query contains no searchable BM25 tokens")
        return np.asarray(self.index.get_scores(tokens), dtype="float64")


class BM25Retriever:
    def __init__(self, store: BM25Store, candidate_count: int = 15):
        if candidate_count <= 0:
            raise ValueError("BM25_CANDIDATES must be positive")
        self.store = store
        self.candidate_count = candidate_count

    def retrieve(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.store.scores(query.strip())
        limit = self.candidate_count if top_k is None else top_k
        eligible = [
            index
            for index, chunk in enumerate(self.store.chunks)
            if matches_filters(chunk.metadata, filters) and scores[index] != 0
        ]
        ranked = sorted(eligible, key=lambda index: (-scores[index], index))[:limit]
        return [
            {
                "text": self.store.chunks[index].text,
                **self.store.chunks[index].metadata,
                "bm25_score": float(scores[index]),
                "bm25_rank": rank,
            }
            for rank, index in enumerate(ranked, start=1)
        ]
