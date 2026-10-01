from __future__ import annotations

from typing import Any

from src.bm25_retriever import BM25Retriever
from src.embeddings import LocalEmbeddings
from src.vectorstore import VectorStore

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


class DenseRetriever:
    def __init__(self, store: VectorStore, embeddings: LocalEmbeddings, candidate_count: int = 15):
        if candidate_count <= 0:
            raise ValueError("DENSE_CANDIDATES must be positive")
        self.store = store
        self.embeddings = embeddings
        self.candidate_count = candidate_count

    def retrieve(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty")
        limit = self.candidate_count if top_k is None else top_k
        return self.store.search(
            self.embeddings.embed_query(query), limit, filters=filters
        )


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    if rrf_k <= 0:
        raise ValueError("RRF_K must be positive")
    fused: dict[str, dict[str, Any]] = {}
    for result in dense_results:
        chunk_id = str(result["chunk_id"])
        fused[chunk_id] = {
            **result,
            "dense_rank": result["dense_rank"],
            "dense_score": result["dense_score"],
            "bm25_rank": None,
            "bm25_score": None,
            "rrf_score": 1.0 / (rrf_k + result["dense_rank"]),
        }
    for result in bm25_results:
        chunk_id = str(result["chunk_id"])
        if chunk_id in fused:
            fused[chunk_id]["bm25_rank"] = result["bm25_rank"]
            fused[chunk_id]["bm25_score"] = result["bm25_score"]
            fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + result["bm25_rank"])
        else:
            fused[chunk_id] = {
                **result,
                "dense_rank": None,
                "dense_score": None,
                "rrf_score": 1.0 / (rrf_k + result["bm25_rank"]),
            }
    return sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], str(item["chunk_id"])),
    )


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ):
        if final_top_k <= 0:
            raise ValueError("TOP_K must be positive")
        if rrf_k <= 0:
            raise ValueError("RRF_K must be positive")
        self.dense = dense
        self.bm25 = bm25
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        mode = mode.strip().lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        limit = self.final_top_k if top_k is None else top_k
        if limit <= 0:
            return []

        if mode == "dense":
            results = self.dense.retrieve(query, filters)
            return [
                {
                    **result,
                    "rank": rank,
                    "final_rank": rank,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": None,
                    "score": result["dense_score"],
                }
                for rank, result in enumerate(results[:limit], start=1)
            ]
        if mode == "bm25":
            results = self.bm25.retrieve(query, filters)
            return [
                {
                    **result,
                    "rank": rank,
                    "final_rank": rank,
                    "dense_rank": None,
                    "dense_score": None,
                    "rrf_score": None,
                    "score": result["bm25_score"],
                }
                for rank, result in enumerate(results[:limit], start=1)
            ]

        dense_results = self.dense.retrieve(query, filters)
        bm25_results = self.bm25.retrieve(query, filters)
        fused = reciprocal_rank_fusion(dense_results, bm25_results, self.rrf_k)
        return [
            {
                **result,
                "rank": rank,
                "final_rank": rank,
                "score": result["rrf_score"],
            }
            for rank, result in enumerate(fused[:limit], start=1)
        ]

    def compare(
        self, query: str, filters: dict[str, str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            mode: self.retrieve(query, filters, mode=mode)
            for mode in RETRIEVAL_MODES
        }
