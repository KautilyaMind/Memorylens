from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.bm25_retriever import BM25Retriever
from src.embeddings import LocalEmbeddings
from src.query_analysis import analyze_query
from src.reranker import CrossEncoderReranker
from src.vectorstore import VectorStore

RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class RetrievalResponse:
    results: list[dict[str, Any]]
    analysis: dict[str, Any]
    timings: dict[str, float]
    mode: str


class DenseRetriever:
    def __init__(self, store: VectorStore, embeddings: LocalEmbeddings, candidate_count: int = 20):
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
        reranker: CrossEncoderReranker | None = None,
        rerank_candidates: int = 20,
        rerank_top_k: int = 5,
        enable_query_analysis: bool = True,
    ):
        for name, value in (
            ("TOP_K", final_top_k),
            ("RRF_K", rrf_k),
            ("RERANK_CANDIDATES", rerank_candidates),
            ("RERANK_TOP_K", rerank_top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.dense = dense
        self.bm25 = bm25
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.rerank_top_k = rerank_top_k
        self.enable_query_analysis = enable_query_analysis
        self.last_response: RetrievalResponse | None = None

    @staticmethod
    def _timings() -> dict[str, float]:
        return {
            "query_analysis_ms": 0.0,
            "dense_retrieval_ms": 0.0,
            "bm25_retrieval_ms": 0.0,
            "rrf_ms": 0.0,
            "reranking_ms": 0.0,
            "total_retrieval_ms": 0.0,
        }

    def retrieve_with_debug(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid_rerank",
        top_k: int | None = None,
    ) -> RetrievalResponse:
        started = time.perf_counter()
        mode = mode.strip().lower()
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        timings = self._timings()

        stage = time.perf_counter()
        analysis = analyze_query(query) if self.enable_query_analysis else {
            "query_type": "disabled",
            "technical_terms": [],
            "numerical_terms": [],
            "is_comparison": False,
            "detected_product_families": [],
        }
        timings["query_analysis_ms"] = (time.perf_counter() - stage) * 1000

        if mode == "dense":
            limit = self.final_top_k if top_k is None else top_k
            stage = time.perf_counter()
            raw = self.dense.retrieve(query, filters)
            timings["dense_retrieval_ms"] = (time.perf_counter() - stage) * 1000
            results = [
                {
                    **result,
                    "rank": rank,
                    "final_rank": rank,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_rank": None,
                    "rrf_score": None,
                    "reranker_rank": None,
                    "reranker_score": None,
                    "score": result["dense_score"],
                }
                for rank, result in enumerate(raw[:limit], start=1)
            ]
        elif mode == "bm25":
            limit = self.final_top_k if top_k is None else top_k
            stage = time.perf_counter()
            raw = self.bm25.retrieve(query, filters)
            timings["bm25_retrieval_ms"] = (time.perf_counter() - stage) * 1000
            results = [
                {
                    **result,
                    "rank": rank,
                    "final_rank": rank,
                    "dense_rank": None,
                    "dense_score": None,
                    "rrf_rank": None,
                    "rrf_score": None,
                    "reranker_rank": None,
                    "reranker_score": None,
                    "score": result["bm25_score"],
                }
                for rank, result in enumerate(raw[:limit], start=1)
            ]
        else:
            stage = time.perf_counter()
            dense_results = self.dense.retrieve(query, filters)
            timings["dense_retrieval_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            bm25_results = self.bm25.retrieve(query, filters)
            timings["bm25_retrieval_ms"] = (time.perf_counter() - stage) * 1000
            stage = time.perf_counter()
            fused = reciprocal_rank_fusion(dense_results, bm25_results, self.rrf_k)
            fused = [
                {
                    **result,
                    "rrf_rank": rank,
                    "rank": rank,
                    "final_rank": rank,
                    "reranker_rank": None,
                    "reranker_score": None,
                    "score": result["rrf_score"],
                }
                for rank, result in enumerate(fused, start=1)
            ]
            timings["rrf_ms"] = (time.perf_counter() - stage) * 1000
            if mode == "hybrid":
                limit = self.final_top_k if top_k is None else top_k
                results = fused[:limit]
            else:
                if self.reranker is None:
                    raise RuntimeError(
                        "Hybrid + Reranker mode is unavailable because no reranker is configured"
                    )
                limit = self.rerank_top_k if top_k is None else top_k
                stage = time.perf_counter()
                results = self.reranker.rerank(
                    query, fused[: self.rerank_candidates], limit
                )
                timings["reranking_ms"] = (time.perf_counter() - stage) * 1000

        timings["total_retrieval_ms"] = (time.perf_counter() - started) * 1000
        response = RetrievalResponse(results, analysis, timings, mode)
        self.last_response = response
        return response

    def retrieve(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid_rerank",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.retrieve_with_debug(query, filters, mode, top_k).results

    def compare(
        self, query: str, filters: dict[str, str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            mode: self.retrieve(query, filters, mode=mode)
            for mode in RETRIEVAL_MODES
        }
