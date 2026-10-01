from __future__ import annotations

import time
from typing import Any

from src.citations import validate_citation_markers
from src.context_builder import build_context
from src.gemini_client import GeminiClient
from src.retriever import HybridRetriever

SYSTEM_RULES = """You are answering questions about memory and storage technology using only the supplied retrieved evidence.

Rules:
1. Use only the provided context.
2. Do not invent technical specifications or product characteristics.
3. Preserve numerical values, units, qualifiers, and product associations exactly.
4. Distinguish measured, theoretical, typical, maximum, and advertised values.
5. If the evidence is insufficient, explicitly say so.
6. Cite factual claims using only supplied source markers such as [1] or [2].
7. Distinguish different memory generations and product families carefully.
8. For comparison questions, use a table only when the supplied evidence supports it.
9. Do not cite sources that do not support the claim.
"""


def build_prompt(question: str, evidence: str) -> str:
    return (
        f"{SYSTEM_RULES}\nRetrieved evidence:\n\n{evidence}"
        f"\n\nQuestion: {question}\n\nAnswer:"
    )


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GeminiClient,
        context_max_chars: int = 16000,
    ):
        self.retriever = retriever
        self.generator = generator
        self.context_max_chars = context_max_chars

    def answer(
        self,
        question: str,
        filters: dict[str, str] | None = None,
        mode: str = "hybrid_rerank",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve_with_debug(question, filters=filters, mode=mode)
        timings = {**retrieval.timings, "gemini_generation_ms": 0.0}
        if not retrieval.results:
            timings["end_to_end_ms"] = (time.perf_counter() - started) * 1000
            return {
                "answer": "The indexed evidence is insufficient to answer this question.",
                "model": None,
                "sources": [],
                "results": [],
                "retrieval_mode": mode,
                "query_analysis": retrieval.analysis,
                "timings": timings,
                "invalid_citations": [],
            }

        context = build_context(retrieval.results, self.context_max_chars)
        stage = time.perf_counter()
        generation = self.generator.generate(build_prompt(question, context.evidence))
        timings["gemini_generation_ms"] = (time.perf_counter() - stage) * 1000
        answer, invalid_citations = validate_citation_markers(
            generation.text, len(context.results)
        )
        sources = [
            {
                "marker": result["source_marker"],
                "chunk_id": result.get("chunk_id", ""),
                "title": result.get("title", "Untitled"),
                "document_id": result.get("document_id", ""),
                "url": result.get("source_url", ""),
                "section": result.get("section", ""),
                "page": result.get("page", 0),
                "content_type": result.get("content_type", "text"),
            }
            for result in context.results
        ]
        timings["end_to_end_ms"] = (time.perf_counter() - started) * 1000
        return {
            "answer": answer,
            "model": generation.model,
            "sources": sources,
            "results": context.results,
            "retrieval_mode": mode,
            "query_analysis": retrieval.analysis,
            "timings": timings,
            "invalid_citations": invalid_citations,
        }
