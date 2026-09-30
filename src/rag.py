from __future__ import annotations

from typing import Any

from src.gemini_client import GeminiClient
from src.retriever import DenseRetriever

SYSTEM_RULES = """You are answering questions about memory and storage technology using only the supplied retrieved evidence.

Rules:
1. Use only the provided context.
2. Do not invent technical specifications or product characteristics.
3. Preserve units exactly when quoting specifications.
4. If the evidence is insufficient, explicitly say so.
5. Cite factual claims using source markers such as [1] or [2].
6. Distinguish information from different products carefully.
7. Do not cite sources that do not support the claim.
"""


def _context_block(result: dict[str, Any], marker: int) -> str:
    location = []
    if result.get("section"):
        location.append(f"Section: {result['section']}")
    if result.get("page"):
        location.append(f"Page: {result['page']}")
    location_text = "\n".join(location)
    return f"""[Source {marker}]
Title: {result.get('title', 'Untitled')}
Document ID: {result.get('document_id', '')}
{location_text}
URL: {result.get('source_url', '')}
Content:
{result['text']}
"""


def build_prompt(question: str, results: list[dict[str, Any]]) -> str:
    evidence = "\n\n".join(_context_block(result, index) for index, result in enumerate(results, 1))
    return f"{SYSTEM_RULES}\nRetrieved evidence:\n\n{evidence}\n\nQuestion: {question}\n\nAnswer:"


class RAGPipeline:
    def __init__(self, retriever: DenseRetriever, generator: GeminiClient):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> dict[str, Any]:
        results = self.retriever.retrieve(question)
        if not results:
            return {
                "answer": "The indexed evidence is insufficient to answer this question.",
                "model": None,
                "sources": [],
                "results": [],
            }
        generation = self.generator.generate(build_prompt(question, results))
        sources = [
            {
                "marker": index,
                "title": result.get("title", "Untitled"),
                "document_id": result.get("document_id", ""),
                "url": result.get("source_url", ""),
                "section": result.get("section", ""),
                "page": result.get("page", 0),
            }
            for index, result in enumerate(results, 1)
        ]
        return {
            "answer": generation.text,
            "model": generation.model,
            "sources": sources,
            "results": results,
        }
