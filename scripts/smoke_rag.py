from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.embeddings import LocalEmbeddings
from src.gemini_client import GeminiClient
from src.rag import RAGPipeline
from src.retriever import DenseRetriever
from src.vectorstore import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one end-to-end MemoryLens RAG query")
    parser.add_argument(
        "question",
        nargs="?",
        default="Why is high-bandwidth memory useful for AI workloads?",
    )
    args = parser.parse_args()

    store = VectorStore.load(settings.vectorstore_dir)
    embeddings = LocalEmbeddings(settings.embedding_model)
    pipeline = RAGPipeline(
        DenseRetriever(store, embeddings, settings.top_k),
        GeminiClient(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_fallback_models,
            settings.gemini_max_retries,
            settings.gemini_retry_base_seconds,
        ),
    )
    response = pipeline.answer(args.question)

    print(f"Primary model: {settings.gemini_model}")
    print(f"Model used: {response['model']}")
    print(f"Question: {args.question}")
    print(f"\nAnswer:\n{response['answer']}")
    print("\nSources:")
    for source in response["sources"]:
        location = source["section"] or ""
        if source["page"]:
            location = f"{location}, p.{source['page']}" if location else f"p.{source['page']}"
        print(
            f"[{source['marker']}] {source['document_id']} - {source['title']}"
            f" - {location} - {source['url']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
