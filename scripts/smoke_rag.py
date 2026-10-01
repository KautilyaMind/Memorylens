from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.bm25_retriever import BM25Retriever, BM25Store
from src.chunk_store import chunk_ids, load_chunks
from src.embeddings import LocalEmbeddings
from src.gemini_client import GeminiClient
from src.rag import RAGPipeline
from src.retriever import DenseRetriever, HybridRetriever
from src.vectorstore import VectorStore


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run one end-to-end MemoryLens RAG query")
    parser.add_argument(
        "question",
        nargs="?",
        default="Why is high-bandwidth memory useful for AI workloads?",
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "bm25", "hybrid"),
        default=settings.retrieval_mode,
    )
    args = parser.parse_args()

    store = VectorStore.load(settings.vectorstore_dir)
    chunks = load_chunks(settings.chunks_dir)
    if chunk_ids(store.chunks) != chunk_ids(chunks):
        raise RuntimeError("FAISS chunk IDs do not match canonical chunks; rerun ingestion")
    embeddings = LocalEmbeddings(settings.embedding_model)
    retriever = HybridRetriever(
        DenseRetriever(store, embeddings, settings.dense_candidates),
        BM25Retriever(BM25Store.load(chunks, settings.bm25_dir), settings.bm25_candidates),
        settings.top_k,
        settings.rrf_k,
    )
    pipeline = RAGPipeline(
        retriever,
        GeminiClient(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_fallback_models,
            settings.gemini_max_retries,
            settings.gemini_retry_base_seconds,
        ),
    )
    response = pipeline.answer(args.question, mode=args.mode)

    print(f"Primary model: {settings.gemini_model}")
    print(f"Model used: {response['model']}")
    print(f"Question: {args.question}")
    print(f"Retrieval mode: {args.mode}")
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
