from __future__ import annotations

from typing import Any

from src.bm25_retriever import BM25Retriever, BM25Store
from src.chunk_store import chunk_id_digest, chunk_ids, load_chunks
from src.config import settings
from src.embeddings import LocalEmbeddings
from src.reranker import CrossEncoderReranker
from src.retriever import DenseRetriever, HybridRetriever
from src.vectorstore import VectorStore, read_index_manifest


def load_retrieval_runtime() -> tuple[HybridRetriever, list[Any], dict[str, Any]]:
    """Load and cross-check the canonical corpus and both retrieval indexes."""
    manifest = read_index_manifest(settings.vectorstore_dir)
    if manifest.get("embedding_model") != settings.embedding_model:
        raise RuntimeError(
            f"Index uses {manifest.get('embedding_model')!r}, but EMBEDDING_MODEL is "
            f"{settings.embedding_model!r}. Run python scripts/ingest.py."
        )
    canonical = load_chunks(settings.chunks_dir)
    canonical_ids = chunk_ids(canonical)
    digest = chunk_id_digest(canonical_ids)
    if manifest.get("chunk_id_digest") != digest:
        raise RuntimeError("FAISS manifest does not match canonical chunk IDs")
    vector_store = VectorStore.load(settings.vectorstore_dir)
    if chunk_ids(vector_store.chunks) != canonical_ids:
        raise RuntimeError("FAISS chunk IDs do not match canonical chunk IDs")
    bm25_store = BM25Store.load(canonical, settings.bm25_dir)
    retriever = HybridRetriever(
        DenseRetriever(
            vector_store,
            LocalEmbeddings(settings.embedding_model),
            settings.dense_candidates,
        ),
        BM25Retriever(bm25_store, settings.bm25_candidates),
        settings.top_k,
        settings.rrf_k,
        CrossEncoderReranker(settings.reranker_model, settings.rerank_batch_size),
        settings.rerank_candidates,
        settings.rerank_top_k,
        settings.enable_query_analysis,
    )
    return retriever, canonical, manifest
