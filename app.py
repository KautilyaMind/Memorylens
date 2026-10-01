from __future__ import annotations

from typing import Any

import streamlit as st

from src.bm25_retriever import BM25Retriever, BM25Store
from src.chunk_store import chunk_id_digest, chunk_ids, load_chunks
from src.config import settings
from src.embeddings import LocalEmbeddings
from src.gemini_client import GeminiClient
from src.rag import RAGPipeline
from src.retriever import DenseRetriever, HybridRetriever
from src.retriever_filters import FILTER_FIELDS
from src.vectorstore import VectorStore, read_index_manifest

EXAMPLES = [
    "Why is high-bandwidth memory important for AI inference?",
    "What does the documentation say about HBM3E?",
    "Find information related to MT25Q.",
    "Which document discusses 8800 MT/s?",
    "How do HBM3E characteristics support high-performance AI workloads?",
]

st.set_page_config(page_title="MemoryLens v0.2", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading dense and BM25 retrieval indexes...")
def load_retrieval() -> tuple[HybridRetriever, list[Any], dict[str, Any]]:
    manifest = read_index_manifest(settings.vectorstore_dir)
    if manifest.get("embedding_model") != settings.embedding_model:
        raise RuntimeError(
            f"Index uses {manifest.get('embedding_model')!r}, but EMBEDDING_MODEL is "
            f"{settings.embedding_model!r}. Run python scripts/ingest.py."
        )
    canonical = load_chunks(settings.chunks_dir)
    canonical_ids = chunk_ids(canonical)
    if manifest.get("chunk_id_digest") != chunk_id_digest(canonical_ids):
        raise RuntimeError("FAISS manifest does not match canonical chunk IDs")
    vector_store = VectorStore.load(settings.vectorstore_dir)
    if chunk_ids(vector_store.chunks) != canonical_ids:
        raise RuntimeError("FAISS chunk IDs do not match canonical chunk IDs")
    bm25_store = BM25Store.load(canonical, settings.bm25_dir)
    retriever = HybridRetriever(
        DenseRetriever(vector_store, LocalEmbeddings(settings.embedding_model), settings.dense_candidates),
        BM25Retriever(bm25_store, settings.bm25_candidates),
        settings.top_k,
        settings.rrf_k,
    )
    return retriever, canonical, manifest


def metadata_options(chunks: list[Any], field: str) -> list[str]:
    return sorted(
        {str(chunk.metadata.get(field, "")).strip() for chunk in chunks if chunk.metadata.get(field)},
        key=str.casefold,
    )


def score_text(label: str, value: Any, rank: Any) -> str:
    if value is None:
        return f"{label}: —"
    return f"{label}: rank {rank}, score {float(value):.4f}"


def render_sources(sources: list[dict[str, Any]]) -> None:
    st.subheader("Sources")
    for source in sources:
        location = source.get("section") or ""
        if source.get("page"):
            location = f"{location}, p.{source['page']}" if location else f"p.{source['page']}"
        label = f"[{source['marker']}] {source['title']}"
        st.markdown(f"{label}  \n{location}  \n[{source['url']}]({source['url']})")


def render_result(result: dict[str, Any], heading: bool = True) -> None:
    if heading:
        st.markdown(
            f"**{result['final_rank']}. {result['title']}**  \n"
            f"`{result['chunk_id']}`"
        )
    location = result.get("section") or "No section"
    if result.get("page"):
        location += f" · p.{result['page']}"
    st.caption(
        f"{result.get('category', '')} · {result.get('product_family', '')} · "
        f"{result.get('technology', '')} · {result.get('document_type', '')} · {location}"
    )
    st.caption(
        " · ".join(
            [
                score_text("Dense", result.get("dense_score"), result.get("dense_rank")),
                score_text("BM25", result.get("bm25_score"), result.get("bm25_rank")),
                f"RRF: {result['rrf_score']:.6f}" if result.get("rrf_score") is not None else "RRF: —",
            ]
        )
    )
    st.text(result["text"])


def render_context(results: list[dict[str, Any]]) -> None:
    with st.expander("Retrieved context", expanded=False):
        for result in results:
            render_result(result)
            st.divider()


def render_comparison(comparison: dict[str, list[dict[str, Any]]]) -> None:
    st.subheader("Compare Retrieval Methods")
    columns = st.columns(3)
    for column, mode in zip(columns, ("dense", "bm25", "hybrid")):
        with column:
            st.markdown(f"**{mode.title()} Top {settings.top_k}**")
            if not comparison[mode]:
                st.caption("No matching chunks")
            for result in comparison[mode]:
                st.markdown(
                    f"{result['final_rank']}. `{result['chunk_id']}`  \n"
                    f"{result['title']} · p.{result.get('page', '—')}"
                )
                st.caption(
                    f"D:{result.get('dense_rank') or '—'} · "
                    f"B:{result.get('bm25_rank') or '—'} · "
                    f"RRF:{result.get('rrf_score') or '—'}"
                )


st.title("MemoryLens v0.2")
st.caption("Hybrid retrieval over public Micron memory and storage documentation")

try:
    retriever, canonical_chunks, index_stats = load_retrieval()
except Exception as exc:
    st.error(str(exc))
    st.info("Build the canonical chunks, FAISS, and BM25 indexes with `python scripts/ingest.py`.")
    st.stop()

with st.sidebar:
    st.subheader("Retrieval")
    modes = ["Hybrid", "Dense", "BM25"]
    configured_mode = settings.retrieval_mode.title()
    mode = st.selectbox(
        "Retrieval Mode",
        modes,
        index=modes.index(configured_mode) if configured_mode in modes else 0,
    ).lower()
    filters: dict[str, str] = {}
    for field in FILTER_FIELDS:
        label = field.replace("_", " ").title()
        selection = st.selectbox(label, ["Any"] + metadata_options(canonical_chunks, field))
        if selection != "Any":
            filters[field] = selection
    compare = st.checkbox("Compare Retrieval Methods", value=False)
    st.divider()
    left, right = st.columns(2)
    left.metric("Documents", index_stats.get("document_count", 0))
    right.metric("Chunks", index_stats.get("chunk_count", 0))
    st.caption("Canonical chunks · FAISS + BM25 · RRF fusion")

example = st.selectbox("Example questions", ["Choose an example..."] + EXAMPLES)
default_question = "" if example.startswith("Choose") else example
question = st.text_area(
    "Question",
    value=default_question,
    placeholder="Ask about Micron memory and storage technology...",
)

if compare and st.button("Run Retrieval Comparison"):
    if not question.strip():
        st.warning("Enter a question before comparing retrieval methods.")
    else:
        try:
            with st.spinner("Comparing dense, BM25, and hybrid retrieval..."):
                render_comparison(retriever.compare(question, filters))
        except Exception as exc:
            st.error(str(exc))

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner(f"Running {mode} retrieval and generating a grounded answer..."):
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
                response = pipeline.answer(question, filters=filters, mode=mode)
            st.subheader("Answer")
            if not response["results"]:
                st.warning("No indexed evidence matched the selected metadata constraints.")
            st.markdown(response["answer"])
            if response["model"]:
                st.caption(f"Generated with `{response['model']}` · retrieval mode `{mode}`")
            render_sources(response["sources"])
            render_context(response["results"])
        except Exception as exc:
            st.error(str(exc))
