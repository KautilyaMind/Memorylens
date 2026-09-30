from __future__ import annotations

import streamlit as st

from src.config import settings
from src.embeddings import LocalEmbeddings
from src.gemini_client import GeminiClient
from src.rag import RAGPipeline
from src.retriever import DenseRetriever
from src.vectorstore import VectorStore, read_index_manifest

EXAMPLES = [
    "Why is high-bandwidth memory useful for AI workloads?",
    "What are the characteristics of SLC NAND?",
    "How does TLC NAND differ from QLC NAND?",
    "What advantages does LPDDR5X provide?",
    "What applications use serial NOR flash?",
    "How does DDR5 improve memory bandwidth?",
]

st.set_page_config(page_title="MemoryLens", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading local retrieval index...")
def load_retriever() -> DenseRetriever:
    manifest = read_index_manifest(settings.vectorstore_dir)
    indexed_model = manifest.get("embedding_model")
    if indexed_model != settings.embedding_model:
        raise RuntimeError(
            f"Index uses {indexed_model!r}, but EMBEDDING_MODEL is {settings.embedding_model!r}. "
            "Run python scripts/ingest.py to rebuild it."
        )
    store = VectorStore.load(settings.vectorstore_dir)
    embeddings = LocalEmbeddings(settings.embedding_model)
    return DenseRetriever(store, embeddings, settings.top_k)


def render_sources(sources: list[dict]) -> None:
    st.subheader("Sources")
    for source in sources:
        location = source.get("section") or ""
        if source.get("page"):
            location = f"{location}, p.{source['page']}" if location else f"p.{source['page']}"
        label = f"[{source['marker']}] {source['title']}"
        if source.get("url"):
            st.markdown(f"{label}  \n{location}  \n[{source['url']}]({source['url']})")
        else:
            st.markdown(f"{label}  \n{location}")


def render_context(results: list[dict]) -> None:
    with st.expander("Retrieved context", expanded=False):
        for result in results:
            location = result.get("section") or "No section"
            if result.get("page"):
                location += f" · p.{result['page']}"
            st.markdown(f"**{result['rank']}. {result['title']}**")
            st.caption(
                f"{result.get('category', '')} · {result.get('product_family', '')} · "
                f"{location} · similarity {result['score']:.3f}"
            )
            st.text(result["text"])
            st.divider()


st.title("MemoryLens")
st.caption("Technical RAG over public memory and storage documentation")

try:
    retriever = load_retriever()
except Exception as exc:
    st.error(str(exc))
    st.info("Build the local corpus and FAISS database with `python scripts/setup_corpus.py`.")
    st.stop()

example = st.selectbox("Example questions", ["Choose an example..."] + EXAMPLES)
default_question = "" if example.startswith("Choose") else example
question = st.text_area("Question", value=default_question, placeholder="Ask about Micron memory and storage technology...")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
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
                response = pipeline.answer(question)
            st.subheader("Answer")
            st.markdown(response["answer"])
            if response["model"]:
                st.caption(f"Generated with `{response['model']}`")
            render_sources(response["sources"])
            render_context(response["results"])
        except Exception as exc:
            st.error(str(exc))
