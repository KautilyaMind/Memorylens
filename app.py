from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import settings
from src.gemini_client import GeminiClient
from src.rag import RAGPipeline
from src.query_analysis import analyze_query
from src.retriever import HybridRetriever
from src.retriever_filters import FILTER_FIELDS
from src.runtime import load_retrieval_runtime

EXAMPLES = [
    "Why is high-bandwidth memory important for AI inference?",
    "What does the documentation say about HBM3E?",
    "Find information related to MT25Q.",
    "Which document discusses 8800 MT/s?",
    "How do HBM3E characteristics support high-performance AI workloads?",
]

st.set_page_config(page_title="MemoryLens v1.0", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading dense and BM25 retrieval indexes...")
def load_retrieval() -> tuple[HybridRetriever, list[Any], dict[str, Any]]:
    return load_retrieval_runtime()


@st.cache_data(show_spinner=False)
def load_saved_evaluation(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        f"Content type: {result.get('content_type', 'text')}"
        + (f" · Table: {result.get('table_title')}" if result.get("table_title") else "")
    )
    st.caption(
        " · ".join(
            [
                score_text("Dense", result.get("dense_score"), result.get("dense_rank")),
                score_text("BM25", result.get("bm25_score"), result.get("bm25_rank")),
                score_text("RRF", result.get("rrf_score"), result.get("rrf_rank")),
                score_text(
                    "Reranker",
                    result.get("reranker_score"),
                    result.get("reranker_rank"),
                ),
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
    columns = st.columns(4)
    modes = ("dense", "bm25", "hybrid", "hybrid_rerank")
    names = {
        "dense": "Dense",
        "bm25": "BM25",
        "hybrid": "Hybrid",
        "hybrid_rerank": "Hybrid + Reranker",
    }
    for column, mode in zip(columns, modes):
        with column:
            st.markdown(f"**{names[mode]} Top {settings.top_k}**")
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
                    f"RRF:{result.get('rrf_rank') or '—'} · "
                    f"CE:{result.get('reranker_rank') or '—'}"
                )


def render_debug(response: dict[str, Any]) -> None:
    with st.expander("Query analysis and latency", expanded=False):
        analysis = response.get("query_analysis", {})
        st.json(analysis)
        timings = response.get("timings", {})
        labels = {
            "query_analysis_ms": "Query analysis",
            "dense_retrieval_ms": "Dense retrieval",
            "bm25_retrieval_ms": "BM25 retrieval",
            "rrf_ms": "RRF",
            "reranking_ms": "Cross-encoder reranking",
            "total_retrieval_ms": "Total retrieval",
            "gemini_generation_ms": "Gemini generation",
            "end_to_end_ms": "End to end",
        }
        for key, label in labels.items():
            st.caption(f"{label}: {float(timings.get(key, 0.0)):.1f} ms")


st.title("MemoryLens v1.0")
st.caption("Advanced retrieval, grounded generation, and measured evaluation over Micron technical documentation")

try:
    retriever, canonical_chunks, index_stats = load_retrieval()
except Exception as exc:
    st.error(str(exc))
    st.info("Build the canonical chunks, FAISS, and BM25 indexes with `python scripts/ingest.py`.")
    st.stop()

with st.sidebar:
    st.subheader("Retrieval")
    mode_options = {
        "Hybrid + Reranker": "hybrid_rerank",
        "Hybrid": "hybrid",
        "Dense": "dense",
        "BM25": "bm25",
    }
    configured_label = next(
        (label for label, value in mode_options.items() if value == settings.retrieval_mode),
        "Hybrid + Reranker",
    )
    mode_label = st.selectbox(
        "Retrieval Mode",
        list(mode_options),
        index=list(mode_options).index(configured_label),
    )
    mode = mode_options[mode_label]
    filters: dict[str, str] = {}
    for field in FILTER_FIELDS:
        label = field.replace("_", " ").title()
        selection = st.selectbox(label, ["Any"] + metadata_options(canonical_chunks, field))
        if selection != "Any":
            filters[field] = selection
    compare = st.checkbox("Compare Retrieval Methods", value=False)
    st.divider()
    left, middle, right = st.columns(3)
    left.metric("Documents", index_stats.get("document_count", 0))
    middle.metric("Chunks", index_stats.get("chunk_count", 0))
    right.metric(
        "Table chunks",
        sum(chunk.metadata.get("content_type") == "table" for chunk in canonical_chunks),
    )
    st.caption("Canonical text/table chunks · FAISS + BM25 · RRF · local reranker")

ask_tab, inspect_tab, evaluate_tab = st.tabs(["Ask", "Inspect", "Evaluate"])

with ask_tab:
    example = st.selectbox("Example questions", ["Choose an example..."] + EXAMPLES)
    default_question = "" if example.startswith("Choose") else example
    question = st.text_area(
        "Question",
        value=default_question,
        placeholder="Ask about Micron memory and storage technology...",
    )

    if settings.enable_query_analysis and question.strip():
        with st.expander("Query analysis preview", expanded=False):
            st.json(analyze_query(question))

    if compare and st.button("Run Retrieval Comparison"):
        if not question.strip():
            st.warning("Enter a question before comparing retrieval methods.")
        else:
            try:
                with st.spinner("Comparing all four retrieval modes..."):
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
                        settings.context_max_chars,
                    )
                    response = pipeline.answer(question, filters=filters, mode=mode)
                st.session_state["last_response"] = response
                st.session_state["last_question"] = question
                st.subheader("Answer")
                st.markdown(response["answer"])
                st.caption("Open Inspect for passages, citations, scores, and latency.")
            except Exception as exc:
                st.error(str(exc))

with inspect_tab:
    response = st.session_state.get("last_response")
    if not response:
        st.info("Ask a question first. Its evidence and diagnostics will appear here.")
    else:
        st.subheader("Last grounded answer")
        st.caption(st.session_state.get("last_question", ""))
        if not response["results"]:
            st.warning("No indexed evidence matched the selected metadata constraints.")
        st.markdown(response["answer"])
        if response["model"]:
            st.caption(
                f"Generated with `{response['model']}` · retrieval mode "
                f"`{response['retrieval_mode']}`"
            )
        if response.get("invalid_citations"):
            st.warning(
                "Removed invalid citation markers: "
                + ", ".join(map(str, response["invalid_citations"]))
            )
        render_sources(response["sources"])
        render_context(response["results"])
        render_debug(response)

with evaluate_tab:
    st.subheader("Frozen retrieval benchmark")
    results_path = settings.project_root / "evaluation" / "results" / "latest.json"
    if not results_path.exists():
        st.info(
            "No saved benchmark results are available. Run "
            "`python scripts/evaluate_retrieval.py` and refresh this page."
        )
    else:
        evaluation = load_saved_evaluation(
            str(results_path), results_path.stat().st_mtime_ns
        )
        dataset = evaluation["dataset_summary"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Answerable questions", dataset["answerable_count"])
        metric_columns[1].metric("Unanswerable controls", dataset["unanswerable_count"])
        metric_columns[2].metric("Corpus chunks", dataset["chunk_count"])
        metric_columns[3].metric(
            "Runs / question",
            evaluation["configuration"]["repeats"],
        )
        st.caption(
            f"Saved run: {evaluation['created_at']} · device: "
            f"{evaluation['hardware'].get('device', 'unknown')} · "
            "unanswerable controls excluded from ranking metrics"
        )

        summary_rows = []
        for aggregate in evaluation["aggregates"].values():
            summary_rows.append(
                {
                    "Method": aggregate["method"],
                    "Recall@3": aggregate["metrics"]["recall_at_3"]["mean"],
                    "Recall@5": aggregate["metrics"]["recall_at_5"]["mean"],
                    "Recall@10": aggregate["metrics"]["recall_at_10"]["mean"],
                    "MRR": aggregate["metrics"]["mrr"]["mean"],
                    "nDCG@5": aggregate["metrics"]["ndcg_at_5"]["mean"],
                    "Median latency (ms)": aggregate["latency_ms"]["total_retrieval_ms"]["median"],
                    "p95 latency (ms)": aggregate["latency_ms"]["total_retrieval_ms"]["p95"],
                }
            )
        st.markdown("**Retrieval quality and latency**")
        st.dataframe(summary_rows, width="stretch", hide_index=True)
        st.bar_chart(summary_rows, x="Method", y=["Recall@5", "MRR", "nDCG@5"])
        st.bar_chart(summary_rows, x="Method", y="Median latency (ms)")

        categories = sorted(
            {row["category"] for row in evaluation["category_results"]}
        )
        selected_category = st.selectbox("Category breakdown", categories)
        category_rows = [
            {
                "Method": row["method"],
                "Recall@5": row["recall_at_5"],
                "MRR": row["mrr"],
                "nDCG@5": row["ndcg_at_5"],
            }
            for row in evaluation["category_results"]
            if row["category"] == selected_category
        ]
        st.dataframe(category_rows, width="stretch", hide_index=True)

        question_rows: dict[str, list[dict[str, Any]]] = {}
        for row in evaluation["per_question"]:
            question_rows.setdefault(row["question_id"], []).append(row)
        selected_id = st.selectbox("Per-question inspector", sorted(question_rows))
        selected_rows = question_rows[selected_id]
        st.markdown(f"**{selected_rows[0]['question']}**")
        columns = st.columns(4)
        for column, row in zip(columns, selected_rows):
            with column:
                st.markdown(f"**{row['method']}**")
                st.caption(
                    f"Recall@5 {row['metrics']['recall_at_5']:.2f} · "
                    f"MRR {row['metrics']['mrr']:.2f} · "
                    f"nDCG@5 {row['metrics']['ndcg_at_5']:.2f}"
                )
                for result in row["ranked_results"]:
                    relevance = result.get("relevance", 0)
                    st.markdown(
                        f"{result['rank']}. `{result['chunk_id']}`  \n"
                        f"relevance **{relevance}** · p.{result.get('page') or '—'}"
                    )

        st.markdown("**Answer-quality review status**")
        answer_summary_path = (
            settings.project_root / "evaluation" / "results" / "answer-review-summary.json"
        )
        if answer_summary_path.exists():
            answer_summary = load_saved_evaluation(
                str(answer_summary_path), answer_summary_path.stat().st_mtime_ns
            )
            review_columns = st.columns(4)
            review_columns[0].metric("Generated answers", answer_summary["generated_answers"])
            review_columns[1].metric(
                "Marker integrity passed",
                answer_summary["citation_marker_integrity_passed"],
            )
            review_columns[2].metric(
                "Explicit abstentions",
                answer_summary["unanswerable_with_explicit_insufficiency"],
            )
            review_columns[3].metric("Human reviewed", answer_summary["human_reviewed"])
            if answer_summary["pending"]:
                st.warning(
                    f"{answer_summary['pending']} generated answers still require named "
                    "human review; correctness, faithfulness, and citation-support scores "
                    "are intentionally not reported yet."
                )
        else:
            st.caption("Run `python scripts/evaluate_answers.py --summarize` to show review status.")
