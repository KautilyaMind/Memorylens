from __future__ import annotations

import csv
import json
import platform
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.dataset import validate_benchmark
from src.evaluation.metrics import percentile, ranking_metrics, summarize
from src.retriever import RETRIEVAL_MODES, HybridRetriever

METHOD_LABELS = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + Reranker",
}
METRIC_NAMES = (
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
)
TIMING_NAMES = (
    "query_analysis_ms",
    "dense_retrieval_ms",
    "bm25_retrieval_ms",
    "rrf_ms",
    "reranking_ms",
    "total_retrieval_ms",
)


def _hardware() -> dict[str, Any]:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_version = torch.__version__
        accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        device, torch_version, accelerator = "unknown", None, None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "device": device,
        "accelerator": accelerator,
        "torch": torch_version,
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.mean(values),
    }


def run_retrieval_benchmark(
    retriever: HybridRetriever,
    questions: list[dict[str, Any]],
    chunks: list[Any],
    *,
    repeats: int = 2,
    depth: int = 10,
    warmup: bool = True,
) -> dict[str, Any]:
    if repeats <= 0 or depth <= 0:
        raise ValueError("repeats and depth must be positive")
    dataset_summary = validate_benchmark(questions, chunks)
    chunk_lookup = {
        str(chunk.metadata["chunk_id"]): chunk for chunk in chunks
    }
    answerable = [item for item in questions if item["answerable"]]
    unanswerable = [item for item in questions if not item["answerable"]]
    model_warmup_ms = 0.0
    if warmup and answerable:
        started = time.perf_counter()
        for mode in RETRIEVAL_MODES:
            retriever.retrieve_with_debug(answerable[0]["question"], mode=mode, top_k=depth)
        model_warmup_ms = (time.perf_counter() - started) * 1000

    per_question: list[dict[str, Any]] = []
    latency_samples: dict[str, dict[str, list[float]]] = {
        mode: {name: [] for name in TIMING_NAMES} for mode in RETRIEVAL_MODES
    }
    for question in answerable:
        for mode in RETRIEVAL_MODES:
            runs = [
                retriever.retrieve_with_debug(
                    question["question"], mode=mode, top_k=depth
                )
                for _ in range(repeats)
            ]
            first = runs[0]
            ranked_ids = [str(item["chunk_id"]) for item in first.results]
            timings = {
                name: _latency_summary([run.timings[name] for run in runs])
                for name in TIMING_NAMES
            }
            for run in runs:
                for name in TIMING_NAMES:
                    latency_samples[mode][name].append(run.timings[name])
            per_question.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "review_status": question["review_status"],
                    "mode": mode,
                    "method": METHOD_LABELS[mode],
                    "relevance_judgments": question["relevance_judgments"],
                    "ranked_results": [
                        {
                            "rank": rank,
                            "chunk_id": result["chunk_id"],
                            "document_id": result.get("document_id", ""),
                            "title": result.get("title", ""),
                            "page": result.get("page", 0),
                            "content_type": result.get("content_type", "text"),
                            "relevance": question["relevance_judgments"].get(
                                result["chunk_id"], 0
                            ),
                            "dense_rank": result.get("dense_rank"),
                            "bm25_rank": result.get("bm25_rank"),
                            "rrf_rank": result.get("rrf_rank"),
                            "reranker_rank": result.get("reranker_rank"),
                            "text": result.get("text", ""),
                        }
                        for rank, result in enumerate(first.results, 1)
                    ],
                    "metrics": ranking_metrics(
                        ranked_ids, question["relevance_judgments"]
                    ),
                    "latency_ms": timings,
                    "repeat_count": repeats,
                }
            )

    aggregates: dict[str, Any] = {}
    category_results: list[dict[str, Any]] = []
    for mode in RETRIEVAL_MODES:
        method_rows = [row for row in per_question if row["mode"] == mode]
        aggregates[mode] = {
            "method": METHOD_LABELS[mode],
            "question_count": len(method_rows),
            "metrics": {
                name: summarize(row["metrics"][name] for row in method_rows)
                for name in METRIC_NAMES
            },
            "latency_ms": {
                name: _latency_summary(latency_samples[mode][name])
                for name in TIMING_NAMES
            },
        }
        categories = sorted({row["category"] for row in method_rows})
        for category in categories:
            rows = [row for row in method_rows if row["category"] == category]
            category_results.append(
                {
                    "mode": mode,
                    "method": METHOD_LABELS[mode],
                    "category": category,
                    "question_count": len(rows),
                    **{
                        name: statistics.mean(row["metrics"][name] for row in rows)
                        for name in METRIC_NAMES
                    },
                }
            )

    failures = []
    for row in per_question:
        if row["metrics"]["recall_at_5"] >= 1.0:
            continue
        expected = [
            chunk_id
            for chunk_id, grade in row["relevance_judgments"].items()
            if grade > 0
        ]
        failures.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "category": row["category"],
                "mode": row["mode"],
                "expected_chunks": expected,
                "retrieved_chunks": [
                    result["chunk_id"] for result in row["ranked_results"][:5]
                ],
                "expected_evidence": [
                    {
                        "chunk_id": chunk_id,
                        "passage": chunk_lookup[chunk_id].text[:500],
                    }
                    for chunk_id in expected
                    if chunk_id in chunk_lookup
                ],
                "retrieved_evidence": [
                    {
                        "chunk_id": result["chunk_id"],
                        "passage": result["text"][:500],
                    }
                    for result in row["ranked_results"][:5]
                ],
                "observation": "One or more judged-relevant chunks were absent from the top 5.",
            }
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": dataset_summary,
        "configuration": {
            "modes": list(RETRIEVAL_MODES),
            "final_depth": depth,
            "repeats": repeats,
            "identical_metadata_filters": {},
            "warmup_enabled": warmup,
            "model_warmup_ms": model_warmup_ms,
        },
        "hardware": _hardware(),
        "aggregates": aggregates,
        "category_results": category_results,
        "per_question": per_question,
        "unanswerable_questions": [
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "category": item["category"],
                "reference_answer": item["reference_answer"],
            }
            for item in unanswerable
        ],
        "failure_cases": failures,
    }


def _report_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# MemoryLens v1.0 Retrieval Evaluation",
        "",
        f"Measured at `{results['created_at']}` using "
        f"{results['configuration']['repeats']} steady-state run(s) per question/mode. "
        "Unanswerable questions are excluded from ranking aggregates.",
        "",
        "## Aggregate results",
        "",
        "| Method | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Median retrieval ms | p95 retrieval ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in RETRIEVAL_MODES:
        row = results["aggregates"][mode]
        metric = row["metrics"]
        latency = row["latency_ms"]["total_retrieval_ms"]
        lines.append(
            f"| {row['method']} | {metric['recall_at_3']['mean']:.3f} | "
            f"{metric['recall_at_5']['mean']:.3f} | {metric['recall_at_10']['mean']:.3f} | "
            f"{metric['mrr']['mean']:.3f} | {metric['ndcg_at_5']['mean']:.3f} | "
            f"{metric['ndcg_at_10']['mean']:.3f} | {latency['median']:.1f} | {latency['p95']:.1f} |"
        )
    advanced = results["aggregates"]["hybrid_rerank"]
    hybrid = results["aggregates"]["hybrid"]
    advanced_latency = advanced["latency_ms"]["total_retrieval_ms"]["median"]
    hybrid_latency = hybrid["latency_ms"]["total_retrieval_ms"]["median"]
    latency_multiple = advanced_latency / hybrid_latency if hybrid_latency else 0.0
    lines.extend(
        [
            "",
            "## Quality versus latency",
            "",
            f"Compared with Hybrid, Hybrid + Reranker changed mean Recall@5 from "
            f"{hybrid['metrics']['recall_at_5']['mean']:.3f} to "
            f"{advanced['metrics']['recall_at_5']['mean']:.3f}, MRR from "
            f"{hybrid['metrics']['mrr']['mean']:.3f} to "
            f"{advanced['metrics']['mrr']['mean']:.3f}, and nDCG@5 from "
            f"{hybrid['metrics']['ndcg_at_5']['mean']:.3f} to "
            f"{advanced['metrics']['ndcg_at_5']['mean']:.3f}. Median retrieval "
            f"latency changed from {hybrid_latency:.1f} ms to {advanced_latency:.1f} ms "
            f"({latency_multiple:.1f}× on this hardware).",
            "",
            "These are measurements for this frozen benchmark, not a general performance claim.",
            "",
            "Per-question Recall@5 population standard deviations were "
            + ", ".join(
                f"{results['aggregates'][mode]['method']} "
                f"{results['aggregates'][mode]['metrics']['recall_at_5']['stddev']:.3f}"
                for mode in RETRIEVAL_MODES
            )
            + ". The spread reinforces the category and failure-case analysis rather than relying only on means.",
        ]
    )
    lines.extend(
        [
            "",
            "Metrics use positive relevance grades as relevant for Recall/MRR. nDCG uses "
            "gain `2^grade - 1` and logarithmic discount `log2(rank + 1)`.",
            "",
            "## Category results",
            "",
            "| Method | Category | Questions | Recall@5 | MRR | nDCG@5 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in results["category_results"]:
        lines.append(
            f"| {row['method']} | {row['category']} | {row['question_count']} | "
            f"{row['recall_at_5']:.3f} | {row['mrr']:.3f} | {row['ndcg_at_5']:.3f} |"
        )
    advanced_categories = {
        row["category"]: row
        for row in results["category_results"]
        if row["mode"] == "hybrid_rerank"
    }
    simpler_wins = [
        row
        for row in results["category_results"]
        if row["mode"] != "hybrid_rerank"
        and row["recall_at_5"] >= advanced_categories[row["category"]]["recall_at_5"]
    ]
    lines.extend(["", "### Where simpler retrieval matched or beat advanced Recall@5", ""])
    if simpler_wins:
        for row in simpler_wins:
            lines.append(
                f"- {row['category']}: {row['method']} {row['recall_at_5']:.3f} vs. "
                f"Hybrid + Reranker {advanced_categories[row['category']]['recall_at_5']:.3f}"
            )
    else:
        lines.append("No category-level matches or wins were measured in this run.")
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            f"{len(results['failure_cases'])} question/method pairs did not retrieve all "
            "judged-relevant chunks in the top five. Representative cases:",
            "",
        ]
    )
    for failure in results["failure_cases"][:12]:
        lines.extend(
            [
                f"### {failure['question_id']} — {failure['mode']}",
                "",
                failure["question"],
                "",
                f"Expected: `{', '.join(failure['expected_chunks'])}`",
                f"Retrieved top 5: `{', '.join(failure['retrieved_chunks'])}`",
                "",
                "Expected passage: "
                + (
                    failure["expected_evidence"][0]["passage"].replace("\n", " ")
                    if failure["expected_evidence"]
                    else "Unavailable"
                ),
                "",
                "Top retrieved passage: "
                + (
                    failure["retrieved_evidence"][0]["passage"].replace("\n", " ")
                    if failure["retrieved_evidence"]
                    else "No result"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- Relevance judgments are corpus-grounded but require independent human sign-off before being described as a final human-reviewed benchmark.",
            "- A valid citation marker does not establish that the cited passage entails a generated claim.",
            "- Results apply only to the frozen corpus, models, chunking configuration, and hardware recorded with this run.",
            "- The benchmark measures retrieval ranking; answer correctness and faithfulness use a separate manual-review workflow.",
            "",
        ]
    )
    return "\n".join(lines)


def save_results(results: dict[str, Any], output_root: Path) -> dict[str, Path]:
    results_dir = output_root / "results"
    reports_dir = output_root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"retrieval-{stamp}.json"
    csv_path = results_dir / f"retrieval-{stamp}.csv"
    summary_path = results_dir / f"summary-{stamp}.csv"
    report_path = reports_dir / f"retrieval-{stamp}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "question_id", "category", "mode", *METRIC_NAMES,
            "median_total_retrieval_ms", "ranked_chunk_ids",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results["per_question"]:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "mode": row["mode"],
                    **row["metrics"],
                    "median_total_retrieval_ms": row["latency_ms"]["total_retrieval_ms"]["median"],
                    "ranked_chunk_ids": "|".join(item["chunk_id"] for item in row["ranked_results"]),
                }
            )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["mode", *METRIC_NAMES, "median_retrieval_ms", "p95_retrieval_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in RETRIEVAL_MODES:
            row = results["aggregates"][mode]
            writer.writerow(
                {
                    "mode": mode,
                    **{name: row["metrics"][name]["mean"] for name in METRIC_NAMES},
                    "median_retrieval_ms": row["latency_ms"]["total_retrieval_ms"]["median"],
                    "p95_retrieval_ms": row["latency_ms"]["total_retrieval_ms"]["p95"],
                }
            )
    report_path.write_text(_report_markdown(results), encoding="utf-8")
    latest_json = results_dir / "latest.json"
    latest_csv = results_dir / "latest.csv"
    latest_summary = results_dir / "latest-summary.csv"
    latest_report = reports_dir / "latest.md"
    for source, target in (
        (json_path, latest_json),
        (csv_path, latest_csv),
        (summary_path, latest_summary),
        (report_path, latest_report),
    ):
        shutil.copyfile(source, target)
    return {
        "json": json_path,
        "csv": csv_path,
        "summary_csv": summary_path,
        "report": report_path,
        "latest_json": latest_json,
        "latest_report": latest_report,
    }
