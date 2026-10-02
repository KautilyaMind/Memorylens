from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RETRIEVAL_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

RUBRIC = {
    "correctness": "0 incorrect, 1 partly correct, 2 matches the reference facts",
    "faithfulness": "0 unsupported, 1 partly supported, 2 all substantive claims supported",
    "citation_correctness": "0 citations do not support claims, 1 mixed, 2 citations support claims",
    "abstention": "For unanswerable items: 0 fabricates, 1 hedges, 2 clearly declines unsupported facts",
}


def select_review_sample(
    questions: list[dict[str, Any]], per_category: int = 4
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in questions:
        by_category[item["category"]].append(item)
    sample: list[dict[str, Any]] = []
    mode_index = 0
    for category in sorted(key for key in by_category if key != "unanswerable"):
        for item in by_category[category][:per_category]:
            sample.append({**item, "evaluation_mode": RETRIEVAL_MODES[mode_index % 4]})
            mode_index += 1
    for item in by_category.get("unanswerable", []):
        sample.append({**item, "evaluation_mode": "hybrid_rerank"})
    return sample


def review_record(
    question: dict[str, Any],
    response: dict[str, Any] | None = None,
    chunk_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = response or {}
    chunk_lookup = chunk_lookup or {}
    retrieved_results = response.get("results", [])

    def passage(chunk_id: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
        if result is not None:
            return {
                "chunk_id": chunk_id,
                "document_id": result.get("document_id", ""),
                "title": result.get("title", ""),
                "page": result.get("page", 0),
                "text": result.get("text", ""),
            }
        chunk = chunk_lookup.get(chunk_id)
        return {
            "chunk_id": chunk_id,
            "document_id": chunk.metadata.get("document_id", "") if chunk else "",
            "title": chunk.metadata.get("title", "") if chunk else "",
            "page": chunk.metadata.get("page", 0) if chunk else 0,
            "text": chunk.text if chunk else "",
        }

    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "category": question["category"],
        "answerable": question["answerable"],
        "retrieval_mode": question["evaluation_mode"],
        "reference_answer": question["reference_answer"],
        "reference_evidence": question["relevance_judgments"],
        "reference_evidence_passages": [
            passage(chunk_id)
            for chunk_id, grade in question["relevance_judgments"].items()
            if grade > 0
        ],
        "generated_answer": response.get("answer", ""),
        "generation_model": response.get("model"),
        "retrieved_chunk_ids": [
            result.get("chunk_id", "") for result in retrieved_results
        ],
        "retrieved_evidence_passages": [
            passage(str(result.get("chunk_id", "")), result)
            for result in retrieved_results
        ],
        "citations": response.get("sources", []),
        "invalid_citation_markers": response.get("invalid_citations", []),
        "citation_marker_integrity": not response.get("invalid_citations", []),
        "generation_latency_ms": response.get("timings", {}).get("gemini_generation_ms"),
        "end_to_end_latency_ms": response.get("timings", {}).get("end_to_end_ms"),
        "rubric": RUBRIC,
        "scores": {
            "correctness": None,
            "faithfulness": None,
            "citation_correctness": None,
            "abstention": None,
        },
        "reviewer": "",
        "review_notes": "",
        "review_status": "pending_human_review",
    }


def write_review_queue(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def load_review_queue(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_reviews(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        record
        for record in records
        if record.get("review_status") == "human_reviewed"
        and record.get("reviewer")
    ]
    summary: dict[str, Any] = {
        "total_records": len(records),
        "generated_answers": sum(bool(record.get("generated_answer")) for record in records),
        "citation_marker_integrity_passed": sum(
            bool(record.get("generated_answer"))
            and bool(record.get("citation_marker_integrity"))
            for record in records
        ),
        "unanswerable_with_explicit_insufficiency": sum(
            not record.get("answerable", True)
            and any(
                phrase in record.get("generated_answer", "").lower()
                for phrase in ("insufficient", "no mention", "no information")
            )
            for record in records
        ),
        "human_reviewed": len(completed),
        "pending": len(records) - len(completed),
        "by_mode": {},
    }
    for mode in RETRIEVAL_MODES:
        all_mode_records = [record for record in records if record["retrieval_mode"] == mode]
        mode_records = [record for record in completed if record["retrieval_mode"] == mode]
        metrics: dict[str, float | None] = {}
        for name in RUBRIC:
            values = [
                record["scores"].get(name)
                for record in mode_records
                if record["scores"].get(name) is not None
            ]
            metrics[name] = statistics.mean(values) if values else None
        summary["by_mode"][mode] = {
            "generated": sum(bool(record.get("generated_answer")) for record in all_mode_records),
            "reviewed": len(mode_records),
            "mean_scores": metrics,
            "automatic_citation_marker_integrity_rate": (
                statistics.mean(
                    bool(record.get("citation_marker_integrity"))
                    for record in all_mode_records
                    if record.get("generated_answer")
                )
                if any(record.get("generated_answer") for record in all_mode_records)
                else None
            ),
        }
    return summary
