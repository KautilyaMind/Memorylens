from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = {
    "question_id",
    "question",
    "category",
    "relevant_documents",
    "relevance_judgments",
    "reference_answer",
    "evidence_notes",
    "difficulty",
    "answerable",
    "review_status",
}
VALID_GRADES = {0, 1, 2}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")
    questions: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid benchmark JSON on line {line_number}: {exc}") from exc
        missing = REQUIRED_FIELDS.difference(item)
        if missing:
            raise ValueError(
                f"Benchmark line {line_number} is missing: {', '.join(sorted(missing))}"
            )
        questions.append(item)
    if not questions:
        raise ValueError("Benchmark dataset is empty")
    return questions


def validate_benchmark(
    questions: Iterable[dict[str, Any]], chunks: Iterable[Any]
) -> dict[str, Any]:
    chunk_list = list(chunks)
    chunk_to_document = {
        str(chunk.metadata["chunk_id"]): str(chunk.metadata["document_id"])
        for chunk in chunk_list
    }
    document_ids = set(chunk_to_document.values())
    errors: list[str] = []
    seen: set[str] = set()
    category_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    question_count = 0
    answerable_count = 0

    for item in questions:
        question_count += 1
        qid = str(item["question_id"])
        if qid in seen:
            errors.append(f"{qid}: duplicate question_id")
        seen.add(qid)
        category_counts[str(item["category"])] += 1
        status_counts[str(item["review_status"])] += 1
        judgments = item["relevance_judgments"]
        if not isinstance(judgments, dict):
            errors.append(f"{qid}: relevance_judgments must be an object")
            continue
        if item["difficulty"] not in VALID_DIFFICULTIES:
            errors.append(f"{qid}: invalid difficulty {item['difficulty']!r}")
        answerable = bool(item["answerable"])
        answerable_count += int(answerable)
        positive = {key for key, grade in judgments.items() if grade in {1, 2}}
        invalid_grades = {key: grade for key, grade in judgments.items() if grade not in VALID_GRADES}
        if invalid_grades:
            errors.append(f"{qid}: invalid relevance grades {invalid_grades}")
        if answerable and not positive:
            errors.append(f"{qid}: answerable question has no positive judgments")
        if not answerable and positive:
            errors.append(f"{qid}: unanswerable question has positive judgments")
        for chunk_id in judgments:
            if chunk_id not in chunk_to_document:
                errors.append(f"{qid}: unknown chunk {chunk_id}")
        relevant_documents = set(item["relevant_documents"])
        unknown_documents = relevant_documents.difference(document_ids)
        if unknown_documents:
            errors.append(f"{qid}: unknown documents {sorted(unknown_documents)}")
        judged_documents = {
            chunk_to_document[chunk_id]
            for chunk_id in positive
            if chunk_id in chunk_to_document
        }
        if answerable and not judged_documents.issubset(relevant_documents):
            errors.append(
                f"{qid}: relevant_documents omits judged documents "
                f"{sorted(judged_documents.difference(relevant_documents))}"
            )

    if errors:
        raise ValueError("Benchmark validation failed:\n- " + "\n- ".join(errors))
    return {
        "question_count": question_count,
        "answerable_count": answerable_count,
        "unanswerable_count": question_count - answerable_count,
        "category_counts": dict(sorted(category_counts.items())),
        "review_status_counts": dict(sorted(status_counts.items())),
        "chunk_count": len(chunk_to_document),
        "document_count": len(document_ids),
    }
