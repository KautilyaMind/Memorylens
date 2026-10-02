from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence


def _relevant_ids(judgments: Mapping[str, int]) -> set[str]:
    return {chunk_id for chunk_id, grade in judgments.items() if grade > 0}


def recall_at_k(
    ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = _relevant_ids(judgments)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def reciprocal_rank(
    ranked_ids: Sequence[str], judgments: Mapping[str, int]
) -> float:
    relevant = _relevant_ids(judgments)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(
        (2 ** judgments.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(
    ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int
) -> float:
    ideal_grades = sorted((grade for grade in judgments.values() if grade > 0), reverse=True)
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades[:k], start=1)
    )
    return dcg_at_k(ranked_ids, judgments, k) / ideal if ideal else 0.0


def ranking_metrics(
    ranked_ids: Sequence[str], judgments: Mapping[str, int]
) -> dict[str, float]:
    return {
        "recall_at_3": recall_at_k(ranked_ids, judgments, 3),
        "recall_at_5": recall_at_k(ranked_ids, judgments, 5),
        "recall_at_10": recall_at_k(ranked_ids, judgments, 10),
        "mrr": reciprocal_rank(ranked_ids, judgments),
        "ndcg_at_5": ndcg_at_k(ranked_ids, judgments, 5),
        "ndcg_at_10": ndcg_at_k(ranked_ids, judgments, 10),
    }


def summarize(values: Iterable[float]) -> dict[str, float]:
    items = list(values)
    if not items:
        return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(items),
        "stddev": pstdev(items),
        "min": min(items),
        "max": max(items),
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(items) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    fraction = position - lower
    return items[lower] + (items[upper] - items[lower]) * fraction
