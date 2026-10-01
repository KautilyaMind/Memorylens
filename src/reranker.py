from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        model: Any | None = None,
        model_factory: Callable[[str], Any] = CrossEncoder,
    ):
        if batch_size <= 0:
            raise ValueError("RERANK_BATCH_SIZE must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model
        self._model_factory = model_factory

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                self._model = self._model_factory(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load reranker model {self.model_name!r}: {exc}"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates or top_k <= 0:
            return []
        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            chunk_id = str(candidate.get("chunk_id", ""))
            if not chunk_id:
                raise ValueError("Reranker candidate is missing chunk_id")
            if chunk_id not in seen:
                seen.add(chunk_id)
                deduplicated.append(candidate)
        pairs = [
            (
                query,
                "\n".join(
                    part
                    for part in (
                        str(candidate.get("title", "")),
                        str(candidate.get("section", "")),
                        candidate["text"],
                    )
                    if part
                ),
            )
            for candidate in deduplicated
        ]
        try:
            scores = np.asarray(
                self._load_model().predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False
                )
            ).reshape(-1)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Cross-encoder reranking failed: {exc}") from exc
        if len(scores) != len(deduplicated):
            raise RuntimeError("Reranker score count does not match candidate count")
        ranked = sorted(
            zip(deduplicated, scores),
            key=lambda item: (-float(item[1]), str(item[0]["chunk_id"])),
        )
        return [
            {
                **candidate,
                "reranker_score": float(score),
                "reranker_rank": rank,
                "rank": rank,
                "final_rank": rank,
                "score": float(score),
            }
            for rank, (candidate, score) in enumerate(ranked[:top_k], start=1)
        ]
