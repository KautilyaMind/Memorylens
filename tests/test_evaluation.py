from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.citations import validate_citation_markers
from src.evaluation.answer_review import review_record, summarize_reviews
from src.evaluation.dataset import load_benchmark, validate_benchmark
from src.evaluation.manifest import validate_manifest
from src.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from src.evaluation.runner import run_retrieval_benchmark, save_results
from src.metadata import Chunk
from src.retriever import RETRIEVAL_MODES, RetrievalResponse


def fixture_chunk(chunk_id: str, document_id: str = "DOC") -> Chunk:
    return Chunk("evidence", {"chunk_id": chunk_id, "document_id": document_id})


def fixture_question(answerable: bool = True) -> dict:
    return {
        "question_id": "Q1" if answerable else "U1",
        "question": "Which evidence is relevant?",
        "category": "semantic" if answerable else "unanswerable",
        "relevant_documents": ["DOC"] if answerable else [],
        "relevance_judgments": {"A": 2, "B": 1} if answerable else {},
        "reference_answer": "A and B" if answerable else "Insufficient evidence",
        "evidence_notes": "fixture",
        "difficulty": "easy",
        "answerable": answerable,
        "review_status": "fixture",
    }


class StaticEvaluationRetriever:
    def retrieve_with_debug(self, query, filters=None, mode="dense", top_k=None):
        order = ["C", "B", "A"] if mode == "dense" else ["A", "C", "B"]
        results = [
            {
                "chunk_id": chunk_id,
                "document_id": "DOC",
                "title": chunk_id,
                "text": "evidence",
                "final_rank": rank,
            }
            for rank, chunk_id in enumerate(order[:top_k], 1)
        ]
        timings = {
            "query_analysis_ms": 1.0,
            "dense_retrieval_ms": 2.0 if mode != "bm25" else 0.0,
            "bm25_retrieval_ms": 2.0 if mode != "dense" else 0.0,
            "rrf_ms": 1.0 if mode in {"hybrid", "hybrid_rerank"} else 0.0,
            "reranking_ms": 3.0 if mode == "hybrid_rerank" else 0.0,
            "total_retrieval_ms": 7.0,
        }
        return RetrievalResponse(results, {}, timings, mode)


class EvaluationTests(unittest.TestCase):
    def test_recall_and_mrr(self) -> None:
        judgments = {"A": 2, "B": 1}
        ranked = ["C", "B", "A"]
        self.assertEqual(0.5, recall_at_k(ranked, judgments, 2))
        self.assertEqual(1.0, recall_at_k(ranked, judgments, 3))
        self.assertEqual(0.5, reciprocal_rank(ranked, judgments))
        self.assertEqual(0.0, reciprocal_rank(["C"], judgments))

    def test_ndcg_uses_graded_gain_and_log_discount(self) -> None:
        judgments = {"A": 2, "B": 1}
        ranked = ["C", "B", "A"]
        actual = ndcg_at_k(ranked, judgments, 3)
        dcg = 1 / math.log2(3) + 3 / math.log2(4)
        ideal = 3 + 1 / math.log2(3)
        self.assertAlmostEqual(dcg / ideal, actual)
        self.assertEqual(0.0, ndcg_at_k([], {}, 5))

    def test_relevance_loading_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(json.dumps(fixture_question()) + "\n", encoding="utf-8")
            loaded = load_benchmark(path)
            summary = validate_benchmark(
                loaded, [fixture_chunk("A"), fixture_chunk("B")]
            )
            self.assertEqual(1, summary["answerable_count"])

    def test_validation_rejects_unknown_chunk_and_accepts_unanswerable(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown chunk"):
            validate_benchmark([fixture_question()], [fixture_chunk("A")])
        summary = validate_benchmark([fixture_question(False)], [fixture_chunk("A")])
        self.assertEqual(1, summary["unanswerable_count"])

    def test_manifest_detects_obsolete_chunk_judgments(self) -> None:
        frozen = {
            "benchmark_version": "1",
            "benchmark_sha256": "a",
            "corpus": {"documents": {"DOC": {"sha256": "x"}}},
            "chunks": {"count": 2, "chunk_id_digest": "old", "chunk_size": 10, "chunk_overlap": 1, "table_extraction": True},
            "retrieval": {"embedding_model": "model"},
        }
        current = json.loads(json.dumps(frozen))
        current["chunks"]["chunk_id_digest"] = "new"
        self.assertEqual(["chunk ID digest"], validate_manifest(frozen, current))

    def test_all_modes_and_result_serialization(self) -> None:
        questions = [fixture_question(), fixture_question(False)]
        chunks = [fixture_chunk("A"), fixture_chunk("B"), fixture_chunk("C")]
        results = run_retrieval_benchmark(
            StaticEvaluationRetriever(), questions, chunks, repeats=2, depth=3, warmup=False
        )
        self.assertEqual(set(RETRIEVAL_MODES), set(results["aggregates"]))
        self.assertEqual(4, len(results["per_question"]))
        self.assertEqual(1, len(results["unanswerable_questions"]))
        with tempfile.TemporaryDirectory() as directory:
            paths = save_results(results, Path(directory))
            self.assertTrue(paths["json"].is_file())
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["report"].is_file())
            payload = json.loads(paths["latest_json"].read_text(encoding="utf-8"))
            self.assertEqual(1, payload["schema_version"])

    def test_citation_marker_integrity_remains_automatic(self) -> None:
        cleaned, invalid = validate_citation_markers("Supported [1], invalid [4].", 2)
        self.assertEqual("Supported [1], invalid.", cleaned)
        self.assertEqual([4], invalid)

    def test_answer_review_retains_evidence_and_never_invents_human_scores(self) -> None:
        question = {**fixture_question(), "evaluation_mode": "dense"}
        response = {
            "answer": "Supported [1].",
            "model": "fixture-model",
            "results": [
                {
                    "chunk_id": "A",
                    "document_id": "DOC",
                    "title": "Fixture",
                    "page": 1,
                    "text": "direct evidence",
                }
            ],
            "sources": [{"marker": 1, "chunk_id": "A"}],
            "invalid_citations": [],
            "timings": {},
        }
        record = review_record(question, response, {"A": fixture_chunk("A"), "B": fixture_chunk("B")})
        self.assertEqual("direct evidence", record["retrieved_evidence_passages"][0]["text"])
        self.assertEqual("pending_human_review", record["review_status"])
        self.assertIsNone(record["scores"]["correctness"])
        summary = summarize_reviews([record])
        self.assertEqual(0, summary["human_reviewed"])
        self.assertEqual(1, summary["citation_marker_integrity_passed"])


if __name__ == "__main__":
    unittest.main()
