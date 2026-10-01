from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.bm25_retriever import technical_tokenize
from src.citations import validate_citation_markers
from src.context_builder import build_context
from src.document_manifest import load_document_manifest
from src.loaders import extract_pdf_tables
from src.query_analysis import analyze_query
from src.reranker import CrossEncoderReranker
from src.retriever import HybridRetriever


class PredictableModel:
    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        return [2.0 if "relevant" in passage else -1.0 for _, passage in pairs]


class StaticRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, filters=None, top_k=None):
        return self.results


class V03FeatureTests(unittest.TestCase):
    def test_query_analysis_classifies_identifiers_specs_and_comparisons(self) -> None:
        analysis = analyze_query("Compare HBM3E versus LPDDR5X at 1.2 TB/s")
        self.assertEqual("comparison", analysis["query_type"])
        self.assertEqual(["HBM3E", "LPDDR5X"], analysis["technical_terms"])
        self.assertEqual(["1.2 TB/s"], analysis["numerical_terms"])
        self.assertTrue(analysis["is_comparison"])
        self.assertEqual(["HBM", "LPDDR"], analysis["detected_product_families"])

    def test_tokenizer_normalizes_spec_variations_without_losing_codes(self) -> None:
        self.assertEqual(
            ["8800", "mt/s"], technical_tokenize("8800MT/s")
        )
        self.assertEqual(
            ["8800", "mt/s"], technical_tokenize("8800 MTps")
        )
        self.assertEqual(
            ["ddr5-8800", "ddr5", "8800", "mt25qu"],
            technical_tokenize("DDR5-8800 MT25QU"),
        )

    def test_reranker_orders_scores_and_preserves_metadata(self) -> None:
        candidates = [
            {"chunk_id": "A", "text": "less useful", "title": "First", "page": 1},
            {"chunk_id": "B", "text": "relevant HBM3E evidence", "title": "Second", "page": 2},
        ]
        reranker = CrossEncoderReranker("fake", model=PredictableModel(), batch_size=2)
        results = reranker.rerank("HBM3E", candidates, top_k=2)
        self.assertEqual("B", results[0]["chunk_id"])
        self.assertEqual(2, results[0]["page"])
        self.assertEqual(1, results[0]["reranker_rank"])
        self.assertGreater(results[0]["reranker_score"], results[1]["reranker_score"])

    def test_unavailable_reranker_has_clear_error(self) -> None:
        def unavailable(name: str):
            raise OSError("model files unavailable")

        reranker = CrossEncoderReranker("missing-model", model_factory=unavailable)
        with self.assertRaisesRegex(RuntimeError, "Could not load reranker model"):
            reranker.rerank(
                "query", [{"chunk_id": "A", "text": "passage", "title": "A"}], 1
            )

    def test_citation_validation_removes_only_invalid_markers(self) -> None:
        answer, invalid = validate_citation_markers(
            "Supported [1]. Mixed [2, 9]. Fabricated [8].", source_count=2
        )
        self.assertEqual("Supported [1]. Mixed [2]. Fabricated.", answer)
        self.assertEqual([8, 9], invalid)

    def test_context_builder_reduces_same_page_overlap_and_numbers_sources(self) -> None:
        results = [
            {
                "chunk_id": "A",
                "document_id": "DOC",
                "title": "Title",
                "page": 1,
                "section": "Specs",
                "content_type": "text",
                "source_url": "https://example.com/a.pdf",
                "text": "HBM3E bandwidth is greater than 1.2 TB/s.",
            },
            {
                "chunk_id": "B",
                "document_id": "DOC",
                "title": "Title",
                "page": 1,
                "section": "Specs",
                "content_type": "text",
                "source_url": "https://example.com/a.pdf",
                "text": "HBM3E bandwidth is greater than 1.2 TB/s.",
            },
        ]
        context = build_context(results, max_chars=2000)
        self.assertEqual(1, len(context.results))
        self.assertEqual(1, context.results[0]["source_marker"])
        self.assertIn("Source ID: A", context.evidence)

    def test_local_pdf_table_extraction_preserves_rows_and_metadata(self) -> None:
        entry = load_document_manifest()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.pdf"
            pdf = pymupdf.open()
            page = pdf.new_page()
            xs = (50, 210, 360)
            ys = (70, 100, 130, 160)
            for x in xs:
                page.draw_line((x, ys[0]), (x, ys[-1]))
            for y in ys:
                page.draw_line((xs[0], y), (xs[-1], y))
            values = (
                ("Parameter", "Value"),
                ("Data Rate", "8800 MT/s"),
                ("Voltage", "1.2 V"),
            )
            for row, cells in enumerate(values):
                for column, value in enumerate(cells):
                    page.insert_text((xs[column] + 5, ys[row] + 20), value)
            pdf.save(path)
            pdf.close()

            tables = extract_pdf_tables(path, entry)
            self.assertEqual(1, len(tables))
            self.assertIn("Parameter: Data Rate", tables[0].text)
            self.assertIn("Value: 8800 MT/s", tables[0].text)
            self.assertEqual("table", tables[0].metadata["content_type"])
            self.assertTrue(tables[0].metadata["contains_table"])
            self.assertEqual(1, tables[0].metadata["page"])
            self.assertEqual(entry["source_url"], tables[0].metadata["source_url"])

    def test_complete_hybrid_to_reranker_path(self) -> None:
        dense = StaticRetriever(
            [
                {"chunk_id": "A", "text": "other", "title": "A", "dense_rank": 1, "dense_score": 0.9},
                {"chunk_id": "B", "text": "relevant evidence", "title": "B", "dense_rank": 2, "dense_score": 0.8},
            ]
        )
        bm25 = StaticRetriever(
            [
                {"chunk_id": "B", "text": "relevant evidence", "title": "B", "bm25_rank": 1, "bm25_score": 4.0}
            ]
        )
        retriever = HybridRetriever(
            dense,
            bm25,
            reranker=CrossEncoderReranker("fake", model=PredictableModel()),
            rerank_candidates=20,
            rerank_top_k=1,
        )
        response = retriever.retrieve_with_debug("relevant", mode="hybrid_rerank")
        self.assertEqual("B", response.results[0]["chunk_id"])
        self.assertEqual(1, response.results[0]["reranker_rank"])
        self.assertGreaterEqual(response.timings["total_retrieval_ms"], 0)


if __name__ == "__main__":
    unittest.main()
