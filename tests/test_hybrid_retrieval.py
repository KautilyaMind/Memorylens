from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.bm25_retriever import BM25Retriever, BM25Store, technical_tokenize
from src.chunk_store import load_chunks, write_chunks
from src.metadata import Chunk
from src.rag import RAGPipeline
from src.retriever import DenseRetriever, HybridRetriever, reciprocal_rank_fusion
from src.vectorstore import VectorStore


def make_chunk(chunk_id: str, text: str, category: str = "DRAM") -> Chunk:
    return Chunk(
        text,
        {
            "chunk_id": chunk_id,
            "document_id": chunk_id.split("_CH_")[0],
            "title": chunk_id,
            "category": category,
            "product_family": "HBM",
            "technology": "HBM3E",
            "document_type": "White paper",
            "page": 1,
            "section": "Overview",
            "source_url": "https://example.com/document.pdf",
        },
    )


class StaticRetriever:
    def __init__(self, results: list[dict]):
        self.results = results

    def retrieve(self, query: str, filters=None, top_k=None):
        return self.results


class FailingGenerator:
    def generate(self, prompt: str):
        raise AssertionError("Gemini must not be called without retrieved evidence")


class StaticEmbeddings:
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype="float32")


class HybridRetrievalTests(unittest.TestCase):
    def test_technical_tokenizer_preserves_identifiers_and_units(self) -> None:
        tokens = technical_tokenize(
            "HBM3E LPDDR5X GDDR7 MT25Q MRDIMM QLC 3D NAND 8800 MT/s PCIe-Gen5"
        )
        self.assertEqual(
            [
                "hbm3e",
                "lpddr5x",
                "gddr7",
                "mt25q",
                "mrdimm",
                "qlc",
                "3d",
                "nand",
                "8800",
                "mt/s",
                "pcie-gen5",
            ],
            tokens,
        )
        self.assertEqual(
            ["mt25q"], technical_tokenize("Find information related to MT25Q")
        )

    def test_canonical_chunks_and_bm25_share_identity(self) -> None:
        chunks = [
            make_chunk("DRAM_001_CH_0001", "HBM3E bandwidth for AI"),
            make_chunk("NOR_001_CH_0001", "MT25Q serial NOR flash", "NOR"),
            make_chunk("NAND_001_CH_0001", "QLC NAND storage", "NAND"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_chunks(chunks, root / "chunks")
            canonical = load_chunks(root / "chunks")
            store = BM25Store.build(canonical, root / "bm25")
            loaded = BM25Store.load(canonical, root / "bm25")
            self.assertEqual(
                [chunk.metadata["chunk_id"] for chunk in canonical],
                [chunk.metadata["chunk_id"] for chunk in loaded.chunks],
            )
            results = BM25Retriever(store).retrieve("MT25Q")
            self.assertEqual("NOR_001_CH_0001", results[0]["chunk_id"])
            self.assertEqual([], BM25Retriever(store).retrieve("MT25Q", {"category": "DRAM"}))

    def test_rrf_combines_ranks_without_adding_raw_scores(self) -> None:
        dense = [
            {"chunk_id": "A", "dense_rank": 1, "dense_score": 0.9},
            {"chunk_id": "B", "dense_rank": 2, "dense_score": 0.8},
        ]
        bm25 = [
            {"chunk_id": "B", "bm25_rank": 1, "bm25_score": 12.0},
            {"chunk_id": "C", "bm25_rank": 2, "bm25_score": 10.0},
        ]
        fused = reciprocal_rank_fusion(dense, bm25, rrf_k=60)
        self.assertEqual("B", fused[0]["chunk_id"])
        self.assertAlmostEqual(1 / 62 + 1 / 61, fused[0]["rrf_score"])
        self.assertIsNone(fused[1]["bm25_rank"])
        self.assertIsNone(fused[2]["dense_rank"])

    def test_dense_filter_searches_full_index_before_candidate_limit(self) -> None:
        chunks = [
            make_chunk("DRAM_001_CH_0001", "dense first", "DRAM"),
            make_chunk("NOR_001_CH_0001", "eligible second", "NOR"),
            make_chunk("NOR_001_CH_0002", "eligible third", "NOR"),
        ]
        vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            store = VectorStore.build(vectors, chunks, Path(directory))
            dense = DenseRetriever(store, StaticEmbeddings(), candidate_count=1)
            results = dense.retrieve("query", {"category": "NOR"})
            self.assertEqual(["NOR_001_CH_0001"], [item["chunk_id"] for item in results])
            self.assertEqual(1, results[0]["dense_rank"])

    def test_hybrid_exposes_all_modes_and_debug_ranks(self) -> None:
        dense = StaticRetriever(
            [{"chunk_id": "A", "title": "A", "dense_rank": 1, "dense_score": 0.9}]
        )
        bm25 = StaticRetriever(
            [{"chunk_id": "A", "title": "A", "bm25_rank": 1, "bm25_score": 5.0}]
        )
        retriever = HybridRetriever(dense, bm25, final_top_k=5, rrf_k=60)
        hybrid = retriever.retrieve("HBM3E", mode="hybrid")
        self.assertEqual(1, hybrid[0]["dense_rank"])
        self.assertEqual(1, hybrid[0]["bm25_rank"])
        self.assertEqual(1, hybrid[0]["final_rank"])
        self.assertEqual({"dense", "bm25", "hybrid"}, set(retriever.compare("HBM3E")))

    def test_no_matching_evidence_does_not_call_generator(self) -> None:
        retriever = HybridRetriever(
            StaticRetriever([]), StaticRetriever([]), final_top_k=5, rrf_k=60
        )
        response = RAGPipeline(retriever, FailingGenerator()).answer(
            "HBM3E", filters={"category": "DOES-NOT-EXIST"}, mode="hybrid"
        )
        self.assertEqual([], response["results"])
        self.assertIsNone(response["model"])


if __name__ == "__main__":
    unittest.main()
