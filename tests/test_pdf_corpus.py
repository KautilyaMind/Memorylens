from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.chunking import chunk_documents
from src.document_manifest import document_path, load_document_manifest
from src.loaders import extract_pdf_pages


class PdfCorpusTests(unittest.TestCase):
    def test_manifest_is_unique_and_pdf_only(self) -> None:
        documents = load_document_manifest()
        self.assertEqual(16, len(documents))
        self.assertEqual(len(documents), len({item["document_id"] for item in documents}))
        self.assertEqual(len(documents), len({item["file_name"] for item in documents}))
        self.assertTrue(all(item["file_name"].endswith(".pdf") for item in documents))
        self.assertEqual(
            Path("corpus/dram/micron-hbm3e-product-brief.pdf"),
            document_path(documents[0], Path("corpus")),
        )

    def test_pdf_cleaning_and_deterministic_chunk_ids(self) -> None:
        entry = load_document_manifest()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            pdf = pymupdf.open()
            for page_number, word in ((1, "Alpha"), (2, "Beta")):
                page = pdf.new_page()
                page.insert_text(
                    (72, 72),
                    f"MICRON TECHNICAL BRIEF\nOVERVIEW\nA wrapped technical\n{word} statement at 12 GB/s."
                    f"\nPage {page_number}\nMicron Technology, Inc.",
                )
            pdf.save(path)
            pdf.close()

            pages = extract_pdf_pages(path, entry)
            self.assertEqual(2, len(pages))
            self.assertNotIn("MICRON TECHNICAL BRIEF", pages[0].text)
            self.assertNotIn("Micron Technology, Inc.", pages[0].text)
            self.assertNotIn("Page 1", pages[0].text)
            self.assertIn("wrapped technical Alpha statement", pages[0].text)

            first = chunk_documents(pages, size=1000, overlap=150)
            second = chunk_documents(pages, size=1000, overlap=150)
            self.assertEqual(
                [chunk.metadata["chunk_id"] for chunk in first],
                [chunk.metadata["chunk_id"] for chunk in second],
            )
            self.assertTrue(first[0].metadata["contains_numeric_specs"])
            self.assertEqual(1, first[0].metadata["page"])


if __name__ == "__main__":
    unittest.main()
