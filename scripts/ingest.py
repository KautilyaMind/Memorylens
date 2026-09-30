from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_documents
from src.config import settings
from src.embeddings import LocalEmbeddings
from src.loaders import load_corpus
from src.validation import ValidationReport, print_validation_report, validate_corpus
from src.vectorstore import VectorStore, write_index_manifest

LOGGER = logging.getLogger(__name__)


def ingest(report: ValidationReport | None = None) -> int:
    if report is None:
        report = validate_corpus(settings.corpus_dir)
        print_validation_report(report)
    if not report.can_ingest:
        LOGGER.error("Corpus validation failed; index was not rebuilt")
        return 1
    if report.missing:
        LOGGER.warning("Building an incomplete index after explicitly reporting %d missing sources", len(report.missing))

    documents = load_corpus(settings.corpus_dir)
    LOGGER.info("Creating chunks...")
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        LOGGER.error("No chunks were generated")
        return 1
    LOGGER.info("Generated %d chunks", len(chunks))
    LOGGER.info("Generating local embeddings with %s...", settings.embedding_model)
    embeddings = LocalEmbeddings(settings.embedding_model).embed_documents([chunk.text for chunk in chunks])
    VectorStore.build(embeddings, chunks, settings.vectorstore_dir)
    write_index_manifest(settings.vectorstore_dir, settings.embedding_model, len(chunks), report.available)
    LOGGER.info("Saved FAISS index to %s", settings.vectorstore_dir)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return ingest()


if __name__ == "__main__":
    raise SystemExit(main())
