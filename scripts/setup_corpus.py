from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import ingest
from src.corpus_downloader import CorpusDownloader, print_download_report
from src.validation import print_validation_report, validate_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Download, validate, and index the MemoryLens corpus")
    parser.add_argument("--force", action="store_true", help="Re-download existing documents")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n=== Download corpus ===")
    download_report = CorpusDownloader(delay=max(0.0, args.delay)).download(force=args.force)
    print_download_report(download_report)
    if download_report.available == 0:
        return 1

    print("\n=== Validate corpus ===")
    validation_report = validate_corpus(PROJECT_ROOT / "corpus")
    print_validation_report(validation_report)
    if not validation_report.can_ingest:
        return 1

    print("\n=== Build vector index ===")
    # The report is passed through so setup validates once while standalone ingest
    # still performs its own validation.
    return ingest(validation_report)


if __name__ == "__main__":
    raise SystemExit(main())
