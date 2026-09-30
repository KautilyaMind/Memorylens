from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_downloader import CorpusDownloader, print_download_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the curated Micron PDF corpus")
    parser.add_argument("--force", action="store_true", help="Replace existing PDFs")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = CorpusDownloader(delay=max(0.0, args.delay)).download(force=args.force)
    print_download_report(report)
    return 0 if report.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
