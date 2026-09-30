from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_builder import CorpusBuilder, print_build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the controlled MemoryLens Micron corpus")
    parser.add_argument("--force", action="store_true", help="Re-download existing documents")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = CorpusBuilder(delay=max(0.0, args.delay)).build(force=args.force)
    print_build_report(report)
    return 0 if report.processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
