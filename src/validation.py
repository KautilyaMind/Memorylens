from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.corpus_builder import load_sources


@dataclass
class ValidationReport:
    expected: int
    available: int
    category_counts: dict[str, int]
    missing: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def can_ingest(self) -> bool:
        return self.available > 0 and not self.empty and not self.duplicates


def validate_corpus(corpus_dir: Path) -> ValidationReport:
    sources = load_sources(require_unique=False)
    ids = [str(source["document_id"]) for source in sources]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    category_counts = {"DRAM": 0, "NAND": 0, "NOR": 0, "TECHNICAL": 0}
    missing: list[str] = []
    empty: list[str] = []
    available = 0
    for source in sources:
        category = str(source["category"]).lower()
        expected_suffix = ".pdf" if source["source_type"] == "pdf" else ".md"
        matches = list((corpus_dir / category).glob(f"{source['document_id']}_*{expected_suffix}"))
        if not matches:
            missing.append(str(source["document_id"]))
            continue
        path = matches[0]
        available += 1
        category_counts[str(source["category"]).upper()] += 1
        if path.stat().st_size == 0:
            empty.append(str(source["document_id"]))
    return ValidationReport(len(sources), available, category_counts, missing, empty, duplicates)


def print_validation_report(report: ValidationReport) -> None:
    print(f"Total expected sources: {report.expected}")
    print(f"Total files available: {report.available}")
    print(f"DRAM count: {report.category_counts['DRAM']}")
    print(f"NAND count: {report.category_counts['NAND']}")
    print(f"NOR count: {report.category_counts['NOR']}")
    print(f"Technical count: {report.category_counts['TECHNICAL']}")
    print(f"Missing sources: {', '.join(report.missing) if report.missing else 'None'}")
    print(f"Empty documents: {', '.join(report.empty) if report.empty else 'None'}")
    print(f"Duplicate document IDs: {', '.join(report.duplicates) if report.duplicates else 'None'}")
