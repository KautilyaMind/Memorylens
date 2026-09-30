from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.document_manifest import document_path, load_document_manifest

SHORT_TEXT_WARNING_CHARS = 500
MIN_EXTRACTABLE_CHARS = 50


@dataclass
class ValidationReport:
    configured: int
    available: int
    valid_pdfs: int
    pages_parsed: int
    category_counts: dict[str, int]
    missing: list[str] = field(default_factory=list)
    invalid: list[tuple[str, str]] = field(default_factory=list)
    short_text_warnings: list[tuple[str, int]] = field(default_factory=list)

    @property
    def can_ingest(self) -> bool:
        return self.valid_pdfs > 0 and not self.invalid


def _inspect_pdf(path: Path) -> tuple[int, int]:
    if path.stat().st_size == 0:
        raise ValueError("file is empty")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("missing PDF signature")
    try:
        with pymupdf.open(path) as pdf:
            pages = len(pdf)
            text_chars = sum(len(page.get_text("text").strip()) for page in pdf)
    except Exception as exc:
        raise ValueError(f"unreadable PDF: {exc}") from exc
    if text_chars < MIN_EXTRACTABLE_CHARS:
        raise ValueError(
            f"insufficient extractable text ({text_chars} characters; "
            f"minimum {MIN_EXTRACTABLE_CHARS})"
        )
    return text_chars, pages


def validate_corpus(corpus_dir: Path) -> ValidationReport:
    documents = load_document_manifest()
    category_counts = {"DRAM": 0, "NAND": 0, "NOR": 0, "TECHNICAL": 0}
    missing: list[str] = []
    invalid: list[tuple[str, str]] = []
    warnings: list[tuple[str, int]] = []
    available = 0
    valid = 0
    pages_parsed = 0
    for entry in documents:
        document_id = str(entry["document_id"])
        path = document_path(entry, corpus_dir)
        if not path.exists():
            missing.append(document_id)
            continue
        available += 1
        try:
            text_chars, page_count = _inspect_pdf(path)
        except (OSError, ValueError) as exc:
            invalid.append((document_id, str(exc)))
            continue
        valid += 1
        pages_parsed += page_count
        category_counts[str(entry["category"])] += 1
        if text_chars < SHORT_TEXT_WARNING_CHARS:
            warnings.append((document_id, text_chars))
    return ValidationReport(
        configured=len(documents),
        available=available,
        valid_pdfs=valid,
        pages_parsed=pages_parsed,
        category_counts=category_counts,
        missing=missing,
        invalid=invalid,
        short_text_warnings=warnings,
    )


def print_validation_report(report: ValidationReport) -> None:
    print(f"Documents configured: {report.configured}")
    print(f"Files available: {report.available}")
    print(f"Valid PDFs: {report.valid_pdfs}")
    print(f"Pages parsed: {report.pages_parsed}")
    print(f"Documents skipped: {report.configured - report.valid_pdfs}")
    print(f"Ready for indexing: {report.valid_pdfs}")
    for category, count in report.category_counts.items():
        print(f"{category} documents: {count}")
    print(f"Missing: {', '.join(report.missing) if report.missing else 'None'}")
    if report.invalid:
        print("Invalid PDFs:")
        for document_id, error in report.invalid:
            print(f"  - {document_id}: {error}")
    else:
        print("Invalid PDFs: None")
    if report.short_text_warnings:
        print("Short-text warnings:")
        for document_id, count in report.short_text_warnings:
            print(f"  - {document_id}: {count} extracted characters")
    else:
        print("Short-text warnings: None")
