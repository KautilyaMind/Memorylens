from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import requests
import yaml

from src.config import settings
from src.metadata import validate_source
from src.webpage_parser import extract_markdown

LOGGER = logging.getLogger(__name__)
# Micron's asset CDN rejects non-browser user agents even for public PDFs.
# A stable browser UA keeps both micron.com HTML and assets.micron.com usable.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


@dataclass
class BuildReport:
    configured: int
    processed: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    markdown_documents: int = 0
    pdf_documents: int = 0


def load_sources(
    path: Path = settings.sources_file, *, require_unique: bool = True
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        sources = yaml.safe_load(handle)
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must contain a list")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source entry must be a mapping")
        validate_source(source)
        document_id = str(source["document_id"])
        if require_unique and document_id in seen:
            raise ValueError(f"Duplicate document_id in manifest: {document_id}")
        seen.add(document_id)
    return sources


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value[:80] or "document"


def _destination(source: dict[str, Any], corpus_dir: Path) -> Path:
    category = str(source["category"]).lower()
    suffix = ".pdf" if source["source_type"] == "pdf" else ".md"
    return corpus_dir / category / f"{source['document_id']}_{_slug(source['title'])}{suffix}"


def _front_matter(source: dict[str, Any], local_file: Path) -> str:
    metadata = {
        "document_id": source["document_id"],
        "title": source["title"],
        "source_url": source["url"],
        "source": "Micron Technology",
        "category": source["category"],
        "product_family": source["product_family"],
        "technology": source["technology"],
        "document_type": source["document_type"],
        "source_type": source["source_type"],
        "retrieved_at": date.today().isoformat(),
        "local_file": local_file.relative_to(settings.project_root).as_posix(),
    }
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n\n"


class CorpusBuilder:
    def __init__(self, corpus_dir: Path = settings.corpus_dir, delay: float = 0.5):
        self.corpus_dir = corpus_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Referer": "https://www.micron.com/"})

    def build(self, force: bool = False) -> BuildReport:
        sources = load_sources()
        report = BuildReport(configured=len(sources))
        for index, source in enumerate(sources):
            destination = _destination(source, self.corpus_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if destination.exists() and destination.stat().st_size > 0 and not force:
                    LOGGER.info("Skipping unchanged local file %s", destination.name)
                elif source["source_type"] == "pdf":
                    self._download_pdf(source, destination)
                else:
                    self._build_webpage(source, destination)
                report.processed += 1
                if source["source_type"] == "pdf":
                    report.pdf_documents += 1
                else:
                    report.markdown_documents += 1
            except Exception as exc:  # one bad URL must not abort the build
                LOGGER.error("Failed %s: %s", source["document_id"], exc)
                report.failed.append((str(source["document_id"]), str(exc)))
            if index < len(sources) - 1 and self.delay:
                time.sleep(self.delay)
        return report

    def _get(self, url: str, accept: str = "*/*") -> requests.Response:
        response = self.session.get(
            url,
            headers={"Accept": accept},
            timeout=(10, 45),
            allow_redirects=True,
        )
        response.raise_for_status()
        return response

    def _build_webpage(self, source: dict[str, Any], destination: Path) -> None:
        LOGGER.info("Fetching %s...", source["document_id"])
        response = self._get(str(source["url"]), "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8")
        # requests defaults text/html without an explicit charset to ISO-8859-1,
        # while Micron serves these pages as UTF-8.
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = "utf-8"
        markdown = extract_markdown(response.text, str(source["title"]))
        if len(markdown) < 100:
            raise ValueError("Extracted content is unexpectedly short")
        destination.write_text(_front_matter(source, destination) + markdown + "\n", encoding="utf-8")
        LOGGER.info("Converted webpage to Markdown: %s", destination.name)

    def _download_pdf(self, source: dict[str, Any], destination: Path) -> None:
        LOGGER.info("Downloading %s...", source["document_id"])
        response = self._get(
            str(source["url"]),
            "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        )
        content = response.content
        if not content.startswith(b"%PDF-"):
            raise ValueError("Response is not a PDF")
        destination.write_bytes(content)
        sidecar = destination.with_suffix(".metadata.yaml")
        metadata = yaml.safe_load(_front_matter(source, destination).split("---", 2)[1])
        sidecar.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8")
        LOGGER.info("Saved original PDF: %s", destination.name)


def print_build_report(report: BuildReport) -> None:
    print(f"Sources configured: {report.configured}")
    print(f"Successfully processed: {report.processed}")
    print(f"Failed: {len(report.failed)}")
    print(f"Markdown documents: {report.markdown_documents}")
    print(f"PDF documents: {report.pdf_documents}")
    if report.failed:
        print("Failures:")
        for document_id, error in report.failed:
            print(f"  - {document_id}: {error}")
