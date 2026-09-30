from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import settings
from src.document_manifest import document_path, load_document_manifest

LOGGER = logging.getLogger(__name__)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


@dataclass
class DownloadReport:
    configured: int
    available: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def _is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


class CorpusDownloader:
    def __init__(self, corpus_dir: Path = settings.corpus_dir, delay: float = 0.25):
        self.corpus_dir = corpus_dir
        self.delay = delay
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.75,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Referer": "https://www.micron.com/"}
        )

    def download(self, force: bool = False) -> DownloadReport:
        documents = load_document_manifest()
        report = DownloadReport(configured=len(documents))
        for index, entry in enumerate(documents):
            destination = document_path(entry, self.corpus_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if _is_pdf(destination) and not force:
                    report.skipped += 1
                    LOGGER.info("Already available: %s", destination.name)
                else:
                    self._download_one(str(entry["source_url"]), destination)
                    report.downloaded += 1
                report.available += 1
            except Exception as exc:
                LOGGER.error("Failed %s: %s", entry["document_id"], exc)
                report.failed.append((str(entry["document_id"]), str(exc)))
            if index < len(documents) - 1 and self.delay:
                time.sleep(self.delay)
        return report

    def _download_one(self, url: str, destination: Path) -> None:
        LOGGER.info("Downloading %s", destination.name)
        response = self.session.get(
            url,
            headers={"Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"},
            timeout=(10, 60),
            allow_redirects=True,
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF-"):
            raise ValueError("server response is not a PDF")
        temporary = destination.with_suffix(".pdf.part")
        temporary.write_bytes(content)
        temporary.replace(destination)


def print_download_report(report: DownloadReport) -> None:
    print(f"Documents configured: {report.configured}")
    print(f"Documents available: {report.available}")
    print(f"Downloaded this run: {report.downloaded}")
    print(f"Already available: {report.skipped}")
    print(f"Failed: {len(report.failed)}")
    for document_id, error in report.failed:
        print(f"  - {document_id}: {error}")
