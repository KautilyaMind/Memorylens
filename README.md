# MemoryLens v0.1

MemoryLens is a Retrieval-Augmented Generation system built over a curated corpus
of publicly available Micron technical documents across DRAM, NAND, NOR and
memory-related AI/HPC technologies. It asks Gemini to answer only from retrieved
evidence and provides page-aware citations.

The baseline favors source quality and inspectability over corpus size. It does
not crawl arbitrary webpages, scrape HTML, use hybrid search, rerank results,
call an LLM during ingestion, or maintain a database beyond local FAISS files.

## Corpus

[`config/documents.yaml`](config/documents.yaml) is the sole source of truth. It
currently identifies 16 authoritative technical PDFs across four categories:

- DRAM: HBM3E, GDDR7, LPDDR5X, LPDDR5X CAMM2, and DDR5
- NAND: Micron 9650, 2600, 2650, and 6550 ION NVMe SSDs
- NOR: a NOR/NAND technical guide and NOR product flyer
- Technical: AI/HPC memory and memory-centric data-center architecture

Every manifest entry has a stable document ID, exact local file name, category,
product family, technology, document type, publisher, and canonical source URL.
Downloaded PDFs are deliberately ignored by Git and can be reproduced from the
manifest.

Earlier ingestion designs considered converting public product webpages into
Markdown. That produced too much navigational and low-information content. The
final v0.1 instead uses a smaller curated set of authoritative product briefs,
white papers, technical guides, and reports.

## Architecture

```text
Official Micron Technical Documents
                 ↓
            PDF Corpus
                 ↓
          PDF Validation
                 ↓
          Text Extraction
                 ↓
       Page/Section Chunking
                 ↓
      Local Sentence Transformer
                 ↓
               FAISS
                 ↓
        Dense Top-K Retrieval
                 ↓
        Retrieved Evidence
                 ↓
         Gemini Generation
        primary + fallback
                 ↓
        Answer + Citations
```

PDF processing is deterministic. It removes repeated headers and footers,
standalone page numbers, normalizes whitespace, and rejoins wrapped prose without
using an LLM. Each chunk retains its document identity, title, source URL,
category, product family, technology, document type, file path, page, section,
content flags, and a stable per-document chunk ID.

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add a Gemini API key to `.env`. The configured primary model is tried first;
retryable capacity, quota, and model-availability errors automatically move to
the configured fallback models.

Build everything in one command:

```powershell
python scripts/setup_corpus.py
```

That command downloads missing PDFs, validates the available corpus, extracts
and cleans text, builds chunks and local embeddings, and replaces the FAISS
index. Existing valid PDFs are skipped; use `--force` to download them again.

The stages can also run separately:

```powershell
python scripts/download_corpus.py
python scripts/ingest.py
```

Validation reports configured, available, valid, missing, invalid, and unusually
short documents. A missing download is reported explicitly and does not prevent
the valid available subset from being indexed. An invalid or empty PDF does
prevent indexing until it is corrected or removed from the manifest.

## Run

```powershell
streamlit run app.py
```

The UI shows the indexed document and chunk counts, the answer, the generating
model, linked sources, page/section locations, and the retrieved context with
similarity scores and technical metadata.

For a command-line smoke test:

```powershell
python scripts/smoke_rag.py "How does DDR5 improve memory bandwidth?"
```

## Configuration

The important `.env` settings are:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite
GEMINI_MAX_RETRIES=1
GEMINI_RETRY_BASE_SECONDS=1.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

If `EMBEDDING_MODEL`, chunk size, or corpus content changes, rerun
`python scripts/ingest.py`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

```text
MemoryLens v0.1
=
Curated Technical Corpus
+
Dense Retrieval
+
FAISS
+
Gemini
+
Grounded Citations
```

Future versions may add BM25, hybrid retrieval, RRF, metadata filtering,
cross-encoder reranking, and retrieval evaluation. None are part of v0.1.
