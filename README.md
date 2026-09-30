# MemoryLens v0.1

MemoryLens is a baseline Retrieval-Augmented Generation application over a controlled corpus of 30 public Micron memory and storage sources. It builds the corpus locally, creates free local embeddings and a persistent FAISS index, retrieves dense top-k evidence, and asks Gemini to answer with grounded citations.

## Architecture

```text
              sources.yaml
                   │
                   ↓
          Approved Micron URLs
                   │
          ┌────────┴────────┐
          ↓                 ↓
      Web pages            PDFs
          ↓                 ↓
      Markdown         Original PDF
          └────────┬────────┘
                   ↓
             Local Corpus
                   ↓
         Metadata + Chunking
                   ↓
       Sentence Transformers
                   ↓
                  FAISS
                   ↓
           Dense Top-K Search
                   ↓
          Retrieved Evidence
                   ↓
           Gemini 3.8 Flash
                   ↓
          Answer + Citations
```

The canonical dataset is [`config/sources.yaml`](config/sources.yaml): 10 DRAM, 10 NAND, 5 NOR, and 5 technical sources. MemoryLens never crawls outside those URLs.

## Requirements

- Python 3.10 or newer (3.11 recommended)
- Internet access for initial corpus and embedding-model downloads
- A Gemini Developer API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

The answer model is configured through `GEMINI_MODEL`; application logic does not hard-code the model. Embeddings run locally with Sentence Transformers and do not use a paid embedding API.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template and add your API key:

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

The defaults are:

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

When the primary model is temporarily unavailable, quota-limited, unsupported, or missing, MemoryLens retries it with exponential backoff and then tries only the ordered models explicitly listed in `GEMINI_FALLBACK_MODELS`. It displays the model that generated the answer. It never discovers or chooses an unlisted model, and an empty fallback list disables fallback behavior.

## Build everything

```bash
python scripts/setup_corpus.py
```

This command:

```text
fetches the approved Micron sources
→ converts webpages to Markdown
→ downloads and preserves original PDFs
→ validates the corpus
→ performs section-aware chunking
→ generates local embeddings
→ creates a persistent FAISS index
```

Existing nonempty corpus files are reused. To refresh them:

```bash
python scripts/setup_corpus.py --force
```

For debugging, run the stages separately:

```bash
python scripts/build_corpus.py
python scripts/ingest.py
```

The builder reports configured, successful, failed, Markdown, and PDF counts. A failed URL is logged and processing continues. Validation reports category counts, missing and empty documents, and duplicate IDs before ingestion. An explicitly reported partial corpus may still be indexed; empty documents and duplicate IDs stop ingestion.

## Run

```bash
streamlit run app.py
```

The Streamlit UI provides a question box, example queries, generated answer, linked citations to original Micron URLs, and an expandable panel containing the retrieved chunks and similarity scores.

## How the corpus is represented

Web pages are fetched with a descriptive user agent and conservative timeouts. Navigation, footers, scripts, cookie elements, and common unrelated-content blocks are removed. The remaining main/article content is converted directly to Markdown—without LLM rewriting—and receives YAML front matter.

PDFs remain byte-for-byte original files. A neighboring `.metadata.yaml` sidecar stores their provenance. During ingestion, PyMuPDF extracts nonempty pages and retains one-based page numbers.

Every document retains:

```text
document_id, title, source_url, source, category, product_family,
technology, document_type, source_type, retrieved_at, file_path
```

Chunks additionally retain deterministic IDs, page, section, subsection, content type, and flags for numeric specifications, part numbers, and tables. Markdown is first split by heading hierarchy; oversized sections and PDF pages are then split with configurable overlap.

## Retrieval and generation

`scripts/ingest.py` encodes every chunk with the configured local Sentence Transformers model, L2-normalizes vectors, and writes an inner-product FAISS index plus JSONL chunk metadata under `data/vectorstore/`. Streamlit only loads this index; it never rebuilds it.

At query time, the same embedding model encodes the question. The top `TOP_K` chunks are formatted as numbered evidence blocks containing title, document ID, section/page, URL, and text. Only those blocks and the question are sent through the official `google-genai` SDK. The prompt requires evidence-only answers, exact units, explicit insufficiency, and numbered claim citations.

## Project layout

```text
app.py                     Streamlit UI
config/sources.yaml        Reproducible 30-source manifest
corpus/                    Generated Markdown and preserved PDFs
data/vectorstore/          Persistent FAISS index and chunk metadata
scripts/build_corpus.py    Corpus acquisition
scripts/ingest.py          Validation, loading, chunking, embeddings, FAISS
scripts/setup_corpus.py    End-to-end orchestration
src/                       Pipeline implementation
```

## v0.1 boundary

MemoryLens v0.1 establishes the baseline dense-retrieval system.

It intentionally does **not** contain:

```text
BM25
Hybrid retrieval
Reciprocal Rank Fusion
Cross-encoder reranking
Query routing
Advanced metadata filtering
Retrieval evaluation
```

These will be introduced incrementally so their effect on retrieval quality can later be measured against the v0.1 baseline.
