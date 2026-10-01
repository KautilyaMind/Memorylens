# MemoryLens v0.2

MemoryLens is a Retrieval-Augmented Generation system over a curated corpus of
public Micron technical documents across DRAM, NAND, NOR, and memory-related
AI/HPC technologies. Version 0.2 preserves the dense FAISS baseline and adds
local BM25 retrieval, Reciprocal Rank Fusion, metadata filtering, and retrieval
comparison tools.

The corpus still prioritizes authoritative product briefs, white papers,
technical guides, and reports over arbitrary document count. Gemini receives
only the final retrieved evidence and produces page-aware citations.

## Architecture

```text
                 Curated Technical Corpus
                           ↓
                    Canonical Chunks
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
        Dense Embeddings             Tokens
              ↓                         ↓
            FAISS                     BM25
              ↓                         ↓
      Semantic Retrieval       Lexical Retrieval
              └────────────┬────────────┘
                           ↓
                 Reciprocal Rank Fusion
                           ↓
                    Metadata Constraints
                           ↓
                      Final Top-K
                           ↓
                         Gemini
                           ↓
                  Answer + Citations
```

In the implementation, selected metadata constraints are applied before each
retriever truncates its candidate list. Dense search scores the complete FAISS
index and then keeps the highest-ranked eligible chunks; BM25 similarly ranks
the complete eligible subset. This prevents restrictive filters from being
limited to an unrelated initial candidate pool.

## Retrieval concepts

### Dense retrieval

FAISS searches normalized local Sentence Transformer embeddings. Dense search
is useful when a question and a document express similar meaning with different
words. Dense-only mode remains available as the v0.1 comparison baseline.

### BM25

BM25 is useful for exact words, technical codes, acronyms, product names, and
numerical specifications such as `HBM3E`, `MT25Q`, `GDDR7`, `8800`, and `MT/s`.
Tokenization lowercases for comparison but preserves letters, numbers, embedded
digits, slashes, dots, plus signs, and internal hyphens. It deliberately avoids
stemming and aggressive stop-word removal. BM25 tokenizes each chunk's title,
section heading, and body text so identifiers present in a heading remain
searchable without creating a second chunk dataset. A deliberately small list
removes generic query words such as “find,” “information,” and “related”; all
technical identifiers and units remain untouched.

### Hybrid retrieval and RRF

Hybrid mode independently retrieves larger dense and BM25 candidate pools and
combines their ranks using Reciprocal Rank Fusion:

```text
RRF(d) = Σ 1 / (RRF_K + rank(d))
```

RRF uses positions rather than directly adding incompatible FAISS and BM25 raw
scores. Chunks found by both systems generally gain a stronger fused score.

### Metadata filtering

The UI can constrain retrieval by category, product family, technology, or
document type. Choices come from the indexed chunk metadata. Filters are
explicit and user-selected; v0.2 does not use an LLM to infer them.

## Shared chunk and index storage

Ingestion creates each page/section-aware chunk once. The deterministic
`chunk_id` is the canonical identity used by both indexes.

```text
data/chunks/chunks.jsonl       flattened canonical chunk records
data/chunks/manifest.json      chunk count and ID digest
data/vectorstore/              FAISS index and aligned chunk metadata
data/bm25/index.json           tokenized corpus and chunk ID order
```

The BM25 object is rebuilt quickly in memory from its persisted tokens. Startup
validates chunk counts, IDs, ordering, and digests across canonical chunks,
FAISS, and BM25. Missing, corrupt, duplicate, or mismatched data produces a
clear rebuild error.

## Setup and indexing

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `GEMINI_API_KEY` to `.env`. Download the corpus when needed:

```powershell
python scripts/download_corpus.py
```

Build canonical chunks plus both indexes:

```powershell
python scripts/ingest.py
```

Or perform download, validation, and ingestion together:

```powershell
python scripts/setup_corpus.py
```

Ingestion validates the PDFs, extracts and cleans their text, generates one
canonical chunk dataset, embeds it locally, builds FAISS, tokenizes it for BM25,
and verifies that all chunk IDs match.

## Run

```powershell
streamlit run app.py
```

The sidebar selects Dense, BM25, or Hybrid retrieval and optional metadata
constraints. The retrieved-context panel displays chunk IDs, final ranks,
document metadata, pages, sections, dense scores/ranks, BM25 scores/ranks, RRF
scores, and text. Enable **Compare Retrieval Methods** to see all three top-five
lists for the same query and filters without calling Gemini.

The command-line smoke test also accepts a mode:

```powershell
python scripts/smoke_rag.py --mode hybrid "What does the documentation say about HBM3E?"
```

## Configuration

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite
GEMINI_MAX_RETRIES=1
GEMINI_RETRY_BASE_SECONDS=1.0
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_MODE=hybrid
TOP_K=5
DENSE_CANDIDATES=15
BM25_CANDIDATES=15
RRF_K=60
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

`TOP_K` is the final result count. The larger candidate settings feed hybrid
fusion. Changing the corpus, embedding model, or chunking configuration requires
another `python scripts/ingest.py` run.

## Useful comparison queries

- Semantic: `Why is high-bandwidth memory important for AI inference?`
- Exact technology: `What does the documentation say about HBM3E?`
- Exact identifier: `Find information related to MT25Q.`
- Numerical: `Which document discusses 8800 MT/s?`
- Mixed: `How do HBM3E characteristics support high-performance AI workloads?`

The corpus may not contain evidence for every example. An empty lexical match or
a metadata filter matching no chunks is reported rather than sent to Gemini as
an invitation to invent an answer.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## v0.2 boundary

```text
MemoryLens v0.2
=
Curated Technical Corpus
+
Dense FAISS Retrieval
+
BM25 Sparse Retrieval
+
Hybrid Retrieval
+
Reciprocal Rank Fusion
+
Metadata Filtering
+
Gemini Grounded Generation
```

The central engineering question is: **Can combining semantic similarity with
exact lexical matching retrieve better technical evidence than dense search
alone?**

Cross-encoder or LLM reranking, query expansion, query decomposition, multi-query
retrieval, HyDE, table-specific retrieval, agents, LangGraph, and advanced
benchmark evaluation remain deliberately excluded for v0.3 or later.
