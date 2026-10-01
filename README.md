# MemoryLens v0.3

MemoryLens is a Retrieval-Augmented Generation system over a curated corpus of
public Micron technical PDFs covering DRAM, NAND, NOR, and memory-related AI/HPC
technologies. Version 0.3 preserves every v0.2 retrieval mode and adds local
cross-encoder reranking, deterministic query analysis, technical-table chunks,
improved context assembly, citation-marker validation, and retrieval latency
instrumentation.

Gemini receives only the selected evidence and retains the explicitly configured
free-model fallback chain. MemoryLens does not use an LLM for query analysis,
reranking, ingestion, or routing.

## Architecture

```text
                   User Query
                       ↓
              Lightweight Query Analysis
                       ↓
                Metadata Filters
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
         Dense Search       BM25 Search
            FAISS               ↓
              └────────┬────────┘
                       ↓
             Reciprocal Rank Fusion
                       ↓
              Top 20 Candidates
                       ↓
             Local Cross-Encoder
                       ↓
                Best 5 Chunks
                       ↓
              Context Assembly
                       ↓
                    Gemini
                       ↓
       Grounded Answer + Validated Citations
```

Ingestion uses one canonical chunk dataset for both indexes:

```text
Curated Micron PDFs
         ↓
    PDF Extraction
     ┌───┴───┐
     ↓       ↓
   Text    Tables
     └───┬───┘
         ↓
 Canonical JSONL Chunks
     ┌───┴───┐
     ↓       ↓
   FAISS    BM25
```

Chunk counts, canonical IDs, ordering, and SHA-256 identity digests are checked
across canonical JSONL, FAISS, and BM25 data.

## Retrieval modes

The Streamlit sidebar supports four comparable modes:

| Mode | Pipeline |
|---|---|
| Dense | Local bi-encoder → FAISS |
| BM25 | Technical tokens → BM25 |
| Hybrid | Dense + BM25 → RRF |
| Hybrid + Reranker | Dense + BM25 → RRF → local cross-encoder |

`Hybrid + Reranker` is the v0.3 default. Metadata constraints for category,
product family, technology, and document type are applied before candidate-list
truncation. Query analysis never silently converts an inferred product family
into a filter.

## Bi-encoder versus cross-encoder

The existing Sentence Transformer is a bi-encoder: it embeds the query and every
chunk independently, making FAISS candidate retrieval efficient. The configured
cross-encoder processes each query-passage pair jointly, allowing it to model
their interaction more closely. Because that is more computationally expensive,
it reranks only a small candidate pool instead of replacing FAISS or BM25.

The default local model is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

It is a compact, established passage-ranking baseline supported directly by
Sentence Transformers and works on CPU. The score is used only for ordering; it
is not presented as a calibrated probability. The model loads lazily once per
application process.

## Why retrieve 20 and rerank 5?

Initial dense and sparse retrieval favors broad candidate coverage. Reciprocal
Rank Fusion combines the two rankings without adding incompatible raw scores.
The cross-encoder then attempts to improve precision inside the top 20 before
the best five chunks are assembled for Gemini. Different ordering is not itself
proof of better retrieval; v1.0 will measure that with a golden dataset.

## Query analysis and technical matching

[`src/query_analysis.py`](src/query_analysis.py) deterministically recognizes:

- technologies and identifiers such as `HBM3E`, `LPDDR5X`, `GDDR7`, and `MT25QU`;
- specifications such as `8800 MT/s`, `1.2 TB/s`, and `24 GB`;
- comparison questions;
- conceptual questions;
- likely product-family labels for debugging only.

BM25 normalization preserves technical distinctions while matching reasonable
variants. For example, `8800 MT/s`, `8800MT/s`, and `8800 MTps` produce compatible
numeric/unit tokens. `DDR5-8800` retains the compound identifier and also emits
`DDR5` and `8800`. No stemming is applied to product codes.

## Why table handling matters

Technical specifications are frequently represented as tables. PyMuPDF inspects
each page for reliable table geometry. Extracted tables retain their headers,
row relationships, units, page, title/caption when available, document metadata,
and Micron URL. Rows become structured text such as:

```text
Columns: Parameter | Value
Parameter: Data Rate | Value: 8800 MT/s
```

Table chunks use deterministic IDs such as `MICRON-DRAM-001_TABLE_0001` and are
indexed by both FAISS and BM25. Regular page text is always preserved. Failed or
unreliable table extraction is logged and never causes the original page text to
be discarded. Perfect extraction from every PDF is not claimed.

## Context, citations, and grounding

After retrieval, highly overlapping chunks from the same page and content type
are reduced while relevant evidence from the same document remains allowed. A
configurable character limit controls free-tier context usage. Each evidence
block includes its canonical chunk ID, document title, page, section, content
type, original URL, and unchanged content.

Gemini is instructed to preserve values, units, qualifiers, product associations,
and distinctions between measured, theoretical, typical, maximum, and advertised
specifications. Returned numeric citation markers are checked against the actual
evidence list. Invalid markers are removed and reported in the UI. This validates
marker existence, not whether a passage logically proves a claim.

## Why latency matters

Cross-encoder inference adds work. MemoryLens records query-analysis, dense,
BM25, RRF, reranking, total retrieval, Gemini-generation, and end-to-end latency
with a consistent monotonic clock. The Streamlit debug panel exposes these values
without mixing Gemini time into retrieval-only latency.

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `GEMINI_API_KEY` to `.env`. Download the corpus if necessary, then rebuild
the canonical chunks and both indexes:

```powershell
python scripts/download_corpus.py
python scripts/ingest.py
```

Or run the combined workflow:

```powershell
python scripts/setup_corpus.py
```

Cross-encoder reranking needs no separate index; its model downloads on the first
reranked query and is then cached locally.

## Run

```powershell
streamlit run app.py
```

The UI shows provenance, content type, dense/BM25/RRF/reranker ranks and scores,
query analysis, latency, and retrieved text. The retrieval-only comparison view
shows all four top-five lists without calling Gemini.

Command-line smoke test:

```powershell
python scripts/smoke_rag.py --mode hybrid_rerank "How does HBM3E improve AI inference?"
```

## Configuration

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite
GEMINI_MAX_RETRIES=1
GEMINI_RETRY_BASE_SECONDS=1.0

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_MODE=hybrid_rerank
TOP_K=5
DENSE_CANDIDATES=20
BM25_CANDIDATES=20
RRF_K=60

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_CANDIDATES=20
RERANK_TOP_K=5
RERANK_BATCH_SIZE=8

ENABLE_QUERY_ANALYSIS=true
ENABLE_TABLE_EXTRACTION=true
CONTEXT_MAX_CHARS=16000
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

The deterministic suite mocks cross-encoder predictions and also creates a local
PDF table to verify extraction without network or model downloads.

## Roadmap

```text
v0.1
Curated Technical Corpus
+ Dense FAISS RAG

         ↓

v0.2
BM25 + Dense
+ Hybrid Retrieval
+ RRF
+ Metadata Filtering

         ↓

v0.3
Cross-Encoder Reranking
+ Query Analysis
+ Technical Table Handling
+ Improved Context Assembly

         ↓

v1.0
Golden Evaluation Dataset
+ Recall@K
+ MRR
+ nDCG
+ Answer Faithfulness
+ Citation Correctness
+ Latency Comparison
```

MemoryLens v0.3 intentionally excludes agents, LangGraph, paid reranking APIs,
LLM reranking, query expansion, HyDE, multi-query retrieval, and the full v1.0
evaluation framework.
