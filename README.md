# MemoryLens v1.0

MemoryLens is an advanced Retrieval-Augmented Generation project over a curated
corpus of publicly available Micron technical PDFs covering DRAM, NAND, NOR,
AI, and high-performance computing. The final version adds a reproducible
retrieval benchmark, latency analysis, answer-review workflow, exportable
reports, and a saved-results Streamlit dashboard to the v0.3 application.

MemoryLens is an independent educational project. It is not affiliated with or
endorsed by Micron Technology.

## Architecture

```text
Curated Micron PDFs
        │
        ▼
PyMuPDF text + table extraction
        │
        ▼
Canonical text/table chunks
        │
   ┌────┴────┐
   ▼         ▼
 FAISS      BM25
   └────┬────┘
        ▼
Reciprocal Rank Fusion
        ▼
Local cross-encoder reranking
        ▼
Context assembly → Gemini → answer + validated citation markers
        │
        ▼
Frozen benchmark, ranking metrics, latency, error analysis, dashboard
```

The application preserves four independently evaluated configurations:

| UI name | Evaluation name | Pipeline |
|---|---|---|
| Dense | Baseline | Sentence Transformer → FAISS |
| BM25 | Sparse | Technical tokenizer → BM25 |
| Hybrid | Hybrid | FAISS + BM25 → RRF |
| Hybrid + Reranker | Advanced | FAISS + BM25 → RRF → cross-encoder |

Advanced remains the default. All modes use the same 343 canonical chunks and
metadata constraints. The advanced pipeline retrieves 20 candidates and reranks
them locally with `cross-encoder/ms-marco-MiniLM-L-6-v2`; its scores are ranking
scores, not calibrated probabilities.

## Frozen evaluation dataset

[`evaluation/datasets/golden.jsonl`](evaluation/datasets/golden.jsonl) contains:

- 60 answerable technical questions;
- 10 each for semantic, exact terminology, numerical, comparison, table, and
  cross-document retrieval;
- six separate unanswerable controls;
- graded chunk relevance (`0`, `1`, `2`), reference answers, notes, difficulty,
  and actual canonical document/chunk IDs.

All references validate against the current 16-document, 343-chunk corpus.
`review_status: corpus_verified` means the label and answer were checked against
the stored evidence passage during implementation. It does not represent
independent human sign-off. The repository deliberately retains that distinction
instead of presenting synthetic or self-reviewed labels as human ground truth.

[`evaluation/manifest.json`](evaluation/manifest.json) freezes the benchmark
hash, PDF hashes, document IDs, canonical chunk digest, chunking parameters,
embedding and reranker models, tokenizer version, FAISS version, and retrieval
candidate configuration. Evaluation stops when the current corpus or settings
do not match this manifest, preventing obsolete chunk judgments from being used
silently.

## Retrieval metrics

The runner reports per-question values, aggregate mean/variability, and category
breakdowns for:

- Recall@3, Recall@5, and Recall@10;
- Mean Reciprocal Rank;
- nDCG@5 and nDCG@10 using gain `2^relevance - 1` and discount
  `log2(rank + 1)`.

Questions without relevant chunks are excluded from ranking aggregates. They
are handled by the answer-abstention review workflow.

## Measured results

The checked-in run used two warmed steady-state retrieval runs for each of 60
questions and each of four modes on CPU. Model/runtime loading and the 2.5-second
warmup were recorded separately from steady-state retrieval latency.

| Method | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.417 | 0.531 | 0.683 | 0.450 | 0.415 | 0.470 | 27.2 | 40.2 |
| BM25 | 0.533 | 0.639 | 0.828 | 0.547 | 0.507 | 0.571 | 2.7 | 3.8 |
| Hybrid | 0.511 | 0.689 | 0.789 | 0.511 | 0.519 | 0.551 | 28.1 | 39.6 |
| Hybrid + Reranker | 0.583 | 0.733 | 0.864 | 0.593 | 0.582 | 0.632 | 1,603.6 | 2,181.7 |

For this frozen benchmark, the cross-encoder improved the overall ranking
metrics relative to Hybrid but increased median retrieval latency by about 57×.
The outcome is not uniformly better:

- Hybrid Recall@5 was `0.670` for comparison questions versus `0.620` for
  reranking.
- Dense and Hybrid Recall@5 were `0.470` and `0.520` for cross-document
  questions versus `0.330` for reranking.
- Reranking performed strongly on table questions (`1.000` Recall@5) and led
  the numerical, semantic, and overall aggregates.
- BM25 was both fast and competitive for exact terminology and numerical
  questions.

These measurements describe this corpus, benchmark, model configuration, and
machine only. They are not a general claim that one retrieval architecture is
universally superior. See
[`evaluation/reports/latest.md`](evaluation/reports/latest.md) for methodology,
category results, representative failures, and evidence excerpts.

## Answer and citation evaluation

Retrieval quality does not establish answer correctness. The separate review
workflow creates a balanced 30-item queue: 24 answerable questions distributed
across the four retrieval modes and six advanced-mode abstention controls.

The rubric scores factual correctness, faithfulness to retrieved evidence,
substantive citation support, and abstention on unsupported questions.
Citation-marker integrity is checked automatically, but evidentiary support
requires a named reviewer. The checked-in queue remains explicitly
`pending_human_review`; no human-review scores are fabricated. All 30 generation
records were populated using the configured fallback chain, all retained citation
markers passed source-range validation, and all six unanswerable outputs explicitly
reported insufficient evidence. Those automatic observations are not substitutes
for the rubric scores.

```powershell
# Create a blank review queue
python scripts/evaluate_answers.py

# Populate the queue with actual Gemini answers before human review
python scripts/evaluate_answers.py --generate

# After a reviewer fills scores, produce the review summary
python scripts/evaluate_answers.py --summarize
```

## Streamlit application

The final UI has three areas:

- **Ask** — query the corpus using any retrieval mode or compare all four.
- **Inspect** — review the last answer, source passages, canonical IDs, metadata,
  scores, citations, query analysis, and stage latency.
- **Evaluate** — explore the saved benchmark summary, metric/latency charts,
  category results, and per-question rankings with relevance grades.

The dashboard loads `evaluation/results/latest.json`; it never reruns the full
benchmark during a Streamlit refresh.

## Reproducible setup

Python 3.11 or newer is recommended.

```powershell
git clone https://github.com/KautilyaMind/Memorylens.git
cd Memorylens
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `GEMINI_API_KEY` to `.env`. Only the explicitly configured primary and
fallback Gemini models are used.

Download and validate the curated corpus, then build canonical chunks and both
indexes:

```powershell
python scripts/download_corpus.py
python scripts/ingest.py
```

Or use the combined corpus workflow:

```powershell
python scripts/setup_corpus.py
```

Validate the frozen evaluation manifest:

```powershell
python scripts/evaluate_retrieval.py --manifest-only
```

To intentionally freeze a newly reviewed corpus/benchmark snapshot:

```powershell
python scripts/evaluate_retrieval.py --write-manifest --manifest-only
```

Do this only after reviewing any chunking or relevance-judgment changes.

## Run

```powershell
streamlit run app.py
```

### Streamlit Community Cloud

Use `app.py` as the main file on the `main` branch. The repository includes the
frozen canonical chunks plus matching FAISS and BM25 artifacts under `data/`, so
the hosted application can start without downloading the source PDFs or running
ingestion during deployment. Add `GEMINI_API_KEY` and any approved model settings
to the app's Streamlit Secrets. The PDFs, local `.env`, and model caches remain
excluded from Git.

End-to-end command-line smoke test:

```powershell
python scripts/smoke_rag.py --mode hybrid_rerank "What bandwidth does the HBM3E document specify?"
```

Run a fresh benchmark and regenerate JSON, CSV, and Markdown reports:

```powershell
python scripts/evaluate_retrieval.py --repeats 2 --depth 10
```

For a completely offline run after both local models are cached:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python scripts/evaluate_retrieval.py --repeats 2 --depth 10
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

The 25-test suite covers ingestion, deterministic chunk IDs, index alignment,
Gemini fallback behavior, technical tokenization, table extraction, reranking,
query analysis, citations, Recall@K, MRR, graded nDCG, benchmark validation,
manifest compatibility, mode consistency, unanswerable items, and result
serialization. Model predictions are mocked in unit tests.

## Project progression

| Version | Main contribution |
|---|---|
| v0.1 | Curated PDF corpus and dense FAISS RAG |
| v0.2 | BM25, hybrid retrieval, RRF, and metadata filtering |
| v0.3 | Cross-encoder reranking, query analysis, table handling, and context improvements |
| v1.0 | Frozen evaluation dataset, ranking/latency benchmark, error analysis, answer-review workflow, and final dashboard |

## Known limitations

- The 60 relevance sets were verified against stored corpus passages but have
  not received independent human assessor sign-off.
- The generated answer-review queue requires human scoring; automatic citation
  syntax validation is not citation correctness.
- PyMuPDF table extraction is best effort; complex layouts can remain noisy,
  although original page text is retained.
- Cross-encoder CPU latency is substantial relative to BM25 and Hybrid.
- The corpus contains 16 vendor-authored documents and does not represent the
  full memory/storage domain.
- No production deployment, commercial adoption, or paid evaluation service is
  claimed.

MemoryLens intentionally excludes agents, LangGraph, paid reranking APIs,
Kubernetes, and the addition of unmeasured retrieval architectures.
