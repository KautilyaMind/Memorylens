from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.dataset import load_benchmark, validate_benchmark
from src.evaluation.manifest import (
    build_evaluation_manifest,
    load_manifest,
    validate_manifest,
)
from src.chunk_store import load_chunks
from src.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark all MemoryLens retrieval modes")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "datasets" / "golden.jsonl")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "evaluation" / "manifest.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--write-manifest", action="store_true", help="Freeze the current compatible corpus/dataset manifest")
    parser.add_argument("--manifest-only", action="store_true", help="Validate/write the manifest without running retrieval")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest_only:
        chunks = load_chunks(settings.chunks_dir)
        questions = load_benchmark(args.dataset)
        summary = validate_benchmark(questions, chunks)
        current_manifest = build_evaluation_manifest(chunks, args.dataset)
        if args.write_manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")
            print(f"Wrote frozen evaluation manifest: {args.manifest}")
        if not args.manifest.exists():
            raise FileNotFoundError(
                f"Evaluation manifest not found: {args.manifest}. Run with --write-manifest."
            )
        mismatches = validate_manifest(load_manifest(args.manifest), current_manifest)
        if mismatches:
            raise RuntimeError(
                "Frozen benchmark is incompatible with the current corpus/index configuration: "
                + ", ".join(mismatches)
            )
        print(json.dumps(summary, indent=2))
        print("Evaluation manifest is compatible with the current corpus and indexes.")
        return
    from src.evaluation.runner import run_retrieval_benchmark, save_results
    from src.runtime import load_retrieval_runtime

    started = time.perf_counter()
    retriever, chunks, _ = load_retrieval_runtime()
    runtime_load_ms = (time.perf_counter() - started) * 1000
    questions = load_benchmark(args.dataset)
    summary = validate_benchmark(questions, chunks)
    current_manifest = build_evaluation_manifest(chunks, args.dataset)
    if args.write_manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")
        print(f"Wrote frozen evaluation manifest: {args.manifest}")
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: {args.manifest}. Run once with --write-manifest."
        )
    mismatches = validate_manifest(load_manifest(args.manifest), current_manifest)
    if mismatches:
        raise RuntimeError(
            "Frozen benchmark is incompatible with the current corpus/index configuration: "
            + ", ".join(mismatches)
        )
    print(f"Validated {summary['answerable_count']} answerable and {summary['unanswerable_count']} unanswerable questions")
    results = run_retrieval_benchmark(
        retriever,
        questions,
        chunks,
        repeats=args.repeats,
        depth=args.depth,
        warmup=not args.no_warmup,
    )
    results["runtime_load_ms"] = runtime_load_ms
    results["evaluation_manifest"] = current_manifest
    paths = save_results(results, args.output)
    for mode, aggregate in results["aggregates"].items():
        metrics = aggregate["metrics"]
        latency = aggregate["latency_ms"]["total_retrieval_ms"]
        print(
            f"{mode:15} Recall@5={metrics['recall_at_5']['mean']:.3f} "
            f"MRR={metrics['mrr']['mean']:.3f} nDCG@5={metrics['ndcg_at_5']['mean']:.3f} "
            f"median={latency['median']:.1f}ms p95={latency['p95']:.1f}ms"
        )
    print(f"JSON: {paths['json']}")
    print(f"CSV: {paths['csv']}")
    print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()
