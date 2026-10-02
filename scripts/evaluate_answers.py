from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.chunk_store import load_chunks
from src.evaluation.answer_review import (
    load_review_queue,
    review_record,
    select_review_sample,
    summarize_reviews,
    write_review_queue,
)
from src.evaluation.dataset import load_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or generate the human answer-review sample")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation" / "datasets" / "golden.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "evaluation" / "results" / "answer-review.jsonl")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--generate", action="store_true", help="Call Gemini and populate answers; otherwise create a blank review queue")
    parser.add_argument("--summarize", action="store_true", help="Summarize an edited human-review queue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summary = summarize_reviews(load_review_queue(args.output))
        summary_path = args.output.with_name("answer-review-summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    questions = load_benchmark(args.dataset)
    sample = select_review_sample(questions, args.per_category)
    chunks = load_chunks(settings.chunks_dir)
    chunk_lookup = {str(chunk.metadata["chunk_id"]): chunk for chunk in chunks}
    records = []
    if args.generate:
        from src.gemini_client import GeminiClient
        from src.rag import RAGPipeline
        from src.runtime import load_retrieval_runtime

        retriever, _, _ = load_retrieval_runtime()
        pipeline = RAGPipeline(
            retriever,
            GeminiClient(
                settings.gemini_api_key,
                settings.gemini_model,
                settings.gemini_fallback_models,
                settings.gemini_max_retries,
                settings.gemini_retry_base_seconds,
            ),
            settings.context_max_chars,
        )
        for index, question in enumerate(sample, 1):
            print(
                f"[{index}/{len(sample)}] {question['question_id']} "
                f"({question['evaluation_mode']})"
            )
            response = pipeline.answer(
                question["question"], mode=question["evaluation_mode"]
            )
            records.append(review_record(question, response, chunk_lookup))
    else:
        records = [review_record(question, chunk_lookup=chunk_lookup) for question in sample]
    write_review_queue(records, args.output)
    print(f"Wrote {len(records)} review records to {args.output}")
    print("Human scores remain blank until a named reviewer completes the rubric.")


if __name__ == "__main__":
    main()
