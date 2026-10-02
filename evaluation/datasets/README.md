# MemoryLens benchmark dataset

`golden.jsonl` contains 60 answerable retrieval questions (10 in each requested
category) plus six unanswerable questions evaluated separately. Every positive
judgment references a chunk in the frozen v0.3-derived canonical corpus.

Relevance grades are:

- `0`: not relevant
- `1`: partially relevant/supporting context
- `2`: highly relevant/direct evidence

`review_status: corpus_verified` means the question, reference answer, and IDs
were checked against the stored passage text during implementation. It does not
mean an independent human assessor has signed off. Independent review should
change the status to `human_reviewed` only after checking the evidence and must
not change labels merely to improve a retrieval method's score.

Unanswerable items have no relevance judgments and are excluded from Recall,
MRR, and nDCG aggregates. They are intended for the separate answer-abstention
review workflow.
