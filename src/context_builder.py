from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class ContextBundle:
    evidence: str
    results: list[dict[str, Any]]


def _overlap(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(left.lower()))
    right_tokens = set(TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _context_block(result: dict[str, Any], marker: int) -> str:
    return "\n".join(
        [
            f"[{marker}]",
            f"Source ID: {result.get('chunk_id', '')}",
            f"Document Title: {result.get('title', 'Untitled')}",
            f"Page: {result.get('page', '')}",
            f"Section: {result.get('section', '')}",
            f"Content Type: {result.get('content_type', 'text')}",
            f"Source URL: {result.get('source_url', '')}",
            "Retrieved Content:",
            result["text"],
        ]
    )


def build_context(
    results: list[dict[str, Any]], max_chars: int = 16000
) -> ContextBundle:
    if max_chars <= 0:
        raise ValueError("CONTEXT_MAX_CHARS must be positive")
    selected: list[dict[str, Any]] = []
    blocks: list[str] = []
    used = 0
    for result in results:
        duplicate = any(
            result.get("document_id") == prior.get("document_id")
            and result.get("page") == prior.get("page")
            and result.get("content_type") == prior.get("content_type")
            and _overlap(result["text"], prior["text"]) >= 0.88
            for prior in selected
        )
        if duplicate:
            continue
        marker = len(selected) + 1
        block = _context_block(result, marker)
        extra = len(block) + (2 if blocks else 0)
        if blocks and used + extra > max_chars:
            break
        if not blocks and extra > max_chars:
            block = block[:max_chars]
            extra = len(block)
        selected.append({**result, "source_marker": marker})
        blocks.append(block)
        used += extra
    return ContextBundle("\n\n".join(blocks), selected)
