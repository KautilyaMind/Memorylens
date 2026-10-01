from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[(?:Source\s+)?(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)


def validate_citation_markers(answer: str, source_count: int) -> tuple[str, list[int]]:
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        markers = [int(value.strip()) for value in match.group(1).split(",")]
        valid = [value for value in markers if 1 <= value <= source_count]
        invalid.extend(value for value in markers if value not in valid)
        if not valid:
            return ""
        return "[" + ", ".join(str(value) for value in valid) + "]"

    cleaned = CITATION_RE.sub(replace, answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned), sorted(set(invalid))
