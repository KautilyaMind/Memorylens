from __future__ import annotations

from typing import Any

FILTER_FIELDS = ("category", "product_family", "technology", "document_type")


def matches_filters(metadata: dict[str, Any], filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(
        not value
        or str(value).lower() == "any"
        or str(metadata.get(field, "")).casefold() == str(value).casefold()
        for field, value in filters.items()
        if field in FILTER_FIELDS
    )
