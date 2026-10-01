from __future__ import annotations

import re
from typing import Any

TECHNOLOGY_RE = re.compile(
    r"\b(?:HBM\d*E?|LPDDR\d+X?|GDDR\d+|DDR\d+(?:-\d+)?|MRDIMM|QLC|TLC|SLC|NAND|NOR)\b",
    re.IGNORECASE,
)
PART_NUMBER_RE = re.compile(r"\b(?:MTFD|MTA|MT)[A-Z0-9-]{3,}\b", re.IGNORECASE)
NUMBER_WITH_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:MT\s*/?\s*s|MTPS|GT\s*/?\s*s|GB\s*/?\s*s|TB\s*/?\s*s|Gbps|GB|Gb|TB|V|mV|MHz|GHz|ns)\b",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(?:vs\.?|versus|compare|comparison|difference|differences|differ)\b",
    re.IGNORECASE,
)
SPECIFICATION_RE = re.compile(
    r"\b(?:bandwidth|data rate|speed|capacity|voltage|power|latency|temperature|specification|specify|maximum|minimum|typical)\b",
    re.IGNORECASE,
)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _normalize_numerical(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"(?i)mt\s*/?\s*s|mtps", "MT/s", value)
    value = re.sub(r"(?i)tb\s*/?\s*s", "TB/s", value)
    value = re.sub(r"(?i)gb\s*/?\s*s", "GB/s", value)
    return value


def analyze_query(query: str) -> dict[str, Any]:
    technical_terms = _unique(
        [match.group(0).upper() for match in TECHNOLOGY_RE.finditer(query)]
        + [match.group(0).upper() for match in PART_NUMBER_RE.finditer(query)]
    )
    numerical_terms = _unique(
        [_normalize_numerical(match.group(0)) for match in NUMBER_WITH_UNIT_RE.finditer(query)]
    )
    is_comparison = bool(COMPARISON_RE.search(query))
    if is_comparison:
        query_type = "comparison"
    elif numerical_terms or SPECIFICATION_RE.search(query):
        query_type = "specification"
    elif technical_terms:
        query_type = "exact_identifier"
    else:
        query_type = "conceptual"

    families: list[str] = []
    for term in technical_terms:
        if term.startswith("HBM"):
            families.append("HBM")
        elif term.startswith("LPDDR"):
            families.append("LPDDR")
        elif term.startswith("GDDR"):
            families.append("GDDR")
        elif term.startswith("DDR") or term == "MRDIMM":
            families.append("DDR")
        elif term in {"NAND", "QLC", "TLC", "SLC"}:
            families.append("NAND")
        elif term == "NOR" or PART_NUMBER_RE.fullmatch(term):
            families.append("NOR")

    return {
        "query_type": query_type,
        "technical_terms": technical_terms,
        "numerical_terms": numerical_terms,
        "is_comparison": is_comparison,
        "detected_product_families": _unique(families),
    }
