import re
from typing import List, Dict, Optional

AUTHORITY_PATTERNS = [
    # Regulations
    (
        r"\bRegulation\s*\(\s?(?:EU|EC|EEC)\s?\)\s*No\s*(?:\d+/\d{4}|\d{4}/\d+)",
        "REGULATION",
    ),
    (
        r"\bRegulation\s*\(\s?(?:EU|EC|EEC)\s?\)\s*(?:\d+/\d{4}|\d{4}/\d+)",
        "REGULATION",
    ),

    # Directives
    (
        r"\bDirective\s*\d{4}/\d{2,4}/?(?:EU|EC|EEC)?",
        "DIRECTIVE",
    ),
    (
        r"\bDirective\s*(?:EU|EC|EEC)\s*\d{4}/\d{2,4}",
        "DIRECTIVE",
    ),

    # Decisions
    (
        r"\bDecision\s*\(\s?(?:EU|EC|EEC)\s?\)\s*(?:\d+/\d{4}|\d{4}/\d+)",
        "DECISION",
    ),
    (
        r"\bDecision\s*(?:EU|EC|EEC)\s*(?:\d+/\d{4}|\d{4}/\d+)",
        "DECISION",
    ),
    (
        r"\b(?:Council\s+)?Decision\s+(?:\d{2,4}/\d{2,4}/(?:EU|EC|EEC))",
        "DECISION",
    ),

    # Articles
    (
        r"\bArticle\s+\d+[a-zA-Z]?",
        "ARTICLE",
    ),
    (
        r"\bArt\.\s*\d+[a-zA-Z]?",
        "ARTICLE",
    ),

    # Cases (simplified)
    (
        r"\bCase\s+[A-Z]\s*\d+/\d{2}",
        "CASE",
    ),
]

YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
YEAR_MIN = 1950
YEAR_MAX = 2025


def classify_level(auth_type: str) -> float:
    """
    Assign approximate lex superior levels:
    Higher = more authoritative.
    """
    if auth_type in {"REGULATION", "DIRECTIVE"}:
        return 3.0
    if auth_type in {"DECISION"}:
        return 2.5
    if auth_type in {"ARTICLE"}:
        return 2.0
    if auth_type in {"CASE"}:
        return 2.0
    return 1.0


def extract_year(text: str) -> Optional[int]:
    """
    Extract a 4-digit year (1950-2039) from authority text.
    Returns None if no year found.
    """
    matches = YEAR_PATTERN.findall(text)
    if not matches:
        return None
    # If multiple years, pick the largest (most recent)
    return max(int(y) for y in matches)


def normalize_recency(year: Optional[int]) -> float:
    """
    Normalize year into [0, 1], with 0.5 as sentinel for unknown.
    """
    if year is None:
        return 0.5
    recency = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    return max(0.0, min(1.0, recency))


def extract_authorities(text: str) -> List[Dict]:
    """
    Extract legal authority mentions with type, level, year, and recency.

    Returns list of:
    {
        "text": str,
        "type": str,
        "start": int,
        "end": int,
        "level": float,
        "year": Optional[int],
        "recency": float,
    }
    """
    results = []
    for pattern, auth_type in AUTHORITY_PATTERNS:
        for match in re.finditer(pattern, text):
            span_text = match.group(0)
            start, end = match.span()
            year = extract_year(span_text)
            results.append(
                {
                    "text": span_text,
                    "type": auth_type,
                    "start": start,
                    "end": end,
                    "level": classify_level(auth_type),
                    "year": year,
                    "recency": normalize_recency(year),
                }
            )
    return results
