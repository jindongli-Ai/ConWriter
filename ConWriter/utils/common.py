"""Common helper utilities used across ConWriter."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List


def ensure_list(value: Any) -> List[Any]:
    """Normalize a value into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving readability."""
    return " ".join((text or "").split())


def short_text(text: str, max_chars: int = 180) -> str:
    """Return a shortened single-line preview string."""
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass instances into nested dictionaries."""
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Expected dataclass instance, got {type(obj)!r}")


def extract_sentences_with_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    """Extract simple sentence snippets that mention any keyword."""
    if not text:
        return []
    snippets: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    lowered = [k.lower() for k in keywords]
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        if any(k in s.lower() for k in lowered):
            snippets.append(s)
    return snippets


def slugify_token(text: str, fallback: str = "item") -> str:
    """Create a lightweight slug for IDs."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return cleaned or fallback

