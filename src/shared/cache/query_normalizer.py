"""检索缓存用的查询归一化工具。"""

from __future__ import annotations

_PUNCTUATION_MAP = str.maketrans(
    {
        "？": "?",
        "！": "!",
        "。": ".",
        "，": ",",
        "；": ";",
        "：": ":",
    }
)
_TRAILING_PUNCTUATION = "?!.,;:。？！，；："


def normalize_query(query: str) -> str:
    """对用户查询做保守归一化，用于精确缓存。"""
    normalized = " ".join(str(query).strip().split())
    normalized = normalized.translate(_PUNCTUATION_MAP).lower()
    return normalized.rstrip(_TRAILING_PUNCTUATION).strip()
