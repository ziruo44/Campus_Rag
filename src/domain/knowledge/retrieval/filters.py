"""Keyword-based query filter extraction for retrieval."""

from __future__ import annotations

from domain.knowledge.retrieval.hybrid_search import HybridRetriever


def extract_query_filters(retriever: HybridRetriever, query: str) -> dict[str, str]:
    """Extract college/major filters from a query using retriever metadata."""
    filters: dict[str, str] = {}

    for college in sorted(retriever.colleges, key=len, reverse=True):
        if college and college in query:
            filters["college"] = college
            break

    for major in sorted(retriever.majors, key=len, reverse=True):
        if major and major in query:
            filters["major"] = major
            break

    return filters
