"""Shared metadata access helpers for retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

COLLEGE_FIELDS = ["college", "二级学院", "学院"]
MAJOR_FIELDS = ["major", "专业名称"]
SECTION_FIELDS = ["section", "章节"]
ACADEMY_FIELD = "academy"


def get_metadata_field(
    doc: Document,
    fields: list[str],
    default: Any = None,
) -> Any:
    """Get a metadata field value while supporting alias names."""
    for field in fields:
        if field in doc.metadata:
            return doc.metadata[field]
    return default
