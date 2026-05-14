"""Academy-Major mapping utilities for structured queries."""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Field aliases for metadata access
_COLLEGE_FIELDS = ["college", "二级学院"]
_MAJOR_FIELDS = ["major", "专业名称"]


def _get_field(doc: Document, fields: List[str], default=None):
    """Get field value from document metadata, trying multiple field names."""
    for field in fields:
        if field in doc.metadata:
            return doc.metadata[field]
    return default


def build_academy_majors_map(chunks: List[Document]) -> dict[str, List[str]]:
    """
    Build academy-to-majors mapping from chunks.

    Args:
        chunks: List of document chunks

    Returns:
        Dict mapping academy name to list of major names
    """
    academy_majors: dict[str, set[str]] = {}

    for chunk in chunks:
        college = _get_field(chunk, _COLLEGE_FIELDS)
        major = _get_field(chunk, _MAJOR_FIELDS)

        if college and major:
            if college not in academy_majors:
                academy_majors[college] = set()
            academy_majors[college].add(major)

    # Convert sets to sorted lists
    return {k: sorted(v) for k, v in academy_majors.items()}


def get_all_colleges(chunks: List[Document]) -> List[str]:
    """
    Get all unique college names.

    Args:
        chunks: List of document chunks

    Returns:
        Sorted list of college names
    """
    colleges: set[str] = set()
    for chunk in chunks:
        college = _get_field(chunk, _COLLEGE_FIELDS)
        if college:
            colleges.add(college)
    return sorted(colleges)


def get_majors_by_college(chunks: List[Document], college: str) -> List[str]:
    """
    Get all majors for a given college.

    Args:
        chunks: List of document chunks
        college: College name

    Returns:
        Sorted list of major names
    """
    majors: set[str] = set()
    for chunk in chunks:
        chunk_college = _get_field(chunk, _COLLEGE_FIELDS)
        major = _get_field(chunk, _MAJOR_FIELDS)
        if chunk_college == college and major:
            majors.add(major)
    return sorted(majors)