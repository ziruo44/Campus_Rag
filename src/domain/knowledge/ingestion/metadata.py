"""Metadata extraction and enrichment utilities."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

from langchain_core.documents import Document


def extract_college_name(header: str) -> Optional[str]:
    """Extract the college name from a level-2 markdown header."""
    match = re.match(r"^##\s+(.+)$", header.strip())
    raw = match.group(1).strip() if match else None
    if not raw:
        return None
    cleaned = re.sub(r"^[\d\.]+、\s*", "", raw)
    return cleaned.strip() if cleaned else None


def extract_major_name(header: str) -> Optional[str]:
    """Extract the major name from a level-3 markdown header."""
    match = re.match(r"^###\s+(.+)$", header.strip())
    raw = match.group(1).strip() if match else None
    if not raw:
        return None
    cleaned = re.sub(r"^[（(]\d+[)）]", "", raw)
    cleaned = re.sub(r"^[\d\.]+、\s*", "", cleaned)
    cleaned = re.sub(r"^[、\s]+", "", cleaned)
    return cleaned.strip() if cleaned else None


def extract_section_name(header: str) -> Optional[str]:
    """Extract the section name from a level-4 markdown header."""
    match = re.match(r"^####\s+(.+)$", header.strip())
    return match.group(1).strip() if match else None


def compute_parent_id(college: str, major: str = "") -> str:
    """Compute a stable parent identifier from college and optional major."""
    combined = f"{college}:{major}" if major else college
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def enrich_metadata(doc: Document, doc_level: str) -> Document:
    """Enrich a parent document with extracted metadata."""
    content = doc.page_content
    lines = content.split("\n")

    college_name = None
    major_name = None
    section_name = None

    for line in lines:
        if not line.strip():
            continue
        if college_name is None:
            college_name = extract_college_name(line)
        if major_name is None:
            major_name = extract_major_name(line)
        if section_name is None:
            section_name = extract_section_name(line)
        if college_name and major_name and section_name:
            break

    if doc_level == "college":
        doc.metadata.update(
            {
                "doc_type": "parent",
                "doc_level": "college",
                "college": college_name,
                "parent_id": compute_parent_id(college_name) if college_name else None,
            }
        )
    else:
        doc.metadata.update(
            {
                "doc_type": "parent",
                "doc_level": "major",
                "college": college_name,
                "major": major_name,
                "section": section_name,
                "parent_id": (
                    compute_parent_id(college_name, major_name)
                    if college_name and major_name
                    else None
                ),
            }
        )

    return doc


def enrich_child_metadata(
    doc: Document,
    parent_id: str,
    chunk_index: int,
    college: str,
    major: str,
    section: str,
) -> Document:
    """Enrich a child chunk with inherited metadata."""
    doc.metadata.update(
        {
            "doc_type": "child",
            "doc_level": "major",
            "parent_id": parent_id,
            "chunk_index": chunk_index,
            "college": college,
            "major": major,
            "section": section,
        }
    )
    return doc
