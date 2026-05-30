"""Knowledge ingestion package."""

from domain.major_knowledge.ingestion.chunker import (
    chunk_documents,
    split_by_college,
)
from domain.major_knowledge.ingestion.loader import (
    load_document,
    load_documents,
)
from domain.major_knowledge.ingestion.metadata import enrich_metadata

__all__ = [
    "chunk_documents",
    "enrich_metadata",
    "load_document",
    "load_documents",
    "split_by_college",
]
