"""Knowledge ingestion package."""

from domain.knowledge.ingestion.chunker import (
    chunk_documents,
    split_by_college,
)
from domain.knowledge.ingestion.loader import (
    load_document,
    load_documents,
)
from domain.knowledge.ingestion.metadata import enrich_metadata

__all__ = [
    "chunk_documents",
    "enrich_metadata",
    "load_document",
    "load_documents",
    "split_by_college",
]
