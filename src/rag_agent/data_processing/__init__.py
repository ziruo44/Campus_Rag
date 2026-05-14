"""Data processing module for RAG knowledge base.

Handles document loading, chunking, and metadata extraction.
"""
from .loader import load_document, load_documents
from .chunker import chunk_documents, split_by_college
from .metadata import enrich_metadata

__all__ = ["load_document", "load_documents", "chunk_documents", "split_by_college", "enrich_metadata"]
