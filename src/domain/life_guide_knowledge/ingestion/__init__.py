"""Ingestion module for life guide knowledge."""

from .loader import load_life_guide
from .chunker import chunk_life_guide_all

__all__ = ["load_life_guide", "chunk_life_guide_all"]