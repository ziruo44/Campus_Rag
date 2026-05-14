"""Retrieval-specific settings"""

from pydantic_settings import BaseSettings


class RetrievalSettings(BaseSettings):
    """Settings for hybrid retrieval"""

    vector_k: int = 5  # Number of results from vector search
    bm25_k: int = 5  # Number of results from BM25 search
    rrf_k: int = 60  # RRF smoothing parameter
    default_top_k: int = 3  # Default number of final results

    class Config:
        env_prefix = "RETRIEVAL_"