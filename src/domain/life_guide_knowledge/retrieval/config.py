"""Settings for life guide retrieval."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LifeGuideRetrievalSettings(BaseSettings):
    """Settings for life guide hybrid retrieval"""

    model_config = SettingsConfigDict(
        env_prefix="LIFE_GUIDE_RETRIEVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vector_k: int = 5  # Number of results from vector search
    bm25_k: int = 3  # Number of results from BM25 search
    rrf_k: int = 60  # RRF smoothing parameter
    default_top_k: int = 3  # Default number of final results