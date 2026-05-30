"""Settings for life guide knowledge indexing."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LifeGuideIndexingSettings(BaseSettings):
    """Settings for life guide vector indexing and embedding generation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str | None = None
    embedding_dimension: int = 768
    dashscope_api_key: str | None = None
    qwen_api_key: str | None = None

    @property
    def resolved_embedding_api_key(self) -> str | None:
        """Return the configured embedding API key with fallback order."""
        primary = (self.dashscope_api_key or "").strip()
        if primary:
            return primary

        fallback = (self.qwen_api_key or "").strip()
        return fallback or None