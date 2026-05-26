"""Shared chat-model settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatModelSettings(BaseSettings):
    """Settings for the upstream chat model provider."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qwen_model: str = "qwen3.5-plus-2026-04-20"
    qwen_base_url: str | None = None
    qwen_api_key: str | None = None

    @property
    def normalized_model(self) -> str:
        """Return the configured model name with whitespace normalized."""
        return self.qwen_model.strip() or "qwen3.5-plus-2026-04-20"

    @property
    def normalized_base_url(self) -> str | None:
        """Return the configured base URL if present."""
        if self.qwen_base_url is None:
            return None
        value = self.qwen_base_url.strip()
        return value or None

    @property
    def normalized_api_key(self) -> str | None:
        """Return the configured API key if present."""
        if self.qwen_api_key is None:
            return None
        value = self.qwen_api_key.strip()
        return value or None
