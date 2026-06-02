"""Redis 检索缓存配置。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalCacheSettings(BaseSettings):
    """检索缓存的配置项。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RETRIEVAL_CACHE_",
        extra="ignore",
    )

    enabled: bool = False
    redis_url: str | None = None
    ttl_seconds: int = 3600
    key_prefix: str = "rag:retrieval"
    knowledge_base_version: str = "major-knowledge-v1"
    major_knowledge_base_version: str | None = None
    life_guide_knowledge_base_version: str | None = None
    prompt_version: str = "v1"

    @property
    def normalized_redis_url(self) -> str | None:
        """返回去空白后的 Redis 连接地址。"""
        if self.redis_url is None:
            return None
        value = self.redis_url.strip()
        return value or None

    @property
    def is_enabled(self) -> bool:
        """判断是否启用真实缓存。"""
        return self.enabled and self.normalized_redis_url is not None

    def resolve_knowledge_base_version(self, domain: str) -> str:
        """根据工作流域返回对应的知识库版本。"""
        normalized_domain = domain.strip().lower()
        if normalized_domain == "major_knowledge" and self.major_knowledge_base_version:
            return self.major_knowledge_base_version.strip()
        if normalized_domain == "life_guide" and self.life_guide_knowledge_base_version:
            return self.life_guide_knowledge_base_version.strip()
        return self.knowledge_base_version.strip() or "knowledge-base-v1"
