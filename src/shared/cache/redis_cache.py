"""Redis 检索缓存封装。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol
import json
import logging
import re

from shared.cache.config import RetrievalCacheSettings
from shared.cache.query_normalizer import normalize_query
from shared.observability.performance import measure_stage
from utils.text import truncate_text

logger = logging.getLogger(__name__)


class RetrievalCache(Protocol):
    """检索缓存协议，用于 workflow service 依赖注入。"""

    def get_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
    ) -> dict[str, Any] | None: ...

    def set_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
        result: dict[str, Any],
    ) -> None: ...


class NullRetrievalCache:
    """禁用缓存或缓存不可用时的空实现。"""

    def get_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
    ) -> dict[str, Any] | None:
        del domain, user_query, model_version
        return None

    def set_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
        result: dict[str, Any],
    ) -> None:
        del domain, user_query, model_version, result


class RedisRetrievalCache:
    """基于 Redis 的精确检索缓存。"""

    def __init__(
        self,
        *,
        settings: RetrievalCacheSettings | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self.settings = settings or RetrievalCacheSettings()
        self._redis_client = redis_client if redis_client is not None else self._build_client()

    def get_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
    ) -> dict[str, Any] | None:
        if self._redis_client is None:
            return None

        normalized_query = normalize_query(user_query)
        key = build_retrieval_cache_key(
            domain=domain,
            normalized_query=normalized_query,
            model_version=model_version,
            settings=self.settings,
        )

        try:
            with measure_stage("cache.retrieval.get"):
                payload = self._redis_client.get(key)
        except Exception as exc:
            logger.warning("Redis 检索缓存读取失败：%s", exc)
            return None

        if payload is None:
            logger.info(
                "retrieval_cache_miss %s",
                json.dumps(
                    _build_log_payload(
                        key=key,
                        domain=domain,
                        normalized_query=normalized_query,
                    ),
                    ensure_ascii=False,
                ),
            )
            return None

        try:
            decoded = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
            parsed = json.loads(decoded)
        except Exception as exc:
            logger.warning("Redis 检索缓存解析失败：%s", exc)
            return None

        result = parsed.get("result")
        if not isinstance(result, dict):
            return None

        logger.info(
            "retrieval_cache_hit %s",
            json.dumps(
                _build_log_payload(
                    key=key,
                    domain=domain,
                    normalized_query=normalized_query,
                ),
                ensure_ascii=False,
            ),
        )
        return result

    def set_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
        result: dict[str, Any],
    ) -> None:
        if self._redis_client is None:
            return

        normalized_query = normalize_query(user_query)
        key = build_retrieval_cache_key(
            domain=domain,
            normalized_query=normalized_query,
            model_version=model_version,
            settings=self.settings,
        )
        payload = {
            "user_query": user_query,
            "normalized_query": normalized_query,
            "domain": domain,
            "kb_version": self.settings.resolve_knowledge_base_version(domain),
            "prompt_version": self.settings.prompt_version,
            "model_version": model_version,
            "result": result,
            "created_at": datetime.now(UTC).isoformat(),
            "expires_at": (
                datetime.now(UTC) + timedelta(seconds=self.settings.ttl_seconds)
            ).isoformat(),
        }

        try:
            with measure_stage("cache.retrieval.set"):
                self._redis_client.set(
                    key,
                    json.dumps(payload, ensure_ascii=False),
                    ex=self.settings.ttl_seconds,
                )
        except Exception as exc:
            logger.warning("Redis 检索缓存写入失败：%s", exc)
            return

        logger.info(
            "retrieval_cache_write %s",
            json.dumps(
                _build_log_payload(
                    key=key,
                    domain=domain,
                    normalized_query=normalized_query,
                ),
                ensure_ascii=False,
            ),
        )

    def _build_client(self) -> Any | None:
        if not self.settings.is_enabled:
            return None

        try:
            from redis import Redis
        except ImportError:
            logger.warning("未安装 redis 依赖，检索缓存将自动禁用。")
            return None

        return Redis.from_url(
            self.settings.normalized_redis_url,
            protocol=2,
            decode_responses=False,
        )


def build_retrieval_cache_key(
    *,
    domain: str,
    normalized_query: str,
    model_version: str,
    settings: RetrievalCacheSettings,
) -> str:
    """构建包含版本信息的检索缓存 key。"""
    query_hash = sha256(normalized_query.encode("utf-8")).hexdigest()
    return ":".join(
        [
            settings.key_prefix,
            _normalize_key_segment(domain),
            _normalize_key_segment(settings.resolve_knowledge_base_version(domain)),
            _normalize_key_segment(settings.prompt_version),
            _normalize_key_segment(model_version),
            query_hash,
        ]
    )


def _normalize_key_segment(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    return normalized or "unknown"


def _build_log_payload(
    *,
    key: str,
    domain: str,
    normalized_query: str,
) -> dict[str, str]:
    return {
        "domain": domain,
        "key": key,
        "normalized_query": truncate_text(normalized_query, limit=80),
    }
