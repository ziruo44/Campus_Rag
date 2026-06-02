"""Redis 检索缓存与 workflow 接入测试。"""

from __future__ import annotations

from langchain_core.documents import Document

from agent.workflows import service as service_module
from shared.cache import RetrievalCacheSettings, build_retrieval_cache_key, normalize_query


class FakeRetrievalCache:
    """用于测试的假缓存实现。"""

    def __init__(self, cached_result: dict | None = None) -> None:
        self.cached_result = cached_result
        self.get_calls: list[dict] = []
        self.set_calls: list[dict] = []

    def get_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
    ) -> dict | None:
        self.get_calls.append(
            {
                "domain": domain,
                "user_query": user_query,
                "model_version": model_version,
            }
        )
        return self.cached_result

    def set_workflow_result(
        self,
        *,
        domain: str,
        user_query: str,
        model_version: str,
        result: dict,
    ) -> None:
        self.set_calls.append(
            {
                "domain": domain,
                "user_query": user_query,
                "model_version": model_version,
                "result": result,
            }
        )


def test_normalize_query_keeps_precise_cache_match_stable() -> None:
    assert normalize_query("  transfer process?  ") == "transfer process"
    assert normalize_query("transfer   process?") == "transfer process"


def test_workflow_service_returns_cached_execution_without_runtime_init() -> None:
    cached_result = {
        "retrieval_context": "cached context",
        "evidence_bundle": [{"content": "cached", "source": "cache.md", "metadata": {}}],
        "resolved_queries": [{"route": "detail", "source_query": "AI major intro"}],
        "route_trace": ["detail"],
        "workflow_trace": [],
    }
    cache = FakeRetrievalCache(cached_result=cached_result)
    workflow_service = service_module.KnowledgeWorkflowService(retrieval_cache=cache)

    def fail_init() -> None:
        raise AssertionError("cache hit should skip runtime initialization")

    workflow_service.ensure_initialized = fail_init  # type: ignore[method-assign]

    result = workflow_service.execute(user_query="AI major intro")

    assert result == cached_result
    assert len(cache.get_calls) == 1
    assert cache.set_calls == []


def test_workflow_service_backfills_cache_after_miss(monkeypatch) -> None:
    cache = FakeRetrievalCache()
    workflow_service = service_module.KnowledgeWorkflowService(retrieval_cache=cache)
    workflow_service._runtime._initialized = True
    workflow_service._runtime._retriever = object()
    workflow_service._runtime._parent_documents = []
    workflow_service._router_step = lambda query: "detail"
    workflow_service._query_rewrite_step = lambda query: "rewritten query"
    workflow_service._query_decomposition_step = lambda query: "DECOMPOSABLE: false"

    monkeypatch.setattr(
        "agent.workflows.steps.retrieval.retrieve_route_documents",
        lambda retriever, *, route, query, parent_documents=None: [
            Document(
                page_content="doc content",
                metadata={"major": "AI", "source": "major.md"},
            )
        ],
    )
    monkeypatch.setattr(
        service_module,
        "build_retrieval_context",
        lambda docs, *, route, query, strategy="passthrough": "retrieval context",
    )
    monkeypatch.setattr(service_module, "extract_query_filters", lambda retriever, query: {})

    result = workflow_service.execute(user_query="What is the AI major?")

    assert result["retrieval_context"] == "retrieval context"
    assert len(cache.get_calls) == 1
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0]["result"] == result
    assert cache.set_calls[0]["domain"] == "major_knowledge"


def test_cache_key_uses_per_domain_knowledge_base_versions() -> None:
    settings = RetrievalCacheSettings(
        knowledge_base_version="fallback-v1",
        major_knowledge_base_version="major-v2",
        life_guide_knowledge_base_version="life-v3",
    )

    major_key = build_retrieval_cache_key(
        domain="major_knowledge",
        normalized_query="ai major",
        model_version="model-v1",
        settings=settings,
    )
    life_key = build_retrieval_cache_key(
        domain="life_guide",
        normalized_query="campus wifi",
        model_version="model-v1",
        settings=settings,
    )

    assert ":major-v2:" in major_key
    assert ":life-v3:" in life_key
