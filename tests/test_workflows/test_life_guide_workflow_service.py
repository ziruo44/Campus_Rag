"""Tests for the life guide workflow service."""

from __future__ import annotations

from langchain_core.documents import Document

from agent.workflows.life_guide_service import LifeGuideWorkflowService


class StubLifeGuideRetriever:
    def hybrid_search(self, query: str, top_k: int = 5) -> list[Document]:
        del top_k
        return [
            Document(
                page_content=f"Campus network instructions: {query}",
                metadata={
                    "category": "student-services",
                    "service_name": "campus-network",
                    "sub_service_name": "network-activation",
                    "source": "life-guide.md",
                    "doc_type": "service_subitem",
                },
            )
        ]


class StubLifeGuideRuntime:
    def __init__(self) -> None:
        self.is_initialized = False
        self.ensure_initialized_calls = 0
        self.retriever = StubLifeGuideRetriever()

    def ensure_initialized(self) -> None:
        self.ensure_initialized_calls += 1
        self.is_initialized = True


class FakeRetrievalCache:
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


def test_life_guide_workflow_service_returns_structured_trace() -> None:
    runtime = StubLifeGuideRuntime()
    workflow_service = LifeGuideWorkflowService(knowledge_runtime=runtime)

    execution = workflow_service.execute(user_query="How do I activate campus wifi?")

    assert runtime.ensure_initialized_calls == 1
    assert execution["resolved_queries"][0]["route"] == "life_guide"
    assert execution["route_trace"] == ["life_guide"]
    assert execution["evidence_bundle"][0]["source"] == "life-guide.md"
    assert execution["workflow_trace"][0]["tool_name"] == "life_guide_retrieval_tool"
    assert "campus-network" in execution["retrieval_context"]


def test_life_guide_workflow_service_reads_from_cache_before_runtime() -> None:
    cached_result = {
        "retrieval_context": "cached life guide context",
        "evidence_bundle": [],
        "resolved_queries": [{"route": "life_guide", "source_query": "wifi"}],
        "route_trace": ["life_guide"],
        "workflow_trace": [],
    }
    runtime = StubLifeGuideRuntime()
    cache = FakeRetrievalCache(cached_result=cached_result)
    workflow_service = LifeGuideWorkflowService(
        knowledge_runtime=runtime,
        retrieval_cache=cache,
    )

    execution = workflow_service.execute(user_query="How do I activate campus wifi?")

    assert execution == cached_result
    assert runtime.ensure_initialized_calls == 0
    assert len(cache.get_calls) == 1
    assert cache.get_calls[0]["domain"] == "life_guide"
    assert cache.set_calls == []


def test_life_guide_workflow_service_backfills_cache_after_miss() -> None:
    runtime = StubLifeGuideRuntime()
    cache = FakeRetrievalCache()
    workflow_service = LifeGuideWorkflowService(
        knowledge_runtime=runtime,
        retrieval_cache=cache,
    )

    execution = workflow_service.execute(user_query="How do I activate campus wifi?")

    assert runtime.ensure_initialized_calls == 1
    assert execution["route_trace"] == ["life_guide"]
    assert len(cache.get_calls) == 1
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0]["domain"] == "life_guide"
    assert cache.set_calls[0]["result"] == execution
