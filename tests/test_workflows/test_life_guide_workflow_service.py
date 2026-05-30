"""Tests for the life guide workflow service."""

from __future__ import annotations

from langchain_core.documents import Document

from agent.workflows.life_guide_service import LifeGuideWorkflowService


class StubLifeGuideRetriever:
    def hybrid_search(self, query: str, top_k: int = 5) -> list[Document]:
        del top_k
        return [
            Document(
                page_content=f"校园网办理说明：{query}",
                metadata={
                    "category": "学习支持类",
                    "service_name": "校园网络",
                    "sub_service_name": "校园网开通",
                    "source": "生活指南.md",
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


def test_life_guide_workflow_service_returns_structured_trace() -> None:
    runtime = StubLifeGuideRuntime()
    workflow_service = LifeGuideWorkflowService(knowledge_runtime=runtime)

    execution = workflow_service.execute(
        user_query="校园网怎么开通",
        retrieval_context_strategy="compressed",
    )

    assert runtime.ensure_initialized_calls == 1
    assert execution["resolved_queries"][0]["route"] == "life_guide"
    assert execution["route_trace"] == ["life_guide"]
    assert execution["evidence_bundle"][0]["source"] == "生活指南.md"
    assert execution["workflow_trace"][0]["tool_name"] == "life_guide_retrieval_tool"
    assert "校园网络" in execution["retrieval_context"]
