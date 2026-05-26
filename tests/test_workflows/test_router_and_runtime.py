"""Tests for workflow orchestration behavior."""

from __future__ import annotations

from langchain_core.documents import Document

from agent.workflows import service as service_module
from agent.workflows.steps.decomposition import detect_decomposition_rule_based
from agent.workflows.steps.retrieval import build_retrieval_context
from agent.workflows.steps.rewrite import rewrite_query_rule_based, should_rewrite_query
from agent.workflows.steps.routing import _classify_query_rule_based
from domain.knowledge import runtime as runtime_module
from domain.knowledge.retrieval.hybrid_search import HybridRetriever
from domain.knowledge.retrieval.metadata import (
    COLLEGE_FIELDS,
    MAJOR_FIELDS,
    SECTION_FIELDS,
    get_metadata_field,
)


def test_rule_based_router_classifies_common_queries() -> None:
    assert _classify_query_rule_based("有哪些专业") == "list"
    assert _classify_query_rule_based("人工智能专业和计算机科学有什么区别") == "detail"
    assert _classify_query_rule_based("学校介绍") == "general"
    assert _classify_query_rule_based("帮我查一下") is None


def test_decomposition_rule_detects_clear_multi_intent_queries() -> None:
    decision, sub_queries = detect_decomposition_rule_based(
        "信息工程学院有什么专业，人工智能专业有什么特点",
    )
    assert decision is True
    assert sub_queries == [
        "信息工程学院有什么专业",
        "人工智能专业有什么特点",
    ]

    single_decision, single_sub_queries = detect_decomposition_rule_based(
        "计算机科学与技术和人工智能有什么区别",
    )
    assert single_decision is False
    assert single_sub_queries == []


def test_rewrite_rules_only_rewrite_short_generic_queries() -> None:
    assert should_rewrite_query("学校介绍", has_domain_entity=False) is True
    assert should_rewrite_query("帮我查一下", has_domain_entity=False) is True
    assert should_rewrite_query("人工智能专业", has_domain_entity=True) is False
    assert rewrite_query_rule_based("学校介绍") == "温州商学院介绍"
    assert rewrite_query_rule_based("帮我查一下") is None


def test_retrieval_metadata_aliases_support_chinese_fields() -> None:
    doc = Document(
        page_content="content",
        metadata={
            "二级学院": "信息工程学院",
            "专业名称": "人工智能",
            "章节": "培养目标",
        },
    )

    assert get_metadata_field(doc, COLLEGE_FIELDS) == "信息工程学院"
    assert get_metadata_field(doc, MAJOR_FIELDS) == "人工智能"
    assert get_metadata_field(doc, SECTION_FIELDS) == "培养目标"


def test_retrieval_context_compressed_strategy_truncates_content() -> None:
    long_doc = Document(
        page_content="A" * 900,
        metadata={"doc_type": "parent", "section": "培养目标", "major": "人工智能"},
    )

    passthrough = build_retrieval_context(
        [long_doc],
        route="detail",
        query="人工智能专业怎么样",
        strategy="passthrough",
    )
    compressed = build_retrieval_context(
        [long_doc],
        route="detail",
        query="人工智能专业怎么样",
        strategy="compressed",
    )

    assert len(compressed) < len(passthrough)
    assert "AAA" in compressed


def test_workflow_service_builds_shared_retriever_once(monkeypatch) -> None:
    workflow_service = service_module.KnowledgeWorkflowService()
    retriever_calls = {"count": 0}
    parent_doc = Document(page_content="parent", metadata={"parent_id": "p1"})
    child_doc = Document(page_content="child", metadata={"parent_id": "p1", "chunk_index": 0})

    class FakeIndexBuilder:
        def load_or_build_index(self, chunks) -> None:
            assert chunks == [child_doc]

    class FakeRetriever:
        def __init__(self, index_builder, chunks, *, parent_documents=None) -> None:
            retriever_calls["count"] += 1
            assert isinstance(index_builder, FakeIndexBuilder)
            assert chunks == [child_doc]
            assert parent_documents == [parent_doc]

    monkeypatch.setattr(runtime_module, "load_documents", lambda _: ["doc"])
    monkeypatch.setattr(runtime_module, "chunk_documents", lambda docs: ([parent_doc], [child_doc]))
    monkeypatch.setattr(runtime_module, "IndexBuilder", FakeIndexBuilder)
    monkeypatch.setattr(runtime_module, "HybridRetriever", FakeRetriever)

    workflow_service.ensure_initialized()
    workflow_service.ensure_initialized()

    assert retriever_calls["count"] == 1
    assert workflow_service._runtime.retriever is not None


def test_workflow_service_execute_returns_structured_trace(monkeypatch) -> None:
    workflow_service = service_module.KnowledgeWorkflowService()
    workflow_service._runtime._initialized = True
    workflow_service._runtime._retriever = object()
    workflow_service._router_step = lambda query: "detail"
    workflow_service._query_rewrite_step = lambda query: "rewritten query"
    workflow_service._query_decomposition_step = lambda query: "DECOMPOSABLE: false"

    monkeypatch.setattr(
        "agent.workflows.steps.retrieval.retrieve_route_documents",
        lambda retriever, *, route, query: [
            Document(page_content="doc content", metadata={"major": "AI", "source": "major.md"})
        ],
    )
    monkeypatch.setattr(
        service_module,
        "build_retrieval_context",
        lambda docs, *, route, query, strategy="passthrough": "retrieval context",
    )
    monkeypatch.setattr(service_module, "extract_query_filters", lambda retriever, query: {})

    execution = workflow_service.execute(
        user_query="What is the AI major?",
        retrieval_context_strategy="compressed",
    )

    assert execution["resolved_queries"][0]["route"] == "detail"
    assert execution["retrieval_context"] == "retrieval context"
    assert execution["evidence_bundle"][0]["source"] == "major.md"
    assert [event["tool_name"] for event in execution["workflow_trace"]] == [
        "query_decomposition_tool",
        "router_tool",
        "detail_retrieval_tool",
    ]
    assert "raw_result" not in execution


def test_hybrid_retriever_groups_child_results_by_parent() -> None:
    class FakeIndexBuilder:
        def similarity_search(self, query: str, k: int = 5):
            del query, k
            return []

    parent_doc = Document(
        page_content="### 人工智能本科专业\n#### 培养目标\n...",
        metadata={
            "doc_type": "parent",
            "doc_level": "major",
            "college": "信息工程学院",
            "major": "人工智能本科专业",
            "parent_id": "ai-parent",
        },
    )
    child_1 = Document(
        page_content="#### 专业特色\n强调 AI+商科。",
        metadata={
            "doc_type": "child",
            "section": "专业特色",
            "college": "信息工程学院",
            "major": "人工智能本科专业",
            "parent_id": "ai-parent",
            "rrf_score": 0.02,
        },
    )
    child_2 = Document(
        page_content="#### 就业前景\n可进入互联网和金融科技公司。",
        metadata={
            "doc_type": "child",
            "section": "就业前景",
            "college": "信息工程学院",
            "major": "人工智能本科专业",
            "parent_id": "ai-parent",
            "rrf_score": 0.015,
        },
    )

    retriever = HybridRetriever(
        FakeIndexBuilder(),
        [child_1, child_2],
        parent_documents=[parent_doc],
    )
    grouped = retriever.group_child_results([child_1, child_2], top_k_groups=3)

    assert len(grouped) == 1
    assert grouped[0].metadata["doc_type"] == "grouped_parent"
    assert grouped[0].metadata["major"] == "人工智能本科专业"
    assert grouped[0].metadata["matched_child_count"] == 2
    assert "专业特色" in grouped[0].page_content
    assert "就业前景" in grouped[0].page_content
