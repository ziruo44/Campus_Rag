"""Tests for routing heuristics and workflow runtime behavior."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_agent.agent_modules.tools.query_decomposition import (
    detect_decomposition_rule_based,
)
from rag_agent.agent_modules.tools.query_rewrite import (
    rewrite_query_rule_based,
    should_rewrite_query,
)
from rag_agent.agent_modules.tools.retrieval import build_retrieval_context
from rag_agent.agent_modules.tools.router import _classify_query_rule_based
from rag_agent.api.services import agent_runtime as runtime_module


def test_rule_based_router_classifies_common_queries() -> None:
    assert _classify_query_rule_based("温州商学院有哪些专业") == "list"
    assert _classify_query_rule_based("计算机科学与技术和人工智能有什么区别") == "detail"
    assert _classify_query_rule_based("温州商学院介绍") == "general"
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
    assert should_rewrite_query("学校介绍", has_domain_entity=False) is False
    assert should_rewrite_query("帮我查一下", has_domain_entity=False) is True
    assert should_rewrite_query("人工智能专业", has_domain_entity=True) is False
    assert rewrite_query_rule_based("学校介绍") == "温州商学院介绍"
    assert rewrite_query_rule_based("帮我查一下") is None


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


def test_agent_runtime_builds_shared_retriever_once(monkeypatch) -> None:
    runtime = runtime_module.AgentRuntime()
    retriever_calls = {"count": 0}

    class FakeIndexBuilder:
        def load_or_build_index(self, chunks) -> None:
            assert chunks == ["parent", "child"]

    class FakeRetriever:
        def __init__(self, index_builder, chunks) -> None:
            retriever_calls["count"] += 1
            assert isinstance(index_builder, FakeIndexBuilder)
            assert chunks == ["parent", "child"]

    monkeypatch.setattr(runtime_module, "load_documents", lambda _: ["doc"])
    monkeypatch.setattr(runtime_module, "chunk_documents", lambda docs: (["parent"], ["child"]))
    monkeypatch.setattr(runtime_module, "IndexBuilder", FakeIndexBuilder)
    monkeypatch.setattr(runtime_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(runtime_module, "create_router_tool", lambda llm: SimpleNamespace(invoke=lambda query: "detail"))
    monkeypatch.setattr(runtime_module, "create_query_rewrite_tool", lambda llm: SimpleNamespace(invoke=lambda query: query))
    monkeypatch.setattr(
        runtime_module,
        "create_query_decomposition_tool",
        lambda llm: SimpleNamespace(invoke=lambda query: "DECOMPOSABLE: false"),
    )

    runtime.ensure_initialized()
    runtime.ensure_initialized()

    assert retriever_calls["count"] == 1
    assert runtime._retriever is not None


def test_agent_runtime_workflow_detail_query_uses_single_final_generation(monkeypatch) -> None:
    runtime = runtime_module.AgentRuntime()
    runtime._initialized = True
    runtime._retriever = object()

    class FakeLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            assert isinstance(messages[-1], HumanMessage)
            return AIMessage(content="final answer")

    runtime._llm = FakeLLM()
    runtime._router_tool = SimpleNamespace(invoke=lambda query: "detail")
    runtime._query_rewrite_tool = SimpleNamespace(invoke=lambda query: "rewritten query")
    runtime._query_decomposition_tool = SimpleNamespace(
        invoke=lambda query: "DECOMPOSABLE: false",
    )

    monkeypatch.setattr(
        runtime_module,
        "build_agent_messages",
        lambda thread: [SystemMessage(content="memory"), HumanMessage(content="history question")],
    )
    monkeypatch.setattr(
        runtime_module,
        "retrieve_route_documents",
        lambda retriever, *, route, query: [Document(page_content="doc content", metadata={"major": "人工智能"})],
    )
    monkeypatch.setattr(
        runtime_module,
        "build_retrieval_context",
        lambda docs, *, route, query, strategy="passthrough": "retrieval context",
    )
    monkeypatch.setattr(runtime_module, "extract_query_filters", lambda retriever, query: {})

    thread = SimpleNamespace(
        turns=[SimpleNamespace(user_message=SimpleNamespace(content="计算机科学与技术和人工智能有什么区别"))],
    )

    result = runtime.invoke_with_strategy(thread, retrieval_context_strategy="compressed")

    assert result.answer == "final answer"
    assert runtime._llm.calls == 1
    assert result.raw_result["resolved_queries"][0]["route"] == "detail"
    assert any(
        call["name"] == "router_tool"
        for message in result.messages
        if getattr(message, "tool_calls", None)
        for call in message.tool_calls
    )
    assert any(
        call["name"] == "detail_retrieval_tool"
        for message in result.messages
        if getattr(message, "tool_calls", None)
        for call in message.tool_calls
    )
