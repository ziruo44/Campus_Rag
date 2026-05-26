"""Tests for broader routing and decomposition behavior."""

from __future__ import annotations

from langchain_core.documents import Document

from agent.workflows.steps.decomposition import detect_decomposition_rule_based
from agent.workflows.steps.retrieval import retrieve_route_documents
from agent.workflows.steps.routing import _classify_query_rule_based
from domain.knowledge.retrieval.hybrid_search import HybridRetriever


class FakeIndexBuilder:
    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        del query, k
        return []


def test_collection_intro_queries_route_to_list() -> None:
    assert _classify_query_rule_based("温州商学院各学院详细介绍") == "general"
    assert _classify_query_rule_based("各学院分别介绍一下") == "general"


def test_decomposition_rewrites_contextless_follow_up_into_standalone_query() -> None:
    decision, sub_queries = detect_decomposition_rule_based("学校有哪些学院，分别具体介绍一下")

    assert decision is True
    assert sub_queries == [
        "学校有哪些学院",
        "学校各学院具体介绍一下",
    ]


def test_list_route_still_returns_name_catalog_for_college_queries() -> None:
    parent_documents = [
        Document(
            page_content="college parent",
            metadata={"doc_type": "parent", "doc_level": "college", "college": "信息工程学院", "parent_id": "c1"},
        ),
        Document(
            page_content="college parent",
            metadata={"doc_type": "parent", "doc_level": "college", "college": "管理学院", "parent_id": "c2"},
        ),
    ]
    retriever = HybridRetriever(FakeIndexBuilder(), [], parent_documents=parent_documents)

    documents = retrieve_route_documents(
        retriever,
        route="list",
        query="学校有哪些学院",
        parent_documents=parent_documents,
    )

    assert len(documents) == 1
    assert documents[0].metadata["list_type"] == "college_catalog"
    assert "信息工程学院" in documents[0].page_content
    assert "管理学院" in documents[0].page_content
