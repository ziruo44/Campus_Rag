"""Tests for metadata-backed list route retrieval."""

from __future__ import annotations

from langchain_core.documents import Document

from agent.workflows.steps.retrieval import retrieve_route_documents
from domain.knowledge.retrieval.hybrid_search import HybridRetriever


class FakeIndexBuilder:
    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        del query, k
        return []


def _build_retriever(*, parent_documents: list[Document], child_documents: list[Document]) -> HybridRetriever:
    return HybridRetriever(
        FakeIndexBuilder(),
        child_documents,
        parent_documents=parent_documents,
    )


def test_retriever_metadata_sets_include_parent_only_entities() -> None:
    parent_documents = [
        Document(
            page_content="college parent",
            metadata={"doc_type": "parent", "doc_level": "college", "college": "继续教育学院", "parent_id": "c1"},
        ),
        Document(
            page_content="major parent",
            metadata={
                "doc_type": "parent",
                "doc_level": "major",
                "college": "继续教育学院",
                "major": "技术转移管理 微专业",
                "parent_id": "m1",
            },
        ),
    ]
    child_documents = [
        Document(
            page_content="child",
            metadata={
                "doc_type": "child",
                "doc_level": "major",
                "college": "信息工程学院",
                "major": "人工智能本科专业",
                "parent_id": "m2",
                "section": "培养目标",
            },
        )
    ]

    retriever = _build_retriever(
        parent_documents=parent_documents,
        child_documents=child_documents,
    )

    assert "继续教育学院" in retriever.colleges
    assert "技术转移管理 微专业" in retriever.majors


def test_list_route_returns_complete_college_catalog_from_parent_metadata() -> None:
    parent_documents = [
        Document(
            page_content="college parent",
            metadata={"doc_type": "parent", "doc_level": "college", "college": "信息工程学院", "parent_id": "c1"},
        ),
        Document(
            page_content="college parent",
            metadata={"doc_type": "parent", "doc_level": "college", "college": "继续教育学院", "parent_id": "c2"},
        ),
    ]
    retriever = _build_retriever(parent_documents=parent_documents, child_documents=[])

    documents = retrieve_route_documents(
        retriever,
        route="list",
        query="有哪些学院",
        parent_documents=parent_documents,
    )

    assert len(documents) == 1
    assert documents[0].metadata["list_type"] == "college_catalog"
    assert "信息工程学院" in documents[0].page_content
    assert "继续教育学院" in documents[0].page_content


def test_list_route_returns_all_majors_for_selected_college_from_parent_metadata() -> None:
    parent_documents = [
        Document(
            page_content="college parent",
            metadata={"doc_type": "parent", "doc_level": "college", "college": "继续教育学院", "parent_id": "c1"},
        ),
        Document(
            page_content="major parent",
            metadata={
                "doc_type": "parent",
                "doc_level": "major",
                "college": "继续教育学院",
                "major": "技术转移管理 微专业",
                "parent_id": "m1",
            },
        ),
        Document(
            page_content="major parent",
            metadata={
                "doc_type": "parent",
                "doc_level": "major",
                "college": "继续教育学院",
                "major": "创业实务 微专业",
                "parent_id": "m2",
            },
        ),
    ]
    retriever = _build_retriever(parent_documents=parent_documents, child_documents=[])

    documents = retrieve_route_documents(
        retriever,
        route="list",
        query="继续教育学院有哪些专业",
        parent_documents=parent_documents,
    )

    assert len(documents) == 1
    assert documents[0].metadata["college"] == "继续教育学院"
    assert "技术转移管理 微专业" in documents[0].page_content
    assert "创业实务 微专业" in documents[0].page_content
