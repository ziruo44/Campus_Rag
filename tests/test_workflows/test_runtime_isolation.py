"""Tests for runtime and workflow isolation between major and life guide knowledge."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

import app_bootstrap
from domain.life_guide_knowledge import runtime as life_runtime_module
from domain.major_knowledge import runtime as major_runtime_module


def test_major_runtime_uses_major_knowledge_assets(monkeypatch) -> None:
    captured: dict[str, object] = {}
    parent_doc = Document(page_content="parent", metadata={"parent_id": "p1"})
    child_doc = Document(page_content="child", metadata={"parent_id": "p1", "chunk_index": 0})
    major_raw_dir = Path("D:/ziruo_project/rag_project/data/raw/majors")

    class FakeIndexBuilder:
        def load_or_build_index(self, chunks) -> None:
            captured["index_chunks"] = list(chunks)

    class FakeRetriever:
        def __init__(self, index_builder, chunks, *, parent_documents=None) -> None:
            captured["retriever_builder_type"] = type(index_builder).__name__
            captured["retriever_chunks"] = list(chunks)
            captured["retriever_parent_documents"] = list(parent_documents or [])

    def fake_load_documents(path):
        captured["load_documents_path"] = path
        return ["major-doc"]

    monkeypatch.setattr(major_runtime_module, "get_raw_data_dir", lambda: major_raw_dir)
    monkeypatch.setattr(major_runtime_module, "load_documents", fake_load_documents)
    monkeypatch.setattr(
        major_runtime_module,
        "chunk_documents",
        lambda docs: ([parent_doc], [child_doc]),
    )
    monkeypatch.setattr(major_runtime_module, "IndexBuilder", FakeIndexBuilder)
    monkeypatch.setattr(major_runtime_module, "HybridRetriever", FakeRetriever)

    runtime = major_runtime_module.KnowledgeRuntime()
    runtime.ensure_initialized()

    assert captured["load_documents_path"] == major_raw_dir
    assert captured["index_chunks"] == [child_doc]
    assert captured["retriever_builder_type"] == "FakeIndexBuilder"
    assert captured["retriever_chunks"] == [child_doc]
    assert captured["retriever_parent_documents"] == [parent_doc]


def test_life_guide_runtime_uses_life_guide_assets(monkeypatch) -> None:
    captured: dict[str, object] = {}
    source_doc = Document(page_content="source", metadata={"source": "生活指南.md"})
    chunk_doc = Document(
        page_content="校园网办理说明",
        metadata={
            "category": "学习支持类",
            "service_name": "校园网络",
            "sub_service_name": "校园网开通",
        },
    )
    life_raw_dir = Path("D:/ziruo_project/rag_project/data/raw/life_guide")

    class FakeIndexBuilder:
        def load_or_build_index(self, chunks) -> None:
            captured["index_chunks"] = list(chunks)

    class FakeRetriever:
        def __init__(self, index_builder, chunks) -> None:
            captured["retriever_builder_type"] = type(index_builder).__name__
            captured["retriever_chunks"] = list(chunks)

    def fake_load_life_guide(path):
        captured["load_life_guide_path"] = path
        return [source_doc]

    monkeypatch.setattr(
        life_runtime_module,
        "get_life_guide_raw_data_dir",
        lambda: life_raw_dir,
    )
    monkeypatch.setattr(life_runtime_module, "load_life_guide", fake_load_life_guide)
    monkeypatch.setattr(
        life_runtime_module,
        "chunk_life_guide_all",
        lambda docs: [chunk_doc],
    )
    monkeypatch.setattr(life_runtime_module, "LifeGuideIndexBuilder", FakeIndexBuilder)
    monkeypatch.setattr(
        life_runtime_module,
        "LifeGuideHybridRetriever",
        FakeRetriever,
    )

    runtime = life_runtime_module.LifeGuideKnowledgeRuntime()
    runtime.ensure_initialized()

    assert captured["load_life_guide_path"] == life_raw_dir
    assert runtime.documents == [source_doc]
    assert runtime.chunks == [chunk_doc]
    assert captured["index_chunks"] == [chunk_doc]
    assert captured["retriever_builder_type"] == "FakeIndexBuilder"
    assert captured["retriever_chunks"] == [chunk_doc]


def test_bootstrap_workflow_services_bind_distinct_runtimes() -> None:
    app_bootstrap.get_major_knowledge_runtime.cache_clear()
    app_bootstrap.get_life_guide_runtime.cache_clear()
    app_bootstrap.get_major_workflow_service.cache_clear()
    app_bootstrap.get_life_guide_workflow_service.cache_clear()

    major_runtime = app_bootstrap.get_major_knowledge_runtime()
    life_runtime = app_bootstrap.get_life_guide_runtime()
    major_service = app_bootstrap.get_major_workflow_service()
    life_service = app_bootstrap.get_life_guide_workflow_service()

    assert major_runtime is not life_runtime
    assert major_service._runtime is major_runtime
    assert life_service._runtime is life_runtime
    assert type(major_runtime).__name__ == "KnowledgeRuntime"
    assert type(life_runtime).__name__ == "LifeGuideKnowledgeRuntime"
