"""Retrieval helpers for explicit routed retrieval."""

from __future__ import annotations

import re

from langchain_core.documents import Document

from agent.workflows.models import RetrievalStepResult, WorkflowTraceEvent
from domain.knowledge.retrieval.academy_map import (
    build_academy_majors_map,
    get_all_colleges,
)
from domain.knowledge.retrieval.filters import extract_query_filters
from domain.knowledge.retrieval.hybrid_search import HybridRetriever
from utils.text import truncate_text

DEFAULT_ROUTE_TOP_K = {
    "list": 10,
    "detail": 5,
    "general": 5,
}

PRIORITY_SECTIONS = (
    "培养目标",
    "主干课程",
    "专业特色",
    "就业前景",
    "毕业去向",
    "专业介绍",
)

CONTENT_LIMITS = {
    "list": 150,
    "list_intro": 350,
    "detail_parent": 600,
    "detail_child": 300,
    "general": 250,
}


def _retrieve_docs(
    retriever: HybridRetriever,
    query: str,
    top_k: int,
    use_filter: bool = False,
) -> list[Document]:
    """Run hybrid search, optionally with keyword-based metadata filter."""
    if use_filter:
        filters = extract_query_filters(retriever, query)
        if filters:
            child_docs = retriever.metadata_search(query=query, top_k=top_k * 3, **filters)
            return retriever.group_child_results(child_docs, top_k_groups=top_k)
    child_docs = retriever.hybrid_search(query, top_k=top_k * 3)
    return retriever.group_child_results(child_docs, top_k_groups=top_k)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", "", query)


def _should_use_metadata_catalog(query: str, filters: dict[str, str]) -> bool:
    normalized = _normalize_query(query)
    if filters:
        return True
    return any(keyword in normalized for keyword in ("学院", "专业", "微专业"))


def _is_college_listing_query(query: str) -> bool:
    normalized = _normalize_query(query)
    return "学院" in normalized and "专业" not in normalized and "微专业" not in normalized


def _build_catalog_document(
    *,
    title: str,
    lines: list[str],
    metadata: dict[str, object],
) -> Document:
    return Document(
        page_content="\n".join([f"### {title}", *lines]).strip(),
        metadata={
            "doc_type": "list_catalog",
            "doc_level": "catalog",
            "source": "metadata_catalog",
            **metadata,
        },
    )
def _build_list_catalog_documents(
    query: str,
    retriever: HybridRetriever,
    parent_documents: list[Document],
) -> list[Document]:
    filters = extract_query_filters(retriever, query)
    if not _should_use_metadata_catalog(query, filters):
        return []

    colleges = get_all_colleges(parent_documents)
    academy_majors = build_academy_majors_map(parent_documents)

    if _is_college_listing_query(query):
        return [
            _build_catalog_document(
                title="学院列表",
                lines=[
                    f"共 {len(colleges)} 个学院。",
                    *[f"{index}. {college}" for index, college in enumerate(colleges, start=1)],
                ],
                metadata={
                    "list_type": "college_catalog",
                    "result_count": len(colleges),
                },
            )
        ]

    college = filters.get("college")
    if college:
        majors = academy_majors.get(college, [])
        lines = [f"学院：{college}", f"共 {len(majors)} 个专业。"]
        if majors:
            lines.extend(f"{index}. {major}" for index, major in enumerate(majors, start=1))
        else:
            lines.append("未找到该学院下的专业条目。")
        return [
            _build_catalog_document(
                title=f"{college}专业列表",
                lines=lines,
                metadata={
                    "list_type": "major_catalog",
                    "college": college,
                    "result_count": len(majors),
                },
            )
        ]

    documents: list[Document] = []
    for college_name in colleges:
        majors = academy_majors.get(college_name, [])
        lines = [f"学院：{college_name}", f"共 {len(majors)} 个专业。"]
        if majors:
            lines.extend(
                f"{index}. {major}" for index, major in enumerate(majors, start=1)
            )
        else:
            lines.append("未找到该学院下的专业条目。")
        documents.append(
            _build_catalog_document(
                title=f"{college_name}专业列表",
                lines=lines,
                metadata={
                    "list_type": "major_catalog",
                    "college": college_name,
                    "result_count": len(majors),
                },
            )
        )
    return documents


def retrieve_route_documents(
    retriever: HybridRetriever,
    *,
    route: str,
    query: str,
    parent_documents: list[Document] | None = None,
) -> list[Document]:
    """Retrieve documents for one routed query without tool wrappers."""
    if route == "list":
        parent_documents = parent_documents or []
        catalog_documents = _build_list_catalog_documents(query, retriever, parent_documents)
        if catalog_documents:
            return catalog_documents
        return _retrieve_docs(
            retriever,
            query=query,
            top_k=DEFAULT_ROUTE_TOP_K["list"],
            use_filter=True,
        )
    if route == "detail":
        return _retrieve_docs(
            retriever,
            query=query,
            top_k=DEFAULT_ROUTE_TOP_K["detail"],
        )
    if route == "general":
        return _retrieve_docs(
            retriever,
            query=query,
            top_k=DEFAULT_ROUTE_TOP_K["general"],
        )
    raise ValueError(f"Unsupported retrieval route: {route}")


def run_retrieval_step(
    retriever: HybridRetriever,
    *,
    route: str,
    query: str,
    parent_documents: list[Document] | None = None,
) -> RetrievalStepResult:
    """Retrieve documents and emit a trace-friendly summary result."""
    documents = retrieve_route_documents(
        retriever,
        route=route,
        query=query,
        parent_documents=parent_documents,
    )
    tool_args = {"query": query}
    if route == "general":
        tool_args = {"rewritten_query": query}

    return RetrievalStepResult(
        route=route,
        query=query,
        documents=documents,
        trace_event=WorkflowTraceEvent(
            step="retrieval",
            source="retrieval",
            tool_name=f"{route}_retrieval_tool",
            tool_args=tool_args,
            tool_output=_build_retrieval_trace_summary(documents, route=route, query=query),
        ),
    )


def format_docs(
    docs: list[Document],
    *,
    route: str,
    query: str,
) -> str:
    """Format retrieved documents as a tool message for the agent."""
    if not docs:
        return (
            f"route={route}\n"
            f"query={query}\n"
            "result_count=0\n"
            "documents:\n"
            "No relevant documents were found."
        )

    parts = [
        f"route={route}",
        f"query={query}",
        f"result_count={len(docs)}",
        "documents:",
    ]

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.extend(
            [
                f"[document {i}]",
                f"college: {meta.get('college', '')}",
                f"major: {meta.get('major', '')}",
                f"section: {meta.get('section', '')}",
                f"doc_level: {meta.get('doc_level', '')}",
                f"doc_type: {meta.get('doc_type', '')}",
                f"matched_sections: {', '.join(meta.get('matched_sections', []))}",
                f"source: {meta.get('source', '')}",
                "content:",
                doc.page_content,
                "---",
            ]
        )

    return "\n".join(parts)


def build_retrieval_context(
    docs: list[Document],
    *,
    route: str,
    query: str,
    strategy: str = "passthrough",
) -> str:
    """Build final-generation retrieval context."""
    if strategy == "compressed":
        docs = _compress_retrieval_docs(docs, route=route)
    return format_docs(docs, route=route, query=query)


def _compress_retrieval_docs(
    docs: list[Document],
    *,
    route: str,
) -> list[Document]:
    if route == "list":
        selected = docs[:10]
    elif route == "detail":
        selected = _select_detail_docs(docs)
    else:
        selected = docs[:5]

    return [_truncate_document(doc, route=route) for doc in selected]


def _select_detail_docs(docs: list[Document]) -> list[Document]:
    grouped_docs = [doc for doc in docs if doc.metadata.get("doc_type") == "grouped_parent"]
    if grouped_docs:
        return grouped_docs[:3]

    parent_docs = [doc for doc in docs if doc.metadata.get("doc_type") == "parent"]
    child_docs = [doc for doc in docs if doc.metadata.get("doc_type") != "parent"]

    ranked_parents = sorted(parent_docs, key=_detail_doc_priority)[:2]
    ranked_children = sorted(child_docs, key=_detail_doc_priority)[:4]
    return ranked_parents + ranked_children


def _detail_doc_priority(doc: Document) -> tuple[int, str]:
    section = str(doc.metadata.get("section", "") or "")
    try:
        section_rank = PRIORITY_SECTIONS.index(section)
    except ValueError:
        section_rank = len(PRIORITY_SECTIONS)
    return section_rank, section


def _truncate_document(doc: Document, *, route: str) -> Document:
    metadata = dict(doc.metadata)
    content = _normalize_doc_content(doc.page_content)

    if route == "list":
        limit = (
            CONTENT_LIMITS["list_intro"]
            if metadata.get("doc_level") == "college" and metadata.get("doc_type") == "parent"
            else CONTENT_LIMITS["list"]
        )
    elif route == "detail":
        if metadata.get("doc_type") == "grouped_parent":
            limit = CONTENT_LIMITS["detail_parent"]
        else:
            limit = (
                CONTENT_LIMITS["detail_parent"]
                if metadata.get("doc_type") == "parent"
                else CONTENT_LIMITS["detail_child"]
            )
    else:
        limit = CONTENT_LIMITS["general"]

    return Document(
        page_content=truncate_text(content, limit=limit),
        metadata=metadata,
    )


def _normalize_doc_content(content: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", content.strip())


def _build_retrieval_trace_summary(
    documents: list[Document],
    *,
    route: str,
    query: str,
) -> str:
    if not documents:
        return f"route={route}\nquery={query}\nresult_count=0"

    preview_lines = [
        f"route={route}",
        f"query={query}",
        f"result_count={len(documents)}",
    ]
    for index, doc in enumerate(documents[:3], start=1):
        metadata = doc.metadata
        preview_lines.append(
            f"[document {index}] source={metadata.get('source', metadata.get('filename', ''))}"
        )
        preview_lines.append(
            f"content_preview={truncate_text(_normalize_doc_content(doc.page_content), limit=120)}"
        )
    return "\n".join(preview_lines)
