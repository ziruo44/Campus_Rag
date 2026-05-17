"""Retrieval tools for explicit routed retrieval."""

import logging
import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from rag_agent.indexing.index_builder import IndexBuilder
from rag_agent.observability.performance import measure_stage, record_retrieval_results
from rag_agent.retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_ROUTE_TOP_K = {
    "list": 10,
    "detail": 5,
    "general": 5,
}


def _extract_filters_from_query(retriever: HybridRetriever, query: str) -> dict:
    """Extract college/major from query using keyword matching (no LLM)."""
    filters = {}
    colleges = set()
    majors = set()
    for chunk in retriever.chunks:
        col = chunk.metadata.get("college")
        if col:
            colleges.add(col)
        mjr = chunk.metadata.get("major")
        if mjr:
            majors.add(mjr)

    # Longest match first (more specific)
    for col in sorted(colleges, key=len, reverse=True):
        if col and col in query:
            filters["college"] = col
            break
    for mjr in sorted(majors, key=len, reverse=True):
        if mjr and mjr in query:
            filters["major"] = mjr
            break
    return filters


def extract_query_filters(retriever: HybridRetriever, query: str) -> dict:
    """Public wrapper for keyword-based query filters."""
    return _extract_filters_from_query(retriever, query)


def _retrieve_docs(
    retriever: HybridRetriever,
    query: str,
    top_k: int,
    use_filter: bool = False,
) -> list[Document]:
    """Run hybrid search, optionally with keyword-based metadata filter."""
    if use_filter:
        filters = _extract_filters_from_query(retriever, query)
        if filters:
            return retriever.metadata_search(query=query, top_k=top_k, **filters)
    return retriever.hybrid_search(query, top_k=top_k)


def retrieve_route_documents(
    retriever: HybridRetriever,
    *,
    route: str,
    query: str,
) -> list[Document]:
    """Retrieve documents for one routed query without tool wrappers."""
    if route == "list":
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
    """Build final-generation retrieval context with a pluggable strategy."""
    if strategy == "passthrough":
        return format_docs(docs, route=route, query=query)
    if strategy == "compressed":
        compressed_docs = _compress_retrieval_docs(docs, route=route)
        return format_docs(compressed_docs, route=route, query=query)
    raise ValueError(f"Unsupported retrieval context strategy: {strategy}")


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
    "detail_parent": 600,
    "detail_child": 300,
    "general": 250,
}


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
        limit = CONTENT_LIMITS["list"]
    elif route == "detail":
        limit = (
            CONTENT_LIMITS["detail_parent"]
            if metadata.get("doc_type") == "parent"
            else CONTENT_LIMITS["detail_child"]
        )
    else:
        limit = CONTENT_LIMITS["general"]

    return Document(
        page_content=_truncate_text(content, limit=limit),
        metadata=metadata,
    )


def _normalize_doc_content(content: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", content.strip())


def _truncate_text(content: str, *, limit: int) -> str:
    if len(content) <= limit:
        return content
    if limit <= 3:
        return content[:limit]
    return f"{content[: limit - 3].rstrip()}..."


def create_retrieval_tool(
    llm: BaseChatModel,
    index_builder: IndexBuilder,
    chunks: list[Document],
    retriever: HybridRetriever | None = None,
):
    """Create a generic retrieval tool for standalone compatibility."""
    del llm
    retriever = retriever or HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Retrieve supporting documents with hybrid search and return them as "
            "plain tool output. Use this only when route-specific retrieval is "
            "not needed."
        )
    )
    def retrieval_tool(
        query: str,
        top_k: int = 3,
    ) -> str:
        with measure_stage("tool.retrieval_tool"):
            docs = _retrieve_docs(retriever, query=query, top_k=top_k)
        record_retrieval_results("retrieval_tool", len(docs))
        return format_docs(docs, route="general", query=query)

    return retrieval_tool


def create_list_retrieval_tool(
    index_builder: IndexBuilder,
    chunks: list[Document],
    retriever: HybridRetriever | None = None,
):
    """Create the retrieval tool for list-style questions."""
    retriever = retriever or HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Use after router_tool returns 'list'. Retrieve a broader set of "
            "documents for list, enumeration, recommendation, or category "
            "questions about colleges and majors."
        )
    )
    def list_retrieval_tool(query: str) -> str:
        with measure_stage("tool.list_retrieval_tool"):
            docs = _retrieve_docs(
                retriever,
                query=query,
                top_k=DEFAULT_ROUTE_TOP_K["list"],
                use_filter=True,
            )
        record_retrieval_results("list_retrieval_tool", len(docs))
        return format_docs(docs, route="list", query=query)

    return list_retrieval_tool


def create_detail_retrieval_tool(
    index_builder: IndexBuilder,
    chunks: list[Document],
    retriever: HybridRetriever | None = None,
):
    """Create the retrieval tool for detailed questions."""
    retriever = retriever or HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Use after router_tool returns 'detail'. Retrieve focused supporting "
            "documents for detailed questions about a specific college, major, "
            "curriculum, training objective, or section."
        )
    )
    def detail_retrieval_tool(query: str) -> str:
        with measure_stage("tool.detail_retrieval_tool"):
            docs = _retrieve_docs(
                retriever,
                query=query,
                top_k=DEFAULT_ROUTE_TOP_K["detail"],
            )
        record_retrieval_results("detail_retrieval_tool", len(docs))
        return format_docs(docs, route="detail", query=query)

    return detail_retrieval_tool


def create_general_retrieval_tool(
    index_builder: IndexBuilder,
    chunks: list[Document],
    retriever: HybridRetriever | None = None,
):
    """Create the retrieval tool for general questions."""
    retriever = retriever or HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Use after router_tool returns 'general' and after "
            "query_rewrite_tool has produced a rewritten query. Retrieve "
            "supporting documents for general explanatory questions when the "
            "request is neither a list request nor a specific detailed lookup."
        )
    )
    def general_retrieval_tool(rewritten_query: str) -> str:
        with measure_stage("tool.general_retrieval_tool"):
            docs = _retrieve_docs(
                retriever,
                query=rewritten_query,
                top_k=DEFAULT_ROUTE_TOP_K["general"],
            )
        record_retrieval_results("general_retrieval_tool", len(docs))
        return format_docs(docs, route="general", query=rewritten_query)

    return general_retrieval_tool


class RetrievalTool:
    """Generic retrieval tool class for standalone usage."""

    def __init__(
        self,
        llm: BaseChatModel,
        index_builder: IndexBuilder,
        chunks: list[Document],
    ):
        self._tool = create_retrieval_tool(llm, index_builder, chunks)

    def invoke(self, query: str, top_k: int = 3) -> str:
        """Invoke the retrieval tool."""
        return self._tool.invoke({"query": query, "top_k": top_k})

    def __call__(self, query: str, top_k: int = 3) -> str:
        return self.invoke(query, top_k)
