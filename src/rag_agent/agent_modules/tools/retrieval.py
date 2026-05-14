"""Retrieval tools for explicit routed retrieval."""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from rag_agent.indexing.index_builder import IndexBuilder
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


def create_retrieval_tool(
    llm: BaseChatModel,
    index_builder: IndexBuilder,
    chunks: list[Document],
):
    """Create a generic retrieval tool for standalone compatibility."""
    del llm
    retriever = HybridRetriever(index_builder, chunks)

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
        docs = _retrieve_docs(retriever, query=query, top_k=top_k)
        return format_docs(docs, route="general", query=query)

    return retrieval_tool


def create_list_retrieval_tool(
    index_builder: IndexBuilder,
    chunks: list[Document],
):
    """Create the retrieval tool for list-style questions."""
    retriever = HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Use after router_tool returns 'list'. Retrieve a broader set of "
            "documents for list, enumeration, recommendation, or category "
            "questions about colleges and majors."
        )
    )
    def list_retrieval_tool(query: str) -> str:
        docs = _retrieve_docs(retriever, query=query, top_k=DEFAULT_ROUTE_TOP_K["list"], use_filter=True)
        return format_docs(docs, route="list", query=query)

    return list_retrieval_tool


def create_detail_retrieval_tool(
    index_builder: IndexBuilder,
    chunks: list[Document],
):
    """Create the retrieval tool for detailed questions."""
    retriever = HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Use after router_tool returns 'detail'. Retrieve focused supporting "
            "documents for detailed questions about a specific college, major, "
            "curriculum, training objective, or section."
        )
    )
    def detail_retrieval_tool(query: str) -> str:
        docs = _retrieve_docs(retriever, query=query, top_k=DEFAULT_ROUTE_TOP_K["detail"])
        return format_docs(docs, route="detail", query=query)

    return detail_retrieval_tool


def create_general_retrieval_tool(
    index_builder: IndexBuilder,
    chunks: list[Document],
):
    """Create the retrieval tool for general questions."""
    retriever = HybridRetriever(index_builder, chunks)

    @tool(
        description=(
            "Use after router_tool returns 'general' and after "
            "query_rewrite_tool has produced a rewritten query. Retrieve "
            "supporting documents for general explanatory questions when the "
            "request is neither a list request nor a specific detailed lookup."
        )
    )
    def general_retrieval_tool(rewritten_query: str) -> str:
        docs = _retrieve_docs(retriever, query=rewritten_query, top_k=DEFAULT_ROUTE_TOP_K["general"])
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
