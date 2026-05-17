"""RAG Agent tool assembly for explicit tool-driven routing."""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from rag_agent.agent_modules.model import model as default_llm
from rag_agent.agent_modules.tools.query_decomposition import (
    create_query_decomposition_tool,
)
from rag_agent.agent_modules.tools.query_rewrite import create_query_rewrite_tool
from rag_agent.agent_modules.tools.retrieval import (
    create_detail_retrieval_tool,
    create_general_retrieval_tool,
    create_list_retrieval_tool,
)
from rag_agent.agent_modules.tools.router import create_router_tool
from rag_agent.indexing.index_builder import IndexBuilder
from rag_agent.prompts import get_system_prompt
from rag_agent.retrieval.hybrid_search import HybridRetriever

logger = logging.getLogger(__name__)


def create_rag_agent_tools(
    llm: BaseChatModel,
    index_builder: IndexBuilder,
    chunks: list[Any],
    retriever: HybridRetriever | None = None,
) -> list:
    """Create all tools for the LangChain agent."""
    return [
        create_query_decomposition_tool(llm),
        create_router_tool(llm),
        create_query_rewrite_tool(llm),
        create_list_retrieval_tool(index_builder, chunks, retriever=retriever),
        create_detail_retrieval_tool(index_builder, chunks, retriever=retriever),
        create_general_retrieval_tool(index_builder, chunks, retriever=retriever),
    ]


class RagAgent:
    """RAG Agent configuration holder for an external tool-calling agent."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        index_builder: IndexBuilder | None = None,
        chunks: list[Any] | None = None,
        system_prompt: str | None = None,
        retriever: HybridRetriever | None = None,
    ):
        self.llm = llm or default_llm
        self.system_prompt = system_prompt or get_system_prompt()
        self.index_builder = index_builder
        self.chunks = chunks or []
        self._tools: list[Any] = []

        if index_builder and chunks:
            self._tools = create_rag_agent_tools(
                llm=self.llm,
                index_builder=index_builder,
                chunks=chunks,
                retriever=retriever,
            )
            logger.info("Initialized %s agent tools", len(self._tools))
        else:
            logger.warning(
                "IndexBuilder or chunks not provided, retrieval tools not initialized"
            )

    def invoke(self, query: str, use_routing: bool = True) -> str:
        """Standalone helper for compatibility.

        This class no longer executes the routing loop itself. The caller should
        pass `get_tools()` and `get_system_prompt()` into a tool-calling agent.
        """
        del use_routing
        return query

    def run(self, query: str) -> str:
        """Alias for invoke."""
        return self.invoke(query)

    def get_tools(self) -> list:
        """Get tools for the external tool-calling agent."""
        return self._tools

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt


def create_agent(
    index_builder: IndexBuilder,
    chunks: list[Any],
    llm: BaseChatModel | None = None,
    system_prompt: str | None = None,
    retriever: HybridRetriever | None = None,
) -> RagAgent:
    """Create a configured RAG Agent."""
    return RagAgent(
        llm=llm,
        index_builder=index_builder,
        chunks=chunks,
        system_prompt=system_prompt,
        retriever=retriever,
    )
