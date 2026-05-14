"""Query decomposition tool for multi-intent queries."""

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from rag_agent.utils.prompt_loader import get_query_decomposition_prompt

logger = logging.getLogger(__name__)


def create_query_decomposition_tool(llm: BaseChatModel):
    """Create a query decomposition tool with the given LLM."""
    prompt = PromptTemplate(
        template=get_query_decomposition_prompt(),
        input_variables=["query"],
    )
    chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    @tool(
        description=(
            "Analyze whether the user's question contains multiple independent "
            "sub-questions. If the user asks about multiple things in one query "
            "(e.g., listing majors AND describing features in one sentence), "
            "this tool will break it down. Returns 'DECOMPOSABLE: true' with "
            "sub-queries or 'DECOMPOSABLE: false' if no decomposition is needed. "
            "Call this BEFORE router_tool when the query seems compound."
        )
    )
    def query_decomposition_tool(query: str) -> str:
        """Analyze and decompose a multi-intent query into sub-queries."""
        result = chain.invoke(query).strip()

        decomposable, sub_queries = _parse_decomposition(result)
        if decomposable:
            logger.info(
                "Query decomposed into %d sub-queries: %s",
                len(sub_queries),
                sub_queries,
            )
        else:
            logger.info("Query is a single intent, no decomposition needed")
        return result

    return query_decomposition_tool


def _parse_decomposition(result: str) -> tuple[bool, list[str]]:
    """Parse the decomposition tool output into structured data."""
    decomposable = False
    sub_queries = []

    for line in result.strip().splitlines():
        line = line.strip()
        if re.match(r"^DECOMPOSABLE:\s*true", line, re.IGNORECASE):
            decomposable = True
        elif re.match(r"^\d+\.\s+", line):
            query = re.sub(r"^\d+\.\s*", "", line)
            if query:
                sub_queries.append(query)

    return decomposable, sub_queries


class QueryDecompositionTool:
    """Query decomposition tool class for standalone usage."""

    def __init__(self, llm: BaseChatModel):
        self._tool = create_query_decomposition_tool(llm)

    def invoke(self, query: str) -> str:
        """Invoke the query decomposition tool."""
        return self._tool.invoke(query)

    def __call__(self, query: str) -> str:
        return self.invoke(query)
