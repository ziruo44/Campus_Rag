"""Query rewrite tool for explicit general-query refinement."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from rag_agent.utils.prompt_loader import get_query_rewrite_prompt

logger = logging.getLogger(__name__)


def create_query_rewrite_tool(llm: BaseChatModel):
    """Create a query rewrite tool with the given LLM."""
    prompt = PromptTemplate(
        template=get_query_rewrite_prompt(),
        input_variables=["query"],
    )

    chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    @tool(
        description=(
            "Use only after router_tool returns 'general'. Rewrite vague or "
            "underspecified general questions into a clearer retrieval query "
            "before calling general_retrieval_tool. The returned text should "
            "be passed into general_retrieval_tool as the rewritten_query "
            "argument. If the query is already clear, return it unchanged."
        )
    )
    def query_rewrite_tool(query: str) -> str:
        """Rewrite a general query for retrieval."""
        result = chain.invoke(query).strip()
        if result != query:
            logger.info("Query rewritten: '%s' -> '%s'", query, result)
        else:
            logger.info("Query unchanged: '%s'", query)
        return result

    return query_rewrite_tool


class QueryRewriteTool:
    """Query rewrite tool class for standalone usage."""

    def __init__(self, llm: BaseChatModel):
        self._tool = create_query_rewrite_tool(llm)

    def invoke(self, query: str) -> str:
        """Invoke the query rewrite tool."""
        return self._tool.invoke(query)

    def __call__(self, query: str) -> str:
        return self.invoke(query)
