"""Router tool for query classification."""

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from rag_agent.prompts import get_router_prompt

logger = logging.getLogger(__name__)

QueryType = Literal["list", "detail", "general"]


def create_router_tool(llm: BaseChatModel):
    """Create a router tool with the given LLM."""
    prompt = ChatPromptTemplate.from_template(get_router_prompt())
    chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    @tool(
        description=(
            "Classify the user's question into exactly one route: 'list', "
            "'detail', or 'general'. This tool must be called first for every "
            "user request."
        )
    )
    def router_tool(query: str) -> str:
        """Classify the user query into a route label."""
        result = chain.invoke(query).strip().lower()
        if result not in ["list", "detail", "general"]:
            logger.warning(
                "Unexpected router result: %s, defaulting to 'general'",
                result,
            )
            return "general"
        logger.info("Router classified query as: %s", result)
        return result

    return router_tool


class RouterTool:
    """Router tool class for standalone usage."""

    def __init__(self, llm: BaseChatModel):
        self._tool = create_router_tool(llm)

    def invoke(self, query: str) -> str:
        """Invoke the router tool."""
        return self._tool.invoke(query)

    def __call__(self, query: str) -> str:
        return self.invoke(query)
