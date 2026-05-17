"""Router tool for query classification."""

import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from rag_agent.observability.performance import increment_llm_calls, measure_stage
from rag_agent.prompts import get_router_prompt

logger = logging.getLogger(__name__)

QueryType = Literal["list", "detail", "general"]

LIST_PATTERNS = (
    "有哪些",
    "有哪",
    "哪几个",
    "哪几种",
    "都有什么",
    "分别有哪些",
    "列举",
    "名单",
    "列表",
    "汇总",
    "推荐",
)

DETAIL_PATTERNS = (
    "区别",
    "不同",
    "差异",
    "比较",
    "对比",
    "哪个好",
    "介绍",
    "怎么样",
    "学什么",
    "是什么",
    "就业",
    "前景",
    "课程",
    "培养目标",
    "特色",
    "方向",
)

GENERAL_PATTERNS = (
    "学校介绍",
    "学校概况",
    "学校简介",
    "学校怎么样",
    "学院概况",
    "学院简介",
    "温州商学院介绍",
    "温州商学院概况",
    "温州商学院简介",
)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", "", query).lower()


def _contains_any(query: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in query for pattern in patterns)


def _classify_query_rule_based(query: str) -> QueryType | None:
    """Use conservative keyword rules before falling back to the LLM router."""
    normalized = _normalize_query(query)
    if not normalized:
        return None

    if _contains_any(normalized, LIST_PATTERNS):
        return "list"

    if _contains_any(normalized, GENERAL_PATTERNS):
        return "general"

    if _contains_any(normalized, DETAIL_PATTERNS):
        return "detail"

    return None


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
        with measure_stage("tool.router_tool"):
            heuristic_route = _classify_query_rule_based(query)
            if heuristic_route is not None:
                logger.info("Router classified query by rule as: %s", heuristic_route)
                return heuristic_route

            increment_llm_calls(1)
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
