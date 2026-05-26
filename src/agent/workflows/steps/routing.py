"""Router tool for query classification."""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from agent.workflows.models import RoutingStepResult, WorkflowTraceEvent
from llm.prompt_loader import get_router_prompt
from shared.observability.performance import increment_llm_calls, measure_stage

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

COLLECTION_SCOPE_PATTERNS = (
    "各学院",
    "所有学院",
    "全部学院",
    "各专业",
    "所有专业",
    "全部专业",
    "分别",
)

COLLECTION_INTRO_PATTERNS = (
    "介绍",
    "简介",
    "概况",
    "详细",
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

    collection_scope = _contains_any(normalized, COLLECTION_SCOPE_PATTERNS)
    collection_intro = _contains_any(normalized, COLLECTION_INTRO_PATTERNS)
    if collection_scope and collection_intro and any(token in normalized for token in ("学院", "专业")):
        return "general"

    if _contains_any(normalized, LIST_PATTERNS):
        return "list"

    if _contains_any(normalized, GENERAL_PATTERNS):
        return "general"

    if _contains_any(normalized, DETAIL_PATTERNS):
        return "detail"

    return None


def build_router_step(llm: BaseChatModel):
    """Build a pure LLM router step with the given LLM."""
    prompt = ChatPromptTemplate.from_template(get_router_prompt())
    chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    def router_step(query: str) -> str:
        """Classify the user query into a route label."""
        with measure_stage("tool.router_tool"):
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

    return router_step


def run_routing_step(
    query: str,
    llm_step,
) -> RoutingStepResult:
    """Resolve one route through rule-first routing."""
    heuristic_route = _classify_query_rule_based(query)
    if heuristic_route is not None:
        logger.info("Router classified query by workflow rule as: %s", heuristic_route)
        return RoutingStepResult(
            route=heuristic_route,
            trace_event=WorkflowTraceEvent(
                step="routing",
                source="rule",
                tool_name="router_tool",
                tool_args={"query": query},
                tool_output=heuristic_route,
            ),
        )

    if llm_step is None:
        raise ValueError("Router step is not initialized.")

    result = llm_step(query)
    return RoutingStepResult(
        route=result,
        trace_event=WorkflowTraceEvent(
            step="routing",
            source="llm",
            tool_name="router_tool",
            tool_args={"query": query},
            tool_output=result,
        ),
    )
