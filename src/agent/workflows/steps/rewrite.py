"""Query rewrite tool for explicit general-query refinement."""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llm.prompt_loader import get_query_rewrite_prompt
from agent.workflows.models import RewriteStepResult, WorkflowTraceEvent
from shared.observability.performance import increment_llm_calls, measure_stage

logger = logging.getLogger(__name__)

GENERIC_QUERY_MAPPINGS = {
    "学校介绍": "温州商学院介绍",
    "学校概况": "温州商学院介绍",
    "学校简介": "温州商学院介绍",
    "学院介绍": "温州商学院学院介绍",
    "学院概况": "温州商学院学院介绍",
    "学院简介": "温州商学院学院介绍",
    "专业": "温州商学院有哪些专业",
    "有哪些专业": "温州商学院有哪些专业",
    "学院": "温州商学院有哪些学院",
    "有哪些学院": "温州商学院有哪些学院",
}

EXPLICIT_FIELD_KEYWORDS = (
    "课程",
    "区别",
    "就业",
    "培养目标",
    "专业",
    "学院",
    "前景",
    "特色",
    "学什么",
    "是什么",
    "介绍",
)


def should_rewrite_query(
    query: str,
    *,
    has_domain_entity: bool,
) -> bool:
    """Return whether a general query is vague enough to justify rewriting."""
    normalized = re.sub(r"\s+", "", query)
    if not normalized:
        return False
    if len(normalized) > 10:
        return False
    if has_domain_entity:
        return False
    if normalized in GENERIC_QUERY_MAPPINGS:
        return True
    if any(keyword in normalized for keyword in EXPLICIT_FIELD_KEYWORDS):
        return False
    return True


def rewrite_query_rule_based(query: str) -> str | None:
    """Return a rewritten query for very generic requests, if possible."""
    normalized = re.sub(r"\s+", "", query)
    if not normalized:
        return None
    return GENERIC_QUERY_MAPPINGS.get(normalized)


def build_query_rewrite_step(llm: BaseChatModel):
    """Build a query rewrite step with the given LLM."""
    prompt = PromptTemplate(
        template=get_query_rewrite_prompt(),
        input_variables=["query"],
    )

    chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    def query_rewrite_step(query: str) -> str:
        """Rewrite a general query for retrieval."""
        with measure_stage("tool.query_rewrite_tool"):
            increment_llm_calls(1)
            result = chain.invoke(query).strip()
        if result != query:
            logger.info("Query rewritten: '%s' -> '%s'", query, result)
        else:
            logger.info("Query unchanged: '%s'", query)
        return result

    return query_rewrite_step


def run_query_rewrite_step(
    query: str,
    *,
    has_domain_entity: bool,
    llm_step,
) -> RewriteStepResult:
    """Resolve retrieval query for a general route."""
    if not should_rewrite_query(query, has_domain_entity=has_domain_entity):
        return RewriteStepResult(
            original_query=query,
            rewritten_query=query,
        )

    rule_based_rewrite = rewrite_query_rule_based(query)
    if rule_based_rewrite is not None:
        logger.info("Query rewritten by workflow rule: '%s' -> '%s'", query, rule_based_rewrite)
        return RewriteStepResult(
            original_query=query,
            rewritten_query=rule_based_rewrite,
            trace_event=WorkflowTraceEvent(
                step="query_rewrite",
                source="rule",
                tool_name="query_rewrite_tool",
                tool_args={"query": query},
                tool_output=rule_based_rewrite,
            ),
        )

    if llm_step is None:
        return RewriteStepResult(
            original_query=query,
            rewritten_query=query,
        )

    result = llm_step(query)
    return RewriteStepResult(
        original_query=query,
        rewritten_query=result,
        trace_event=WorkflowTraceEvent(
            step="query_rewrite",
            source="llm",
            tool_name="query_rewrite_tool",
            tool_args={"query": query},
            tool_output=result,
        ),
    )
