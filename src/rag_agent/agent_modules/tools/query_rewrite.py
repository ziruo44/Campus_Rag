"""Query rewrite tool for explicit general-query refinement."""

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from rag_agent.observability.performance import increment_llm_calls, measure_stage
from rag_agent.utils.prompt_loader import get_query_rewrite_prompt

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
    if any(keyword in normalized for keyword in EXPLICIT_FIELD_KEYWORDS):
        return False
    return True


def rewrite_query_rule_based(query: str) -> str | None:
    """Return a rewritten query for very generic requests, if possible."""
    normalized = re.sub(r"\s+", "", query)
    if not normalized:
        return None
    return GENERIC_QUERY_MAPPINGS.get(normalized)


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
        with measure_stage("tool.query_rewrite_tool"):
            increment_llm_calls(1)
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
