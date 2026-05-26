"""Query decomposition tool for multi-intent queries."""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from agent.workflows.models import DecompositionStepResult, WorkflowTraceEvent
from llm.prompt_loader import get_query_decomposition_prompt
from shared.observability.performance import increment_llm_calls, measure_stage

logger = logging.getLogger(__name__)

MULTI_INTENT_SPLITTERS = (
    "，",
    ",",
    "；",
    ";",
    "以及",
    "并且",
    "同时",
    "另外",
    "还有",
)

SINGLE_INTENT_KEYWORDS = (
    "区别",
    "对比",
    "比较",
    "介绍",
    "培养目标",
    "课程",
    "就业",
    "前景",
    "特色",
    "学什么",
    "是什么",
)

FOLLOW_UP_DETAIL_PATTERNS = (
    "分别介绍",
    "分别详细介绍",
    "分别具体介绍",
    "分别详细说一下",
    "分别具体说一下",
    "分别说一下",
)


def detect_decomposition_rule_based(query: str) -> tuple[bool | None, list[str]]:
    """Return a heuristic decomposition decision and optional sub-queries."""
    normalized = " ".join(query.strip().split())
    if not normalized:
        return False, []

    if any(keyword in normalized for keyword in SINGLE_INTENT_KEYWORDS):
        comparison_like = any(keyword in normalized for keyword in ("区别", "对比", "比较"))
        if comparison_like and not any(token in normalized for token in ("同时", "另外", "还有")):
            return False, []

    if not any(token in normalized for token in MULTI_INTENT_SPLITTERS):
        return False, []

    parts = _split_multi_intent_query(normalized)
    if len(parts) >= 2 and all(len(part) >= 4 for part in parts):
        return True, parts

    return None, []


def _split_multi_intent_query(query: str) -> list[str]:
    """Split a clearly compound query into standalone sub-queries."""
    raw_parts = re.split(r"[，；,;]", query)
    parts = [part.strip() for part in raw_parts if part.strip()]
    if len(parts) >= 2:
        return _contextualize_split_parts(parts)

    for token in ("以及", "并且", "同时", "另外", "还有"):
        if token not in query:
            continue
        split_parts = [part.strip() for part in query.split(token) if part.strip()]
        if len(split_parts) >= 2:
            return _contextualize_split_parts(split_parts)

    return []


def _contextualize_split_parts(parts: list[str]) -> list[str]:
    if len(parts) < 2:
        return parts

    contextualized = [parts[0]]
    for index in range(1, len(parts)):
        previous = contextualized[index - 1]
        contextualized.append(_rewrite_follow_up_query(previous, parts[index]))
    return contextualized


def _rewrite_follow_up_query(previous: str, current: str) -> str:
    normalized = current.replace(" ", "")
    if not any(pattern in normalized for pattern in FOLLOW_UP_DETAIL_PATTERNS):
        return current

    suffix = current
    if suffix.startswith("分别"):
        suffix = suffix[len("分别"):]
    suffix = suffix.strip() or "介绍一下"

    if "有哪些学院" in previous:
        return previous.replace("有哪些学院", f"各学院{suffix}")
    if "有哪些专业" in previous:
        return previous.replace("有哪些专业", f"各专业{suffix}")

    if "学院" in previous and "学院" not in current:
        return f"{previous}{suffix}"
    if "专业" in previous and "专业" not in current:
        return f"{previous}{suffix}"
    return current


def build_query_decomposition_step(llm: BaseChatModel):
    """Build a decomposition step callable with the given LLM."""
    prompt = PromptTemplate(
        template=get_query_decomposition_prompt(),
        input_variables=["query"],
    )
    chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    def query_decomposition_step(query: str) -> str:
        """Analyze and decompose a multi-intent query into sub-queries."""
        with measure_stage("tool.query_decomposition_tool"):
            increment_llm_calls(1)
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

    return query_decomposition_step


def run_query_decomposition_step(
    query: str,
    llm_step,
) -> DecompositionStepResult:
    """Resolve sub-queries through rule-first decomposition."""
    decision, sub_queries = detect_decomposition_rule_based(query)
    if decision is True and sub_queries:
        result = _format_decomposition_result(True, sub_queries)
        return DecompositionStepResult(
            decomposable=True,
            sub_queries=sub_queries,
            trace_event=WorkflowTraceEvent(
                step="query_decomposition",
                source="rule",
                tool_name="query_decomposition_tool",
                tool_args={"query": query},
                tool_output=result,
            ),
        )

    if llm_step is None:
        return DecompositionStepResult(
            decomposable=False,
            sub_queries=[query],
            trace_event=WorkflowTraceEvent(
                step="query_decomposition",
                source="rule",
                tool_name="query_decomposition_tool",
                tool_args={"query": query},
                tool_output=_format_decomposition_result(False, []),
            ),
        )

    result = llm_step(query)
    decomposable, parsed_sub_queries = _parse_decomposition(result)
    return DecompositionStepResult(
        decomposable=decomposable,
        sub_queries=parsed_sub_queries or [query],
        trace_event=WorkflowTraceEvent(
            step="query_decomposition",
            source="llm",
            tool_name="query_decomposition_tool",
            tool_args={"query": query},
            tool_output=result,
        ),
    )


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


def _format_decomposition_result(
    decomposable: bool,
    sub_queries: list[str],
) -> str:
    if not decomposable:
        return "DECOMPOSABLE: false"
    parts = ["DECOMPOSABLE: true", "SUB_QUERIES:"]
    for index, item in enumerate(sub_queries, start=1):
        parts.append(f"{index}. {item}")
    return "\n".join(parts)
