"""专业知识工作流服务。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from agent.workflows.models import ResolvedWorkflowQuery, WorkflowTraceEvent
from agent.workflows.steps.decomposition import (
    build_query_decomposition_step,
    run_query_decomposition_step,
)
from agent.workflows.steps.retrieval import build_retrieval_context, run_retrieval_step
from agent.workflows.steps.rewrite import build_query_rewrite_step, run_query_rewrite_step
from agent.workflows.steps.routing import build_router_step, run_routing_step
from domain.major_knowledge.retrieval.filters import extract_query_filters
from domain.major_knowledge.runtime import KnowledgeRuntime, RuntimeUnavailableError
from llm.health import ModelProviderHealthProbe, ModelProviderHealthResult
from llm.model import model as default_llm
from shared.observability.performance import (
    increment_tool_calls,
    measure_stage,
    record_retrieval_results,
)

logger = logging.getLogger(__name__)


class MajorKnowledgeWorkflowService:
    """面向专业知识库的工作流编排服务。"""

    def __init__(
        self,
        *,
        knowledge_runtime: KnowledgeRuntime | None = None,
        health_probe: ModelProviderHealthProbe | None = None,
        llm: BaseChatModel = default_llm,
    ) -> None:
        self._runtime = knowledge_runtime or KnowledgeRuntime()
        self._health_probe = health_probe or ModelProviderHealthProbe()
        self._llm = llm
        self._router_step: Any | None = None
        self._query_rewrite_step: Any | None = None
        self._query_decomposition_step: Any | None = None

    @property
    def is_initialized(self) -> bool:
        """运行时是否已经初始化。"""
        return self.runtime.is_initialized

    def ensure_initialized(self) -> None:
        """按需初始化专业知识库运行时。"""
        self.runtime.ensure_initialized()

    def execute(
        self,
        *,
        user_query: str,
        retrieval_context_strategy: str = "passthrough",
    ) -> dict[str, Any]:
        """执行专业知识工作流并返回结构化结果。"""
        self.ensure_initialized()
        self._ensure_steps()

        trace_events: list[WorkflowTraceEvent] = []
        resolved_queries = self._resolve_queries(
            user_query,
            trace_events,
            retrieval_context_strategy=retrieval_context_strategy,
        )
        retrieval_context = self._join_retrieval_contexts(resolved_queries)
        documents = [doc for item in resolved_queries for doc in item.documents]
        resolved_query_dicts = [item.to_dict() for item in resolved_queries]

        return {
            "retrieval_context": retrieval_context,
            "evidence_bundle": self._build_evidence_bundle(documents),
            "resolved_queries": resolved_query_dicts,
            "route_trace": [item["route"] for item in resolved_query_dicts if item.get("route")],
            "workflow_trace": [event.to_dict() for event in trace_events],
        }

    def probe_model_provider(
        self,
        check_connection: bool = False,
    ) -> ModelProviderHealthResult:
        """返回模型提供方配置与可选连通性状态。"""
        return self._health_probe.probe(check_connection=check_connection)

    def _resolve_queries(
        self,
        query: str,
        trace_events: list[WorkflowTraceEvent],
        *,
        retrieval_context_strategy: str,
    ) -> list[ResolvedWorkflowQuery]:
        decomposition_result = run_query_decomposition_step(
            query,
            self._query_decomposition_step,
        )
        increment_tool_calls(1)
        trace_events.append(decomposition_result.trace_event)

        sub_queries = decomposition_result.sub_queries if decomposition_result.decomposable else [query]
        resolved: list[ResolvedWorkflowQuery] = []

        for sub_query in sub_queries:
            routing_result = run_routing_step(sub_query, self._router_step)
            increment_tool_calls(1)
            trace_events.append(routing_result.trace_event)

            retrieval_query = sub_query
            if routing_result.route == "general":
                runtime = self.runtime
                has_domain_entity = bool(
                    extract_query_filters(runtime.retriever, sub_query)
                )
                rewrite_result = run_query_rewrite_step(
                    sub_query,
                    has_domain_entity=has_domain_entity,
                    llm_step=self._query_rewrite_step,
                )
                if rewrite_result.trace_event is not None:
                    increment_tool_calls(1)
                    trace_events.append(rewrite_result.trace_event)
                retrieval_query = rewrite_result.rewritten_query

            with measure_stage(f"tool.{routing_result.route}_retrieval_tool"):
                runtime = self.runtime
                retrieval_result = run_retrieval_step(
                    runtime.retriever,
                    route=routing_result.route,
                    query=retrieval_query,
                    parent_documents=runtime.parent_documents,
                )
            increment_tool_calls(1)
            record_retrieval_results(
                retrieval_result.trace_event.tool_name,
                len(retrieval_result.documents),
            )
            trace_events.append(retrieval_result.trace_event)

            retrieval_context = build_retrieval_context(
                retrieval_result.documents,
                route=routing_result.route,
                query=retrieval_query,
                strategy=retrieval_context_strategy,
            )
            resolved.append(
                ResolvedWorkflowQuery(
                    source_query=sub_query,
                    route=routing_result.route,
                    retrieval_query=retrieval_query,
                    documents=retrieval_result.documents,
                    retrieval_context=retrieval_context,
                )
            )

        return resolved

    def _join_retrieval_contexts(
        self,
        resolved_queries: list[ResolvedWorkflowQuery],
    ) -> str:
        if len(resolved_queries) == 1:
            return resolved_queries[0].retrieval_context

        parts: list[str] = []
        for index, item in enumerate(resolved_queries, start=1):
            parts.extend(
                [
                    f"[sub_query {index}]",
                    f"source_query={item.source_query}",
                    item.retrieval_context,
                ]
            )
        return "\n".join(parts)

    @property
    def runtime(self) -> KnowledgeRuntime:
        """按需返回运行时实例。"""
        return self._runtime

    def _ensure_steps(self) -> None:
        if self._router_step is None:
            self._router_step = build_router_step(self._llm)
        if self._query_rewrite_step is None:
            self._query_rewrite_step = build_query_rewrite_step(self._llm)
        if self._query_decomposition_step is None:
            self._query_decomposition_step = build_query_decomposition_step(self._llm)

    def _build_evidence_bundle(
        self,
        documents,
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for doc in documents[:5]:
            metadata = dict(doc.metadata)
            evidence.append(
                {
                    "content": doc.page_content[:280],
                    "source": str(metadata.get("source", "") or metadata.get("filename", "")),
                    "metadata": metadata,
                }
            )
        return evidence


KnowledgeWorkflowService = MajorKnowledgeWorkflowService

__all__ = [
    "MajorKnowledgeWorkflowService",
    "KnowledgeWorkflowService",
    "ModelProviderHealthResult",
    "RuntimeUnavailableError",
]
