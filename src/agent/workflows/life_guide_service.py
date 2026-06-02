"""生活指南工作流服务。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from agent.workflows.models import ResolvedWorkflowQuery, WorkflowTraceEvent
from domain.life_guide_knowledge.runtime import (
    LifeGuideKnowledgeRuntime,
    RuntimeUnavailableError,
)
from llm.health import ModelProviderHealthProbe, ModelProviderHealthResult
from shared.cache import NullRetrievalCache, RetrievalCache
from shared.observability.performance import (
    increment_tool_calls,
    measure_stage,
    record_retrieval_results,
)
from utils.text import truncate_text

_TOP_K = 5
_CACHE_DOMAIN = "life_guide"
_CACHE_MODEL_VERSION = "life-guide-v1"


class LifeGuideWorkflowService:
    """面向生活指南知识库的工作流编排服务。"""

    def __init__(
        self,
        *,
        knowledge_runtime: LifeGuideKnowledgeRuntime | None = None,
        health_probe: ModelProviderHealthProbe | None = None,
        retrieval_cache: RetrievalCache | None = None,
    ) -> None:
        self._runtime = knowledge_runtime or LifeGuideKnowledgeRuntime()
        self._health_probe = health_probe or ModelProviderHealthProbe()
        self._retrieval_cache = retrieval_cache or NullRetrievalCache()

    @property
    def is_initialized(self) -> bool:
        """运行时是否已经初始化。"""
        return self.runtime.is_initialized

    def ensure_initialized(self) -> None:
        """按需初始化生活指南运行时。"""
        self.runtime.ensure_initialized()

    def execute(
        self,
        *,
        user_query: str,
    ) -> dict[str, Any]:
        """执行生活指南检索工作流并返回结构化结果。"""
        cached_result = self._retrieval_cache.get_workflow_result(
            domain=_CACHE_DOMAIN,
            user_query=user_query,
            model_version=_CACHE_MODEL_VERSION,
        )
        if cached_result is not None:
            return cached_result

        self.ensure_initialized()
        runtime = self.runtime

        with measure_stage("tool.life_guide_retrieval_tool"):
            documents = runtime.retriever.hybrid_search(user_query, top_k=_TOP_K)

        increment_tool_calls(1)
        record_retrieval_results("life_guide_retrieval_tool", len(documents))
        trace_event = WorkflowTraceEvent(
            step="retrieval",
            source="retrieval",
            tool_name="life_guide_retrieval_tool",
            tool_args={"query": user_query},
            tool_output=self._build_trace_summary(documents, query=user_query),
        )
        retrieval_context = self._build_retrieval_context(
            documents,
            query=user_query,
        )
        resolved_query = ResolvedWorkflowQuery(
            source_query=user_query,
            route="life_guide",
            retrieval_query=user_query,
            documents=documents,
            retrieval_context=retrieval_context,
        )

        result = {
            "retrieval_context": retrieval_context,
            "evidence_bundle": self._build_evidence_bundle(documents),
            "resolved_queries": [resolved_query.to_dict()],
            "route_trace": ["life_guide"],
            "workflow_trace": [trace_event.to_dict()],
        }
        self._retrieval_cache.set_workflow_result(
            domain=_CACHE_DOMAIN,
            user_query=user_query,
            model_version=_CACHE_MODEL_VERSION,
            result=result,
        )
        return result

    def probe_model_provider(
        self,
        check_connection: bool = False,
    ) -> ModelProviderHealthResult:
        """返回模型提供方配置与可选连通性状态。"""
        return self._health_probe.probe(check_connection=check_connection)

    @property
    def runtime(self) -> LifeGuideKnowledgeRuntime:
        """按需返回运行时实例。"""
        return self._runtime

    def _build_retrieval_context(
        self,
        documents: list[Document],
        *,
        query: str,
    ) -> str:
        documents = list(documents[:_TOP_K])

        if not documents:
            return (
                "route=life_guide\n"
                f"query={query}\n"
                "result_count=0\n"
                "documents:\n"
                "No relevant documents were found."
            )

        parts = [
            "route=life_guide",
            f"query={query}",
            f"result_count={len(documents)}",
            "documents:",
        ]
        for index, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            parts.extend(
                [
                    f"[document {index}]",
                    f"category: {metadata.get('category', '')}",
                    f"service_name: {metadata.get('service_name', '')}",
                    f"sub_service_name: {metadata.get('sub_service_name', '')}",
                    f"doc_type: {metadata.get('doc_type', '')}",
                    f"source: {metadata.get('source', metadata.get('filename', ''))}",
                    "content:",
                    doc.page_content,
                    "---",
                ]
            )
        return "\n".join(parts)

    def _build_trace_summary(
        self,
        documents: list[Document],
        *,
        query: str,
    ) -> str:
        if not documents:
            return f"route=life_guide\nquery={query}\nresult_count=0"

        lines = [
            "route=life_guide",
            f"query={query}",
            f"result_count={len(documents)}",
        ]
        for index, doc in enumerate(documents[:3], start=1):
            metadata = doc.metadata
            lines.append(
                "[document "
                f"{index}] category={metadata.get('category', '')} "
                f"service_name={metadata.get('service_name', '')}"
            )
            lines.append(
                f"content_preview={truncate_text(doc.page_content.strip(), limit=120)}"
            )
        return "\n".join(lines)

    def _build_evidence_bundle(self, documents: list[Document]) -> list[dict[str, Any]]:
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


__all__ = [
    "LifeGuideWorkflowService",
    "ModelProviderHealthResult",
    "RuntimeUnavailableError",
]
