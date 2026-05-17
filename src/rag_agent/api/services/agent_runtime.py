"""Shared agent runtime reused by CLI and FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import AsyncIterator
import logging
import os
from threading import Lock
from typing import Any
import uuid

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from rag_agent.agent_modules.message_builder import build_agent_messages
from rag_agent.agent_modules.model import model as default_llm
from rag_agent.agent_modules.tools.query_decomposition import (
    _parse_decomposition,
    create_query_decomposition_tool,
    detect_decomposition_rule_based,
)
from rag_agent.agent_modules.tools.query_rewrite import (
    create_query_rewrite_tool,
    rewrite_query_rule_based,
    should_rewrite_query,
)
from rag_agent.agent_modules.tools.retrieval import (
    build_retrieval_context,
    extract_query_filters,
    retrieve_route_documents,
)
from rag_agent.agent_modules.tools.router import (
    _classify_query_rule_based,
    create_router_tool,
)
from rag_agent.api.services.workflow_prompts import (
    WORKFLOW_ANSWER_SYSTEM_PROMPT,
    build_final_answer_user_prompt,
)
from rag_agent.data_processing import chunk_documents, load_documents
from rag_agent.indexing import IndexBuilder
from rag_agent.memory_session.session import ManagedThread
from rag_agent.observability.performance import (
    increment_llm_calls,
    increment_tool_calls,
    measure_stage,
    record_prompt_messages,
    record_retrieval_results,
    record_thread_turn_count,
)
from rag_agent.retrieval.hybrid_search import HybridRetriever
from rag_agent.utils.path import get_raw_data_dir

logger = logging.getLogger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """Raised when the shared runtime cannot be initialized."""


@dataclass(slots=True)
class AgentInvocationResult:
    """Structured result returned by agent invocation."""

    answer: str
    messages: list[Any]
    raw_result: dict[str, Any]


@dataclass(slots=True)
class ResolvedWorkflowQuery:
    """Resolved execution plan for one query or sub-query."""

    source_query: str
    route: str
    retrieval_query: str
    documents: list[Document]
    retrieval_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_query": self.source_query,
            "route": self.route,
            "retrieval_query": self.retrieval_query,
            "document_count": len(self.documents),
            "retrieval_context": self.retrieval_context,
        }


@dataclass(slots=True)
class ModelProviderHealthResult:
    """Connectivity and configuration status for the upstream chat model."""

    configured: bool
    checked: bool
    reachable: bool | None
    model: str | None
    base_url: str | None
    detail: str | None = None


class AgentRuntime:
    """Reusable runtime for loading data, retrieval, and workflow execution."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False
        self._chunks: list[Any] = []
        self._index_builder: IndexBuilder | None = None
        self._retriever: HybridRetriever | None = None
        self._llm = default_llm
        self._router_tool: Any | None = None
        self._query_rewrite_tool: Any | None = None
        self._query_decomposition_tool: Any | None = None

    @property
    def is_initialized(self) -> bool:
        """Whether the runtime has been initialized."""
        return self._initialized

    def ensure_initialized(self) -> None:
        """Initialize data, vector index, and workflow helpers lazily."""
        if self._initialized:
            return

        with measure_stage("runtime.ensure_initialized"):
            with self._lock:
                if self._initialized:
                    return

                try:
                    with measure_stage("runtime.load_documents"):
                        docs = load_documents(get_raw_data_dir())
                    with measure_stage("runtime.chunk_documents"):
                        parents, children = chunk_documents(docs)
                    self._chunks = parents + children

                    builder = IndexBuilder()
                    with measure_stage("runtime.load_or_build_index"):
                        builder.load_or_build_index(self._chunks)
                    self._index_builder = builder

                    with measure_stage("runtime.build_shared_retriever"):
                        self._retriever = HybridRetriever(builder, self._chunks)

                    self._router_tool = create_router_tool(self._llm)
                    self._query_rewrite_tool = create_query_rewrite_tool(self._llm)
                    self._query_decomposition_tool = create_query_decomposition_tool(self._llm)
                    self._initialized = True
                    logger.info("Agent runtime initialized with %s chunks", len(self._chunks))
                except Exception as exc:
                    logger.exception("Failed to initialize agent runtime")
                    raise RuntimeUnavailableError("Failed to initialize agent runtime.") from exc

    def invoke(self, thread: ManagedThread) -> AgentInvocationResult:
        """Invoke the workflow with the default compressed retrieval context."""
        return self.invoke_with_strategy(thread, retrieval_context_strategy="compressed")

    def invoke_with_strategy(
        self,
        thread: ManagedThread,
        *,
        retrieval_context_strategy: str = "compressed",
    ) -> AgentInvocationResult:
        """Invoke the workflow for the current state of the thread."""
        with measure_stage("runtime.invoke"):
            final_messages, workflow_messages, raw_result = self._prepare_workflow(
                thread,
                retrieval_context_strategy=retrieval_context_strategy,
            )
            answer_message = self._generate_answer_message(final_messages)

        workflow_messages.append(answer_message)
        answer = self._extract_message_text(answer_message)
        raw_result["messages"] = workflow_messages
        return AgentInvocationResult(answer=answer, messages=workflow_messages, raw_result=raw_result)

    def stream(
        self,
        thread: ManagedThread,
        *,
        retrieval_context_strategy: str = "compressed",
    ) -> AsyncIterator[str]:
        """Stream only final assistant text chunks for the current workflow."""
        with measure_stage("runtime.stream.prepare"):
            final_messages, _, _ = self._prepare_workflow(
                thread,
                retrieval_context_strategy=retrieval_context_strategy,
            )

        async def iterator() -> AsyncIterator[str]:
            increment_llm_calls(1)
            with measure_stage("runtime.final_answer_generation"):
                async for chunk in self._llm.astream(final_messages):
                    text = self._extract_stream_text(chunk)
                    if text:
                        yield text

        return iterator()

    def _prepare_workflow(
        self,
        thread: ManagedThread,
        *,
        retrieval_context_strategy: str,
    ) -> tuple[list[Any], list[Any], dict[str, Any]]:
        """Prepare retrieval context and final generation messages."""
        self.ensure_initialized()
        if self._retriever is None:
            raise RuntimeUnavailableError("Shared retriever is not initialized.")

        current_query = self._get_current_query(thread)
        workflow_messages: list[Any] = [HumanMessage(content=current_query)]
        resolved_queries = self._resolve_queries(
            current_query,
            workflow_messages,
            retrieval_context_strategy=retrieval_context_strategy,
        )
        retrieval_context = self._join_retrieval_contexts(resolved_queries)
        final_messages = self._build_final_messages(
            thread=thread,
            query=current_query,
            retrieval_context=retrieval_context,
        )
        raw_result = {
            "query": current_query,
            "resolved_queries": [item.to_dict() for item in resolved_queries],
            "retrieval_context": retrieval_context,
        }
        return final_messages, workflow_messages, raw_result

    def _resolve_queries(
        self,
        query: str,
        workflow_messages: list[Any],
        *,
        retrieval_context_strategy: str,
    ) -> list[ResolvedWorkflowQuery]:
        sub_queries = self._resolve_sub_queries(query, workflow_messages)
        resolved: list[ResolvedWorkflowQuery] = []

        for sub_query in sub_queries:
            route = self._route_query(sub_query, workflow_messages)
            retrieval_query = self._rewrite_query_if_needed(
                route=route,
                query=sub_query,
                workflow_messages=workflow_messages,
            )
            documents = self._retrieve_documents(
                route=route,
                query=retrieval_query,
                workflow_messages=workflow_messages,
            )
            retrieval_context = build_retrieval_context(
                documents,
                route=route,
                query=retrieval_query,
                strategy=retrieval_context_strategy,
            )
            resolved.append(
                ResolvedWorkflowQuery(
                    source_query=sub_query,
                    route=route,
                    retrieval_query=retrieval_query,
                    documents=documents,
                    retrieval_context=retrieval_context,
                )
            )

        return resolved

    def _resolve_sub_queries(
        self,
        query: str,
        workflow_messages: list[Any],
    ) -> list[str]:
        decision, sub_queries = detect_decomposition_rule_based(query)
        if self._query_decomposition_tool is None:
            if decision is True and sub_queries:
                increment_tool_calls(1)
                with measure_stage("tool.query_decomposition_tool"):
                    result = self._format_decomposition_result(True, sub_queries)
                self._append_tool_trace(
                    workflow_messages,
                    tool_name="query_decomposition_tool",
                    tool_args={"query": query},
                    tool_output=result,
                )
                return sub_queries
            return [query]

        increment_tool_calls(1)
        result = self._query_decomposition_tool.invoke(query)
        self._append_tool_trace(
            workflow_messages,
            tool_name="query_decomposition_tool",
            tool_args={"query": query},
            tool_output=result,
        )
        decomposable, parsed_sub_queries = _parse_decomposition(result)
        if decomposable and parsed_sub_queries:
            return parsed_sub_queries
        return [query]

    def _route_query(self, query: str, workflow_messages: list[Any]) -> str:
        heuristic_route = _classify_query_rule_based(query)
        if heuristic_route is not None:
            increment_tool_calls(1)
            with measure_stage("tool.router_tool"):
                result = heuristic_route
            logger.info("Router classified query by workflow rule as: %s", result)
            self._append_tool_trace(
                workflow_messages,
                tool_name="router_tool",
                tool_args={"query": query},
                tool_output=result,
            )
            return result

        if self._router_tool is None:
            raise RuntimeUnavailableError("Router tool is not initialized.")

        increment_tool_calls(1)
        result = self._router_tool.invoke(query)
        self._append_tool_trace(
            workflow_messages,
            tool_name="router_tool",
            tool_args={"query": query},
            tool_output=result,
        )
        return result

    def _rewrite_query_if_needed(
        self,
        *,
        route: str,
        query: str,
        workflow_messages: list[Any],
    ) -> str:
        if route != "general" or self._retriever is None:
            return query

        has_domain_entity = bool(extract_query_filters(self._retriever, query))
        if not should_rewrite_query(query, has_domain_entity=has_domain_entity):
            return query

        increment_tool_calls(1)
        rule_based_rewrite = rewrite_query_rule_based(query)
        if rule_based_rewrite is not None:
            with measure_stage("tool.query_rewrite_tool"):
                result = rule_based_rewrite
            logger.info("Query rewritten by workflow rule: '%s' -> '%s'", query, result)
            self._append_tool_trace(
                workflow_messages,
                tool_name="query_rewrite_tool",
                tool_args={"query": query},
                tool_output=result,
            )
            return result

        if self._query_rewrite_tool is None:
            return query

        result = self._query_rewrite_tool.invoke(query)
        self._append_tool_trace(
            workflow_messages,
            tool_name="query_rewrite_tool",
            tool_args={"query": query},
            tool_output=result,
        )
        return result

    def _retrieve_documents(
        self,
        *,
        route: str,
        query: str,
        workflow_messages: list[Any],
    ) -> list[Document]:
        if self._retriever is None:
            raise RuntimeUnavailableError("Shared retriever is not initialized.")

        increment_tool_calls(1)
        with measure_stage(f"tool.{route}_retrieval_tool"):
            documents = retrieve_route_documents(
                self._retriever,
                route=route,
                query=query,
            )

        record_retrieval_results(f"{route}_retrieval_tool", len(documents))
        tool_args = {"query": query}
        if route == "general":
            tool_args = {"rewritten_query": query}

        self._append_tool_trace(
            workflow_messages,
            tool_name=f"{route}_retrieval_tool",
            tool_args=tool_args,
            tool_output=build_retrieval_context(
                documents,
                route=route,
                query=query,
                strategy="passthrough",
            ),
        )
        return documents

    def _build_final_messages(
        self,
        *,
        thread: ManagedThread,
        query: str,
        retrieval_context: str,
    ) -> list[Any]:
        with measure_stage("runtime.build_agent_messages"):
            history_messages = build_agent_messages(thread=thread)
        record_thread_turn_count(len(thread.turns))
        final_messages = [
            SystemMessage(content=WORKFLOW_ANSWER_SYSTEM_PROMPT),
            *history_messages,
            HumanMessage(content=build_final_answer_user_prompt(query, retrieval_context)),
        ]
        record_prompt_messages(final_messages)
        return final_messages

    def _generate_answer_message(self, final_messages: list[Any]) -> AIMessage:
        increment_llm_calls(1)
        with measure_stage("runtime.final_answer_generation"):
            response = self._llm.invoke(final_messages)
        return AIMessage(content=self._extract_message_text(response))

    def _join_retrieval_contexts(self, resolved_queries: list[ResolvedWorkflowQuery]) -> str:
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

    def _get_current_query(self, thread: ManagedThread) -> str:
        turns = thread.turns
        if not turns or turns[-1].user_message is None:
            raise RuntimeUnavailableError("Current thread does not contain a user query.")
        return turns[-1].user_message.content

    def _append_tool_trace(
        self,
        workflow_messages: list[Any],
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: str,
    ) -> None:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        workflow_messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": tool_name,
                        "args": tool_args,
                        "type": "tool_call",
                    }
                ],
            )
        )
        workflow_messages.append(
            ToolMessage(
                content=tool_output,
                tool_call_id=call_id,
            )
        )

    def _format_decomposition_result(self, decomposable: bool, sub_queries: list[str]) -> str:
        if not decomposable:
            return "DECOMPOSABLE: false"
        parts = ["DECOMPOSABLE: true", "SUB_QUERIES:"]
        for index, item in enumerate(sub_queries, start=1):
            parts.append(f"{index}. {item}")
        return "\n".join(parts)

    def probe_model_provider(
        self,
        check_connection: bool = False,
    ) -> ModelProviderHealthResult:
        """Return model-provider configuration and optional live connectivity status."""
        load_dotenv()
        model = os.getenv("QWEN_MODEL", "qwen3.5-plus").strip() or "qwen3.5-plus"
        base_url = self._normalize_optional_env("QWEN_BASE_URL")
        api_key = self._normalize_optional_env("QWEN_API_KEY")
        configured = bool(base_url and api_key)

        if not configured:
            return ModelProviderHealthResult(
                configured=False,
                checked=check_connection,
                reachable=False if check_connection else None,
                model=model,
                base_url=base_url,
                detail="Missing QWEN_BASE_URL or QWEN_API_KEY.",
            )

        if not check_connection:
            return ModelProviderHealthResult(
                configured=True,
                checked=False,
                reachable=None,
                model=model,
                base_url=base_url,
                detail=None,
            )

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=10.0,
                max_retries=0,
            )
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return ModelProviderHealthResult(
                configured=True,
                checked=True,
                reachable=True,
                model=model,
                base_url=base_url,
                detail="Model provider responded successfully.",
            )
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            logger.warning("Model provider health probe failed: %s", exc)
            return ModelProviderHealthResult(
                configured=True,
                checked=True,
                reachable=False,
                model=model,
                base_url=base_url,
                detail=self._describe_exception(exc),
            )

    def _normalize_optional_env(self, key: str) -> str | None:
        value = os.getenv(key)
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def _extract_stream_text(self, chunk: Any) -> str:
        """Extract plain text tokens from a streamed model chunk."""
        if chunk is None:
            return ""

        content = getattr(chunk, "content", chunk)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    continue

                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)

            return "".join(parts)

        return ""

    def _extract_message_text(self, message: Any) -> str:
        """Extract plain-text content from a model response message."""
        return self._extract_stream_text(getattr(message, "content", message))

    def _describe_exception(self, exc: BaseException) -> str:
        """Return the deepest available exception message."""
        messages: list[str] = []
        current: BaseException | None = exc
        while current is not None:
            text = str(current).strip()
            if text:
                messages.append(text)
            current = current.__cause__ or current.__context__

        if not messages:
            return exc.__class__.__name__
        return messages[-1]
