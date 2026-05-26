"""Typed workflow results shared across orchestration steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(slots=True)
class WorkflowTraceEvent:
    """Structured trace emitted by one workflow step."""

    step: str
    source: str
    tool_name: str
    tool_args: dict[str, Any]
    tool_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "source": self.source,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_output": self.tool_output,
        }


@dataclass(slots=True)
class DecompositionStepResult:
    """Resolved decomposition decision for one user query."""

    decomposable: bool
    sub_queries: list[str]
    trace_event: WorkflowTraceEvent


@dataclass(slots=True)
class RoutingStepResult:
    """Resolved route for one sub-query."""

    route: str
    trace_event: WorkflowTraceEvent


@dataclass(slots=True)
class RewriteStepResult:
    """Resolved retrieval query after optional rewrite."""

    original_query: str
    rewritten_query: str
    trace_event: WorkflowTraceEvent | None = None


@dataclass(slots=True)
class RetrievalStepResult:
    """Retrieved documents for one routed query."""

    route: str
    query: str
    documents: list[Document]
    trace_event: WorkflowTraceEvent


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
