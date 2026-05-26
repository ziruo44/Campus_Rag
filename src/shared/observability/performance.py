"""Performance tracing helpers for stage-1 observability."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import json
import logging
from time import perf_counter
from typing import Any, Iterable
import uuid

from utils.text import truncate_text

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PerformanceTrace:
    """Aggregated performance metrics for one chat turn."""

    trace_id: str
    channel: str
    started_at: float
    thread_id: str | None = None
    query_preview: str = ""
    status: str = "in_progress"
    error: str | None = None
    total_duration_ms: float | None = None
    first_token_latency_ms: float | None = None
    llm_call_count: int = 0
    tool_call_count: int = 0
    retrieval_result_count: int = 0
    prompt_input_char_count: int = 0
    message_count: int = 0
    thread_turn_count: int = 0
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    retrieval_results_by_tool: dict[str, int] = field(default_factory=dict)

    def add_stage_duration(self, stage: str, elapsed_ms: float) -> None:
        self.stage_durations_ms[stage] = self.stage_durations_ms.get(stage, 0.0) + elapsed_ms
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "channel": self.channel,
            "thread_id": self.thread_id,
            "query_preview": self.query_preview,
            "status": self.status,
            "error": self.error,
            "total_duration_ms": _round_or_none(self.total_duration_ms),
            "first_token_latency_ms": _round_or_none(self.first_token_latency_ms),
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "retrieval_result_count": self.retrieval_result_count,
            "retrieval_results_by_tool": dict(sorted(self.retrieval_results_by_tool.items())),
            "prompt_input_char_count": self.prompt_input_char_count,
            "message_count": self.message_count,
            "thread_turn_count": self.thread_turn_count,
            "stage_durations_ms": {
                key: round(value, 2)
                for key, value in sorted(self.stage_durations_ms.items())
            },
            "stage_counts": dict(sorted(self.stage_counts.items())),
        }
        return payload


@dataclass(slots=True)
class PerformanceTraceHandle:
    """Handle for the active performance trace context."""

    trace: PerformanceTrace
    token: Token


_CURRENT_TRACE: ContextVar[PerformanceTrace | None] = ContextVar(
    "rag_agent_performance_trace",
    default=None,
)


def start_trace(
    channel: str,
    *,
    thread_id: str | None = None,
    query: str = "",
) -> PerformanceTraceHandle:
    """Start a new per-request trace in the current context."""
    trace = PerformanceTrace(
        trace_id=uuid.uuid4().hex[:12],
        channel=channel,
        started_at=perf_counter(),
        thread_id=thread_id,
        query_preview=truncate_text(query, limit=120),
    )
    token = _CURRENT_TRACE.set(trace)
    return PerformanceTraceHandle(trace=trace, token=token)


def finish_trace(
    handle: PerformanceTraceHandle | None,
    *,
    status: str,
    error: str | None = None,
) -> PerformanceTrace | None:
    """Finalize and emit the trace payload."""
    if handle is None:
        return None

    trace = handle.trace
    trace.status = status
    trace.error = truncate_text(error or "", limit=240) or None
    trace.total_duration_ms = (perf_counter() - trace.started_at) * 1000.0
    logger.info("performance_trace %s", json.dumps(trace.to_payload(), ensure_ascii=False))
    _CURRENT_TRACE.reset(handle.token)
    return trace


def get_current_trace() -> PerformanceTrace | None:
    """Return the active trace, if any."""
    return _CURRENT_TRACE.get()


def set_thread_id(thread_id: str) -> None:
    """Update the active trace thread id."""
    trace = get_current_trace()
    if trace is not None:
        trace.thread_id = thread_id


def record_thread_turn_count(turn_count: int) -> None:
    """Record the number of turns currently in the active thread."""
    trace = get_current_trace()
    if trace is not None:
        trace.thread_turn_count = turn_count


def record_prompt_messages(messages: Iterable[Any]) -> None:
    """Record prompt message count and concatenated content size."""
    trace = get_current_trace()
    if trace is None:
        return

    normalized = list(messages)
    trace.message_count = len(normalized)
    trace.prompt_input_char_count = sum(_content_length(getattr(msg, "content", "")) for msg in normalized)


def increment_llm_calls(count: int = 1) -> None:
    """Increment the active LLM-call counter."""
    trace = get_current_trace()
    if trace is not None:
        trace.llm_call_count += count


def increment_tool_calls(count: int = 1) -> None:
    """Increment the active tool-call counter."""
    trace = get_current_trace()
    if trace is not None:
        trace.tool_call_count += count


def record_retrieval_results(tool_name: str, doc_count: int) -> None:
    """Accumulate retrieved document counts."""
    trace = get_current_trace()
    if trace is None:
        return

    trace.retrieval_result_count += doc_count
    trace.retrieval_results_by_tool[tool_name] = (
        trace.retrieval_results_by_tool.get(tool_name, 0) + doc_count
    )


def mark_first_token() -> None:
    """Capture first-token latency once for the active trace."""
    trace = get_current_trace()
    if trace is not None and trace.first_token_latency_ms is None:
        trace.first_token_latency_ms = (perf_counter() - trace.started_at) * 1000.0


@contextmanager
def measure_stage(stage: str):
    """Measure one named stage and aggregate it into the active trace."""
    trace = get_current_trace()
    if trace is None:
        yield
        return

    started_at = perf_counter()
    try:
        yield
    finally:
        trace.add_stage_duration(stage, (perf_counter() - started_at) * 1000.0)


def _content_length(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(item)
                continue
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
                continue
            total += len(str(getattr(item, "text", "")))
        return total
    return len(str(content))


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 2)


def _truncate(value: str, *, limit: int) -> str:
    """Backward-compatible wrapper for legacy imports."""
    return truncate_text(value, limit=limit)
