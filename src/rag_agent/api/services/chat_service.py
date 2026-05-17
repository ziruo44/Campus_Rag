"""Chat service shared by the CLI and FastAPI routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import logging
from typing import Any, Callable

import httpx
from openai import APIConnectionError, APITimeoutError

from rag_agent.api.schemas import ChatStreamChunk
from rag_agent.api.services.agent_runtime import AgentRuntime, RuntimeUnavailableError
from rag_agent.memory_session.session import ManagedThread, SessionManager
from rag_agent.observability.performance import (
    finish_trace,
    mark_first_token,
    measure_stage,
    record_thread_turn_count,
    set_thread_id,
    start_trace,
)

logger = logging.getLogger(__name__)


class ThreadNotFoundError(FileNotFoundError):
    """Raised when a requested thread does not exist."""


class TurnNotFoundError(FileNotFoundError):
    """Raised when a requested turn does not exist."""


class ChatExecutionError(RuntimeError):
    """Raised when a single chat turn fails."""


@dataclass(slots=True)
class ChatTurnResult:
    """Structured chat result for one completed turn."""

    thread_id: str
    answer: str
    messages: list[Any]


class ChatService:
    """Coordinate thread resolution and agent invocation."""

    def __init__(
        self,
        runtime: AgentRuntime,
        session_manager_factory: Callable[[], SessionManager] | None = None,
    ) -> None:
        self.runtime = runtime
        self.session_manager_factory = session_manager_factory or SessionManager

    def chat(
        self,
        message: str,
        thread_id: str | None = None,
        precise_mode: bool = False,
    ) -> ChatTurnResult:
        """Open or create a thread and run a single chat turn."""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self.session_manager_factory() as manager:
            thread = self._resolve_thread(manager, normalized_thread_id)
            return self.invoke_thread(thread, message, precise_mode=precise_mode)

    def stream_chat(
        self,
        message: str,
        thread_id: str | None = None,
        precise_mode: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        """Open or create a thread and stream one chat turn."""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        manager = self.session_manager_factory()
        trace_handle = start_trace(
            "chat.stream",
            thread_id=normalized_thread_id,
            query=message,
        )
        turn_id: str | None = None

        try:
            thread = self._resolve_thread(manager, normalized_thread_id)
            set_thread_id(thread.thread_id)
            self.runtime.ensure_initialized()
            with measure_stage("memory.append_user_turn"):
                turn_id = thread.append_user_turn(message)
            record_thread_turn_count(len(thread.turns))
            stream = self.runtime.stream(
                thread,
                retrieval_context_strategy=self._resolve_context_strategy(precise_mode),
            )
        except Exception as exc:
            if turn_id is not None:
                resolved_error = self._normalize_execution_exception(exc)
                with measure_stage("memory.fail_turn"):
                    thread.fail_turn(turn_id, str(resolved_error))
                finish_trace(trace_handle, status="failed", error=str(resolved_error))
                manager.close()
                raise resolved_error from exc
            finish_trace(trace_handle, status="failed", error=str(exc))
            manager.close()
            raise

        async def iterator() -> AsyncIterator[ChatStreamChunk]:
            full_answer = ""
            stream_started = False

            try:
                stream_started = True
                yield ChatStreamChunk(
                    content="",
                    is_final=False,
                    thread_id=thread.thread_id,
                    turn_id=turn_id,
                )

                async for chunk_text in stream:
                    full_answer += chunk_text
                    if chunk_text:
                        mark_first_token()
                    yield ChatStreamChunk(
                        content=chunk_text,
                        is_final=False,
                        thread_id=thread.thread_id,
                        turn_id=turn_id,
                    )

                with measure_stage("memory.complete_turn"):
                    thread.complete_turn(turn_id, full_answer)
                finish_trace(trace_handle, status="completed")
                yield ChatStreamChunk(
                    content="",
                    is_final=True,
                    thread_id=thread.thread_id,
                    turn_id=turn_id,
                )
            except Exception as exc:
                resolved_error = self._normalize_execution_exception(exc)
                with measure_stage("memory.fail_turn"):
                    thread.fail_turn(turn_id, str(resolved_error))
                finish_trace(trace_handle, status="failed", error=str(resolved_error))
                if stream_started:
                    yield ChatStreamChunk(
                        content="",
                        is_final=True,
                        thread_id=thread.thread_id,
                        turn_id=turn_id,
                        error=str(resolved_error),
                    )
                    return
                raise resolved_error from exc
            finally:
                manager.close()

        return iterator()

    def invoke_thread(
        self,
        thread: ManagedThread,
        message: str,
        *,
        precise_mode: bool = False,
    ) -> ChatTurnResult:
        """Run a single chat turn on an explicit thread."""
        trace_handle = start_trace(
            "chat.invoke",
            thread_id=thread.thread_id,
            query=message,
        )
        with measure_stage("memory.append_user_turn"):
            turn_id = thread.append_user_turn(message)
        record_thread_turn_count(len(thread.turns))
        try:
            result = self.runtime.invoke_with_strategy(
                thread,
                retrieval_context_strategy=self._resolve_context_strategy(precise_mode),
            )
            with measure_stage("memory.complete_turn"):
                thread.complete_turn(turn_id, result.answer)
            finish_trace(trace_handle, status="completed")
            return ChatTurnResult(
                thread_id=thread.thread_id,
                answer=result.answer,
                messages=result.messages,
            )
        except Exception as exc:
            resolved_error = self._normalize_execution_exception(exc)
            with measure_stage("memory.fail_turn"):
                thread.fail_turn(turn_id, str(resolved_error))
            finish_trace(trace_handle, status="failed", error=str(resolved_error))
            raise resolved_error from exc

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        """Fetch thread state by ID."""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        if normalized_thread_id is None:
            raise ThreadNotFoundError("Thread not found.")
        with self.session_manager_factory() as manager:
            try:
                thread = manager.open_thread(normalized_thread_id)
            except FileNotFoundError as exc:
                raise ThreadNotFoundError(
                    f"Thread not found: {normalized_thread_id}"
                ) from exc
            return thread.to_dict()

    def list_threads(self) -> list[dict[str, Any]]:
        """Return summarized thread metadata for history listings."""
        with self.session_manager_factory() as manager:
            summaries: list[dict[str, Any]] = []
            for thread_id in manager.list_threads():
                thread = manager.open_thread(thread_id, read_only=True)
                payload = thread.to_dict()
                turns = payload.get("turns", [])
                summaries.append(
                    {
                        "thread_id": payload["thread_id"],
                        "title": payload.get("title", "") or "New Session",
                        "summary": payload.get("summary", ""),
                        "updated_at": payload.get("updated_at", ""),
                        "turn_count": len(turns),
                        "preview": self._build_preview(payload),
                    }
                )

            return sorted(
                summaries,
                key=lambda item: item["updated_at"],
                reverse=True,
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete a persisted thread JSON document."""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        if normalized_thread_id is None:
            raise ThreadNotFoundError("Thread not found.")

        with self.session_manager_factory() as manager:
            try:
                manager.delete_thread(normalized_thread_id)
            except FileNotFoundError as exc:
                raise ThreadNotFoundError(
                    f"Thread not found: {normalized_thread_id}"
                ) from exc

    def delete_turn(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        """Delete one user turn and the paired assistant answer."""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        normalized_turn_id = self._normalize_turn_id(turn_id)
        if normalized_thread_id is None:
            raise ThreadNotFoundError("Thread not found.")
        if normalized_turn_id is None:
            raise TurnNotFoundError("Turn not found.")

        with self.session_manager_factory() as manager:
            try:
                thread = manager.open_thread(normalized_thread_id)
            except FileNotFoundError as exc:
                raise ThreadNotFoundError(
                    f"Thread not found: {normalized_thread_id}"
                ) from exc

            try:
                thread.delete_turn(normalized_turn_id)
            except ValueError as exc:
                raise TurnNotFoundError(f"Turn not found: {normalized_turn_id}") from exc

            return thread.to_dict()

    def _resolve_thread(
        self,
        manager: SessionManager,
        thread_id: str | None,
    ) -> ManagedThread:
        if thread_id is None:
            return manager.create_new_thread(switch=False)
        try:
            return manager.open_thread(thread_id)
        except FileNotFoundError as exc:
            raise ThreadNotFoundError(f"Thread not found: {thread_id}") from exc

    def _normalize_thread_id(self, thread_id: str | None) -> str | None:
        """Normalize optional thread IDs from external clients."""
        if thread_id is None:
            return None
        normalized = thread_id.strip()
        return normalized or None

    def _normalize_turn_id(self, turn_id: str | None) -> str | None:
        if turn_id is None:
            return None
        normalized = turn_id.strip()
        return normalized or None

    def _build_preview(self, payload: dict[str, Any]) -> str:
        """Build a compact preview string for a thread."""
        summary = str(payload.get("summary", "")).strip()
        if summary:
            return summary[:96]

        turns = list(payload.get("turns", []))
        for turn in reversed(turns):
            assistant = turn.get("assistant_message") or {}
            assistant_content = str(assistant.get("content", "")).strip()
            if assistant_content:
                return assistant_content.replace("\n", " ")[:96]

            user = turn.get("user_message") or {}
            user_content = str(user.get("content", "")).strip()
            if user_content:
                return user_content.replace("\n", " ")[:96]

        return "Empty session"

    def _describe_exception(self, exc: BaseException) -> str:
        """Return the deepest non-empty exception message for diagnostics."""
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

    def _normalize_execution_exception(self, exc: Exception) -> Exception:
        """Normalize runtime exceptions into API-facing chat errors."""
        if isinstance(exc, (RuntimeUnavailableError, ChatExecutionError)):
            return exc

        if isinstance(exc, (APIConnectionError, APITimeoutError, httpx.HTTPError)):
            detail = self._describe_exception(exc)
            logger.exception("Model provider request failed")
            return RuntimeUnavailableError(f"Model provider request failed: {detail}")

        return ChatExecutionError("Failed to execute chat request.")

    def _resolve_context_strategy(self, precise_mode: bool) -> str:
        return "passthrough" if precise_mode else "compressed"
