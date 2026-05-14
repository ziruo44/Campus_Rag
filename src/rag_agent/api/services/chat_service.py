"""Chat service shared by the CLI and FastAPI routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rag_agent.api.services.agent_runtime import AgentRuntime, RuntimeUnavailableError
from rag_agent.memory_session.session import ManagedThread, SessionManager


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

    def chat(self, message: str, thread_id: str | None = None) -> ChatTurnResult:
        """Open or create a thread and run a single chat turn."""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self.session_manager_factory() as manager:
            thread = self._resolve_thread(manager, normalized_thread_id)
            return self.invoke_thread(thread, message)

    def invoke_thread(self, thread: ManagedThread, message: str) -> ChatTurnResult:
        """Run a single chat turn on an explicit thread."""
        turn_id = thread.append_user_turn(message)
        try:
            result = self.runtime.invoke(thread)
            thread.complete_turn(turn_id, result.answer)
            return ChatTurnResult(
                thread_id=thread.thread_id,
                answer=result.answer,
                messages=result.messages,
            )
        except Exception as exc:
            thread.fail_turn(turn_id, str(exc))
            if isinstance(exc, RuntimeUnavailableError):
                raise
            if isinstance(exc, ChatExecutionError):
                raise
            raise ChatExecutionError("Failed to execute chat request.") from exc

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
