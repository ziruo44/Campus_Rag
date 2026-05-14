"""High-level session and thread management APIs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import logging
import uuid

from rag_agent.memory_session.config import MemorySettings
from rag_agent.memory_session.models import (
    ConversationTurn,
    ThreadDocument,
    ThreadMessage,
    ThreadReference,
    utc_now_iso,
)
from rag_agent.memory_session.store import ThreadStore

logger = logging.getLogger(__name__)


class ManagedThread:
    """High-level thread API with autosave semantics."""

    def __init__(
        self,
        *,
        store: ThreadStore,
        document: ThreadDocument,
        read_only: bool = False,
    ):
        self.store = store
        self._document = document
        self.read_only = read_only

    @property
    def thread_id(self) -> str:
        return self._document.thread_id

    @property
    def summary(self) -> str:
        return self._document.summary

    @property
    def title(self) -> str:
        return self._document.title

    @property
    def turns(self) -> list[ConversationTurn]:
        return deepcopy(self._document.turns)

    def refresh(self) -> "ManagedThread":
        self._document = self.store.load_thread_document(self.thread_id)
        return self

    def flush(self) -> None:
        self.refresh()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ManagedThread":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def to_dict(self) -> dict[str, Any]:
        return self._document.to_dict()

    def build_context(
        self,
        include_references: bool = True,
        include_reference_turns: bool = False,
    ) -> str:
        return self.store.build_context(
            thread_id=self.thread_id,
            include_references=include_references,
            include_reference_turns=include_reference_turns,
        )

    def get_conversation_string(self) -> str:
        return self.build_context(include_references=True, include_reference_turns=False)

    def append_user_turn(self, content: str) -> str:
        self._ensure_writable()
        turn = ConversationTurn.start(content)

        def updater(document: ThreadDocument) -> None:
            document.turns.append(turn)

        self._document = self.store.update_thread(self.thread_id, updater=updater)
        return turn.turn_id

    def complete_turn(self, turn_id: str, assistant_content: str) -> None:
        self._ensure_writable()

        def updater(document: ThreadDocument) -> None:
            turn = self._find_turn(document, turn_id)
            turn.assistant_message = ThreadMessage.create(
                role="assistant",
                content=assistant_content,
            )
            turn.state = "completed"
            turn.updated_at = turn.assistant_message.timestamp
            turn.error = None

        self._document = self.store.update_thread(self.thread_id, updater=updater)

    def fail_turn(self, turn_id: str, error: str) -> None:
        self._ensure_writable()

        def updater(document: ThreadDocument) -> None:
            turn = self._find_turn(document, turn_id)
            turn.state = "failed"
            turn.error = error
            turn.updated_at = utc_now_iso()

        self._document = self.store.update_thread(self.thread_id, updater=updater)

    def complete_latest_pending_turn(self, assistant_content: str) -> None:
        self._ensure_writable()

        def updater(document: ThreadDocument) -> None:
            pending_turn = next(
                (turn for turn in reversed(document.turns) if turn.state == "pending"),
                None,
            )
            if pending_turn is None:
                timestamp = utc_now_iso()
                standalone = ConversationTurn(
                    turn_id=f"turn_{uuid.uuid4().hex[:12]}",
                    user_message=None,
                    assistant_message=ThreadMessage(
                        role="assistant",
                        content=assistant_content,
                        timestamp=timestamp,
                    ),
                    state="completed",
                    started_at=timestamp,
                    updated_at=timestamp,
                )
                document.turns.append(standalone)
                return

            pending_turn.assistant_message = ThreadMessage.create(
                role="assistant",
                content=assistant_content,
            )
            pending_turn.state = "completed"
            pending_turn.updated_at = pending_turn.assistant_message.timestamp
            pending_turn.error = None

        self._document = self.store.update_thread(self.thread_id, updater=updater)

    def add_user(self, content: str) -> str:
        return self.append_user_turn(content)

    def add_assistant(self, content: str) -> None:
        self.complete_latest_pending_turn(content)

    def set_profile(self, key: str, value: Any) -> None:
        self._ensure_writable()

        def updater(document: ThreadDocument) -> None:
            document.profile[key] = value

        self._document = self.store.update_thread(self.thread_id, updater=updater)

    def get_profile(self, key: str) -> Any:
        self.refresh()
        return self._document.profile.get(key)

    def get_all_profile(self) -> dict[str, Any]:
        self.refresh()
        return deepcopy(self._document.profile)

    def set_summary(self, summary: str) -> None:
        self._ensure_writable()

        def updater(document: ThreadDocument) -> None:
            document.summary = summary

        self._document = self.store.update_thread(self.thread_id, updater=updater)

    def attach_reference(
        self,
        target_thread_id: str,
        *,
        alias: str | None = None,
        priority: int | None = None,
        include_profile: bool = True,
        include_summary: bool = True,
        recent_turns_limit: int | None = None,
    ) -> None:
        self._ensure_writable()
        self._document = self.store.attach_reference(
            self.thread_id,
            target_thread_id,
            alias=alias,
            priority=priority,
            include_profile=include_profile,
            include_summary=include_summary,
            recent_turns_limit=recent_turns_limit,
        )

    def detach_reference(self, target_thread_id: str) -> None:
        self._ensure_writable()
        self._document = self.store.detach_reference(self.thread_id, target_thread_id)

    def list_references(self) -> list[ThreadReference]:
        self.refresh()
        return list(self._document.references)

    def delete_turn(self, turn_id: str) -> None:
        self._ensure_writable()

        def updater(document: ThreadDocument) -> None:
            remaining_turns = [
                turn for turn in document.turns if turn.turn_id != turn_id
            ]
            if len(remaining_turns) == len(document.turns):
                raise ValueError(f"Turn not found: {turn_id}")
            document.turns = remaining_turns

        self._document = self.store.update_thread(self.thread_id, updater=updater)

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise PermissionError(f"Thread {self.thread_id} is read-only.")

    def _find_turn(self, document: ThreadDocument, turn_id: str) -> ConversationTurn:
        turn = next((item for item in document.turns if item.turn_id == turn_id), None)
        if turn is None:
            raise ValueError(f"Turn not found: {turn_id}")
        return turn


class SessionManager:
    """Compatibility wrapper around explicit thread management."""

    def __init__(self, settings: MemorySettings | None = None):
        self.settings = settings or MemorySettings()
        self.store = ThreadStore(self.settings)
        self._current_thread: ManagedThread | None = None
        self._load_or_create_current_thread()

    @property
    def current_thread_id(self) -> str:
        return self.thread.thread_id

    @property
    def current_session_id(self) -> str:
        return self.current_thread_id

    @property
    def thread(self) -> ManagedThread:
        if self._current_thread is None:
            self._load_or_create_current_thread()
        return self._current_thread

    @property
    def memory(self) -> ManagedThread:
        return self.thread

    def save(self) -> None:
        self.thread.flush()

    def flush(self) -> None:
        self.save()

    def close(self) -> None:
        if self._current_thread is not None:
            self._current_thread.close()

    def __enter__(self) -> "SessionManager":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def create_new_thread(
        self,
        *,
        switch: bool = True,
        max_turns: int | None = None,
    ) -> ManagedThread:
        thread = self.store.create_thread(max_turns=max_turns)
        if switch:
            self._switch_to_thread(thread)
        return thread

    def open_thread(
        self,
        thread_id: str,
        *,
        create: bool = False,
        read_only: bool = False,
    ) -> ManagedThread:
        return self.store.open_thread(thread_id, create=create, read_only=read_only)

    def switch_thread(self, thread_id: str, *, create: bool = False) -> ManagedThread:
        thread = self.store.open_thread(thread_id, create=create)
        self._switch_to_thread(thread)
        return thread

    def switch_session(self, session_id: str) -> ManagedThread:
        return self.switch_thread(session_id)

    def list_threads(self) -> list[str]:
        return self.store.list_thread_ids()

    def list_sessions(self) -> list[str]:
        return self.list_threads()

    def build_context(
        self,
        thread_id: str | None = None,
        *,
        include_references: bool = True,
        include_reference_turns: bool = False,
    ) -> str:
        return self.store.build_context(
            thread_id=thread_id or self.current_thread_id,
            include_references=include_references,
            include_reference_turns=include_reference_turns,
        )

    def attach_reference(self, source_thread_id: str, target_thread_id: str, **kwargs: Any) -> None:
        self.store.attach_reference(source_thread_id, target_thread_id, **kwargs)
        if self._current_thread and self._current_thread.thread_id == source_thread_id:
            self._current_thread.refresh()

    def detach_reference(self, source_thread_id: str, target_thread_id: str) -> None:
        self.store.detach_reference(source_thread_id, target_thread_id)
        if self._current_thread and self._current_thread.thread_id == source_thread_id:
            self._current_thread.refresh()

    def list_references(self, thread_id: str | None = None) -> list[ThreadReference]:
        return self.store.list_references(thread_id or self.current_thread_id)

    def delete_thread(self, thread_id: str) -> None:
        self.store.delete_thread(thread_id)
        if self._current_thread and self._current_thread.thread_id == thread_id:
            self._current_thread = None

    def _load_or_create_current_thread(self) -> None:
        thread_id = self.store.read_current_thread_id()
        if thread_id is not None:
            try:
                self._current_thread = self.store.open_thread(thread_id)
                return
            except FileNotFoundError:
                logger.warning(
                    "Current thread pointer %s is stale. Creating a new thread.",
                    thread_id,
                )

        self._current_thread = self.store.create_thread()
        self.store.write_current_thread_id(self._current_thread.thread_id)

    def _switch_to_thread(self, thread: ManagedThread) -> None:
        if self._current_thread is not None:
            self._current_thread.flush()
        self._current_thread = thread
        self.store.write_current_thread_id(thread.thread_id)


Message = ThreadMessage
SimpleMemory = ManagedThread
ThreadStore = ThreadStore
