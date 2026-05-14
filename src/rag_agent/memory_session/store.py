"""Persistence layer for thread-based conversational memory."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable
import json
import logging
import os
import uuid

from rag_agent.memory_session.config import MemorySettings
from rag_agent.memory_session.locks import locked_path
from rag_agent.memory_session.models import (
    THREAD_DOCUMENT_VERSION,
    ConversationTurn,
    ThreadDocument,
    ThreadReference,
    migrate_legacy_document,
    utc_now_iso,
)

if TYPE_CHECKING:
    from rag_agent.memory_session.session import ManagedThread

logger = logging.getLogger(__name__)

DocumentUpdater = Callable[[ThreadDocument], None]


class ThreadStore:
    """Persistence layer for thread documents."""

    def __init__(self, settings: MemorySettings | None = None):
        self.settings = settings or MemorySettings()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.settings.session_dir.mkdir(parents=True, exist_ok=True)
        self.settings.current_session_file.parent.mkdir(parents=True, exist_ok=True)

    def create_thread(
        self,
        thread_id: str | None = None,
        max_turns: int | None = None,
    ) -> "ManagedThread":
        from rag_agent.memory_session.session import ManagedThread

        thread_id = thread_id or uuid.uuid4().hex[:8]
        document = ThreadDocument.new(
            thread_id=thread_id,
            max_turns=max_turns or self.settings.max_turns,
        )
        with self._thread_lock(thread_id):
            path = self.settings.get_session_path(thread_id)
            if path.exists():
                raise FileExistsError(f"Thread already exists: {thread_id}")
            self._write_document(path, document)
        return ManagedThread(store=self, document=document)

    def open_thread(
        self,
        thread_id: str,
        create: bool = False,
        read_only: bool = False,
    ) -> "ManagedThread":
        from rag_agent.memory_session.session import ManagedThread

        document = self.load_thread_document(thread_id=thread_id, create=create)
        return ManagedThread(store=self, document=document, read_only=read_only)

    def load_thread_document(
        self,
        thread_id: str,
        create: bool = False,
    ) -> ThreadDocument:
        with self._thread_lock(thread_id):
            path = self.settings.get_session_path(thread_id)
            if not path.exists():
                if not create:
                    raise FileNotFoundError(f"Thread not found: {thread_id}")
                document = ThreadDocument.new(thread_id, self.settings.max_turns)
                self._write_document(path, document)
                return document

            return self._read_document(path=path, thread_id=thread_id)

    def update_thread(
        self,
        thread_id: str,
        updater: DocumentUpdater,
        create: bool = False,
    ) -> ThreadDocument:
        with self._thread_lock(thread_id):
            path = self.settings.get_session_path(thread_id)
            if not path.exists():
                if not create:
                    raise FileNotFoundError(f"Thread not found: {thread_id}")
                document = ThreadDocument.new(thread_id, self.settings.max_turns)
            else:
                document = self._read_document(path=path, thread_id=thread_id)

            updater(document)
            self._trim_turns(document)
            self._sync_thread_title(document)
            document.updated_at = utc_now_iso()
            self._write_document(path, document)
            return document

    def list_thread_ids(self) -> list[str]:
        return sorted(path.stem for path in self.settings.session_dir.glob("*.json"))

    def delete_thread(self, thread_id: str) -> None:
        with self._thread_lock(thread_id):
            path = self.settings.get_session_path(thread_id)
            if not path.exists():
                raise FileNotFoundError(f"Thread not found: {thread_id}")
            path.unlink(missing_ok=True)

        current_thread_id = self.read_current_thread_id()
        if current_thread_id == thread_id:
            remaining = self.list_thread_ids()
            if remaining:
                self.write_current_thread_id(remaining[0])
            else:
                with self._pointer_lock():
                    self.settings.current_session_file.unlink(missing_ok=True)

    def read_current_thread_id(self) -> str | None:
        with self._pointer_lock():
            if not self.settings.current_session_file.exists():
                return None
            value = self.settings.current_session_file.read_text(encoding="utf-8").strip()
            return value or None

    def write_current_thread_id(self, thread_id: str) -> None:
        with self._pointer_lock():
            self._write_text_atomic(self.settings.current_session_file, thread_id)

    def build_context(
        self,
        thread_id: str,
        include_references: bool = True,
        include_reference_turns: bool = False,
    ) -> str:
        primary = self.load_thread_document(thread_id=thread_id)
        sections = [
            self._render_thread_section(
                title=f"Current Thread: {primary.thread_id}",
                document=primary,
                include_profile=True,
                include_summary=bool(primary.summary),
                recent_turn_limit=None,
            )
        ]

        if include_references:
            for reference in self._sorted_references(primary.references):
                with suppress(FileNotFoundError):
                    target = self.load_thread_document(reference.thread_id)
                    sections.append(
                        self._render_thread_section(
                            title=f"Referenced Thread: {reference.display_name()} ({reference.thread_id})",
                            document=target,
                            include_profile=reference.include_profile,
                            include_summary=reference.include_summary,
                            recent_turn_limit=(
                                self._effective_reference_turn_limit(reference)
                                if include_reference_turns
                                else 0
                            ),
                        )
                    )

        return "\n\n".join(section for section in sections if section).strip()

    def attach_reference(
        self,
        source_thread_id: str,
        target_thread_id: str,
        *,
        alias: str | None = None,
        priority: int | None = None,
        include_profile: bool = True,
        include_summary: bool = True,
        recent_turns_limit: int | None = None,
    ) -> ThreadDocument:
        if source_thread_id == target_thread_id:
            raise ValueError("A thread cannot reference itself.")
        if self._would_create_cycle(source_thread_id, target_thread_id):
            raise ValueError(
                f"Attaching {target_thread_id} to {source_thread_id} would create a cycle."
            )
        self.load_thread_document(target_thread_id)

        def updater(document: ThreadDocument) -> None:
            existing = {
                reference.thread_id: reference for reference in document.references
            }
            if target_thread_id not in existing and (
                len(document.references) >= self.settings.max_references_per_thread
            ):
                raise ValueError("Maximum number of thread references reached.")

            next_priority = priority
            if next_priority is None:
                next_priority = max(
                    [reference.priority for reference in document.references],
                    default=-1,
                ) + 1

            existing[target_thread_id] = ThreadReference(
                thread_id=target_thread_id,
                alias=alias,
                priority=next_priority,
                include_profile=include_profile,
                include_summary=include_summary,
                recent_turns_limit=recent_turns_limit
                if recent_turns_limit is not None
                else self.settings.reference_recent_turns_limit,
            )
            document.references = self._sorted_references(existing.values())

        return self.update_thread(source_thread_id, updater=updater)

    def detach_reference(self, source_thread_id: str, target_thread_id: str) -> ThreadDocument:
        def updater(document: ThreadDocument) -> None:
            document.references = [
                reference
                for reference in document.references
                if reference.thread_id != target_thread_id
            ]

        return self.update_thread(source_thread_id, updater=updater)

    def list_references(self, thread_id: str) -> list[ThreadReference]:
        document = self.load_thread_document(thread_id)
        return list(self._sorted_references(document.references))

    def prune_expired_threads(self, exclude_thread_ids: set[str] | None = None) -> list[str]:
        exclude_thread_ids = exclude_thread_ids or set()
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.retention_days)
        removed: list[str] = []

        for thread_id in self.list_thread_ids():
            if thread_id in exclude_thread_ids:
                continue
            path = self.settings.get_session_path(thread_id)
            with suppress(Exception):
                document = self.load_thread_document(thread_id)
                updated_at = datetime.fromisoformat(document.updated_at)
                if updated_at < cutoff:
                    path.unlink(missing_ok=True)
                    removed.append(thread_id)
        return removed

    def _thread_lock(self, thread_id: str):
        return locked_path(
            self.settings.get_lock_path(thread_id),
            timeout_seconds=self.settings.lock_timeout_seconds,
        )

    def _pointer_lock(self):
        return locked_path(
            self.settings.get_current_session_lock_path(),
            timeout_seconds=self.settings.lock_timeout_seconds,
        )

    def _read_document(self, path: Path, thread_id: str) -> ThreadDocument:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt memory file detected for thread %s: %s", thread_id, exc)
            if self.settings.backup_corrupt_files:
                self._backup_corrupt_file(path)
                document = ThreadDocument.new(thread_id, self.settings.max_turns)
                self._write_document(path, document)
                return document
            raise

        migrated = False
        if "version" not in payload:
            document = migrate_legacy_document(
                data=payload,
                thread_id=thread_id,
                default_max_turns=self.settings.max_turns,
            )
            migrated = True
        else:
            document = ThreadDocument.from_dict(payload, default_thread_id=thread_id)

        document.version = THREAD_DOCUMENT_VERSION
        document.thread_id = thread_id
        if migrated or payload.get("version") != THREAD_DOCUMENT_VERSION:
            self._write_document(path, document)
        return document

    def _write_document(self, path: Path, document: ThreadDocument) -> None:
        self._write_json_atomic(path, document.to_dict())

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def _write_text_atomic(self, path: Path, value: str) -> None:
        temp_path = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    def _backup_corrupt_file(self, path: Path) -> None:
        backup_path = path.with_name(
            f"{path.stem}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}{path.suffix}"
        )
        os.replace(path, backup_path)

    def _trim_turns(self, document: ThreadDocument) -> None:
        while len(document.turns) > document.max_turns:
            removable_index = next(
                (
                    index
                    for index, turn in enumerate(document.turns)
                    if turn.state in {"completed", "failed"}
                ),
                None,
            )
            if removable_index is None:
                break
            document.turns.pop(removable_index)

    def _sync_thread_title(self, document: ThreadDocument) -> None:
        for turn in document.turns:
            if turn.user_message and turn.user_message.content:
                document.title = self._build_thread_title(turn.user_message.content)
                return
        document.title = "New Session"

    def _build_thread_title(self, content: str) -> str:
        normalized = " ".join(content.split())
        if not normalized:
            return "New Session"
        return normalized[:40]

    def _sorted_references(self, references) -> list[ThreadReference]:
        return sorted(
            list(references),
            key=lambda reference: (reference.priority, reference.created_at, reference.thread_id),
        )

    def _effective_reference_turn_limit(self, reference: ThreadReference) -> int:
        per_reference = max(reference.recent_turns_limit, 0)
        global_limit = max(self.settings.reference_recent_turns_limit, 0)
        if per_reference == 0 or global_limit == 0:
            return 0
        return min(per_reference, global_limit)

    def _would_create_cycle(self, source_thread_id: str, target_thread_id: str) -> bool:
        visited: set[str] = set()
        stack = [target_thread_id]

        while stack:
            current = stack.pop()
            if current == source_thread_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            with suppress(FileNotFoundError):
                document = self.load_thread_document(current)
                stack.extend(reference.thread_id for reference in document.references)
        return False

    def _render_thread_section(
        self,
        *,
        title: str,
        document: ThreadDocument,
        include_profile: bool,
        include_summary: bool,
        recent_turn_limit: int | None,
    ) -> str:
        lines = [title]
        if include_summary and document.summary:
            lines.append(f"Summary: {self._truncate_summary(document.summary)}")
        if include_profile and document.profile:
            lines.append("Profile:")
            for key, value in sorted(document.profile.items()):
                lines.append(f"- {key}: {value}")

        turns = self._select_turns(document.turns, limit=recent_turn_limit)
        if turns:
            lines.append("Turns:")
            for turn in turns:
                if turn.user_message:
                    lines.append(f"[user] {turn.user_message.content}")
                if turn.assistant_message:
                    lines.append(f"[assistant] {turn.assistant_message.content}")
                if turn.state == "failed" and turn.error:
                    lines.append(f"[turn-error] {turn.error}")
        return "\n".join(lines)

    def _select_turns(
        self,
        turns: list[ConversationTurn],
        limit: int | None,
    ) -> list[ConversationTurn]:
        if limit == 0:
            return []
        if limit is None:
            return list(turns)
        return list(turns[-limit:])

    def _truncate_summary(self, summary: str) -> str:
        limit = max(self.settings.reference_summary_char_limit, 0)
        if limit == 0 or len(summary) <= limit:
            return summary
        if limit <= 3:
            return summary[:limit]
        return f"{summary[: limit - 3]}..."
