"""Data models for file-backed conversational memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from utils.time import new_id, utc_now_iso

THREAD_DOCUMENT_VERSION = 4


@dataclass(slots=True)
class ThreadMessage:
    """A single user or assistant message."""

    role: str
    content: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def create(cls, role: str, content: str) -> "ThreadMessage":
        return cls(role=role, content=content, timestamp=utc_now_iso())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadMessage":
        return cls(
            role=str(data.get("role", "")),
            content=str(data.get("content", "")),
            timestamp=str(data.get("timestamp") or utc_now_iso()),
        )


@dataclass(slots=True)
class ConversationTurn:
    """A complete or in-progress dialogue turn."""

    turn_id: str
    user_message: ThreadMessage | None = None
    assistant_message: ThreadMessage | None = None
    state: str = "pending"
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message.to_dict() if self.user_message else None,
            "assistant_message": (
                self.assistant_message.to_dict() if self.assistant_message else None
            ),
            "state": self.state,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "artifacts": self.artifacts,
        }

    @classmethod
    def start(cls, content: str, turn_id: str | None = None) -> "ConversationTurn":
        message = ThreadMessage.create(role="user", content=content)
        return cls(
            turn_id=turn_id or new_id("turn"),
            user_message=message,
            state="pending",
            started_at=message.timestamp,
            updated_at=message.timestamp,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        user_payload = data.get("user_message")
        assistant_payload = data.get("assistant_message")
        return cls(
            turn_id=str(data.get("turn_id") or new_id("turn")),
            user_message=ThreadMessage.from_dict(user_payload) if user_payload else None,
            assistant_message=(
                ThreadMessage.from_dict(assistant_payload) if assistant_payload else None
            ),
            state=str(data.get("state") or "pending"),
            started_at=str(data.get("started_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            error=data.get("error"),
            artifacts=dict(data.get("artifacts") or {}),
        )


@dataclass(slots=True)
class ThreadReference:
    """Read-only link to another thread."""

    thread_id: str
    alias: str | None = None
    priority: int = 0
    include_profile: bool = True
    include_summary: bool = True
    recent_turns_limit: int = 0
    created_at: str = field(default_factory=utc_now_iso)

    def display_name(self) -> str:
        return self.alias or self.thread_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "alias": self.alias,
            "priority": self.priority,
            "include_profile": self.include_profile,
            "include_summary": self.include_summary,
            "recent_turns_limit": self.recent_turns_limit,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadReference":
        return cls(
            thread_id=str(data.get("thread_id", "")),
            alias=data.get("alias"),
            priority=int(data.get("priority", 0)),
            include_profile=bool(data.get("include_profile", True)),
            include_summary=bool(data.get("include_summary", True)),
            recent_turns_limit=int(data.get("recent_turns_limit", 0)),
            created_at=str(data.get("created_at") or utc_now_iso()),
        )


@dataclass(slots=True)
class ThreadDocument:
    """Persistent thread document."""

    version: int
    thread_id: str
    title: str
    profile: dict[str, Any]
    turns: list[ConversationTurn]
    summary: str
    context_summary: str
    context_summary_updated_at: str
    context_compacted_turn_count: int
    notices: list[dict[str, Any]]
    references: list[ThreadReference]
    max_turns: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "thread_id": self.thread_id,
            "title": self.title,
            "profile": self.profile,
            "turns": [turn.to_dict() for turn in self.turns],
            "summary": self.summary,
            "context_summary": self.context_summary,
            "context_summary_updated_at": self.context_summary_updated_at,
            "context_compacted_turn_count": self.context_compacted_turn_count,
            "notices": list(self.notices),
            "references": [reference.to_dict() for reference in self.references],
            "max_turns": self.max_turns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def new(cls, thread_id: str, max_turns: int) -> "ThreadDocument":
        timestamp = utc_now_iso()
        return cls(
            version=THREAD_DOCUMENT_VERSION,
            thread_id=thread_id,
            title="New Session",
            profile={},
            turns=[],
            summary="",
            context_summary="",
            context_summary_updated_at=timestamp,
            context_compacted_turn_count=0,
            notices=[],
            references=[],
            max_turns=max_turns,
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_thread_id: str) -> "ThreadDocument":
        return cls(
            version=int(data.get("version", THREAD_DOCUMENT_VERSION)),
            thread_id=str(data.get("thread_id") or default_thread_id),
            title=str(data.get("title") or ""),
            profile=dict(data.get("profile") or {}),
            turns=[
                ConversationTurn.from_dict(item)
                for item in list(data.get("turns") or [])
            ],
            summary=str(data.get("summary") or ""),
            context_summary=str(data.get("context_summary") or ""),
            context_summary_updated_at=str(
                data.get("context_summary_updated_at") or data.get("updated_at") or utc_now_iso()
            ),
            context_compacted_turn_count=int(data.get("context_compacted_turn_count", 0)),
            notices=list(data.get("notices") or []),
            references=[
                ThreadReference.from_dict(item)
                for item in list(data.get("references") or [])
            ],
            max_turns=int(data.get("max_turns", 5)),
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
        )


def migrate_legacy_document(
    data: dict[str, Any],
    thread_id: str,
    default_max_turns: int,
) -> ThreadDocument:
    """Convert the legacy flat message shape into the versioned thread shape."""
    messages = list(data.get("messages") or [])
    turns: list[ConversationTurn] = []
    pending_turn: ConversationTurn | None = None

    for item in messages:
        message = ThreadMessage.from_dict(item)
        if message.role == "user":
            if pending_turn is not None:
                pending_turn.updated_at = pending_turn.user_message.timestamp
                turns.append(pending_turn)
            pending_turn = ConversationTurn(
                turn_id=new_id("turn"),
                user_message=message,
                assistant_message=None,
                state="pending",
                started_at=message.timestamp,
                updated_at=message.timestamp,
            )
            continue

        if message.role == "assistant" and pending_turn is not None:
            pending_turn.assistant_message = message
            pending_turn.state = "completed"
            pending_turn.updated_at = message.timestamp
            turns.append(pending_turn)
            pending_turn = None
            continue

        turns.append(
            ConversationTurn(
                turn_id=new_id("turn"),
                user_message=None,
                assistant_message=message if message.role == "assistant" else None,
                state="completed" if message.role == "assistant" else "pending",
                started_at=message.timestamp,
                updated_at=message.timestamp,
            )
        )

    if pending_turn is not None:
        turns.append(pending_turn)

    created_at = turns[0].started_at if turns else utc_now_iso()
    updated_at = turns[-1].updated_at if turns else created_at
    return ThreadDocument(
        version=THREAD_DOCUMENT_VERSION,
        thread_id=thread_id,
        title=str(data.get("title") or ""),
        profile=dict(data.get("profile") or {}),
        turns=turns,
        summary=str(data.get("summary") or ""),
        context_summary="",
        context_summary_updated_at=updated_at,
        context_compacted_turn_count=0,
        notices=[],
        references=[],
        max_turns=int(data.get("max_turns", default_max_turns)),
        created_at=created_at,
        updated_at=updated_at,
    )
