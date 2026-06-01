"""线程相关 schema。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ThreadMessageDTO(BaseModel):
    """线程消息。"""

    role: str
    content: str
    timestamp: str


class ThreadTurnDTO(BaseModel):
    """线程中的单轮问答。"""

    turn_id: str
    state: str
    user_message: ThreadMessageDTO | None
    assistant_message: ThreadMessageDTO | None
    updated_at: str


class ThreadResponse(BaseModel):
    """线程详情响应。"""

    thread_id: str
    title: str
    summary: str
    context_summary: str = ""
    context_summary_updated_at: str = ""
    context_compacted_turn_count: int = 0
    profile: dict[str, Any]
    turns: list[ThreadTurnDTO]


class ThreadListItemResponse(BaseModel):
    """线程摘要响应。"""

    thread_id: str
    title: str
    summary: str
    updated_at: str
    turn_count: int
    preview: str
