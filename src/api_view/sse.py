"""SSE 序列化工具。"""

from __future__ import annotations

from application.chat import StreamMessageEvent
from api_view.schemas.chat import StreamEventPayload


def format_sse_event(event: StreamMessageEvent) -> str:
    """将事件序列化为带 event 名称的 SSE。"""
    payload = StreamEventPayload(
        thread_id=event.thread_id,
        turn_id=event.turn_id,
        content=event.content or None,
        error=event.error,
    )
    return (
        f"event: {event.event}\n"
        f"data: {payload.model_dump_json(exclude_none=True)}\n\n"
    )
