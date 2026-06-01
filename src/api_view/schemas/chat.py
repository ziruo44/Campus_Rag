"""对话相关 schema。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求体。"""

    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """非流式对话响应体。"""

    thread_id: str
    answer: str


class StreamEventPayload(BaseModel):
    """新版 SSE 事件数据。"""

    thread_id: str
    turn_id: str
    content: str | None = None
    error: str | None = None
