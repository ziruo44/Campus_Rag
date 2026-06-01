"""对话路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from application.chat import SendMessageUseCase, StreamMessageUseCase
from api_view.dependencies import (
    get_send_message_use_case,
    get_stream_message_use_case,
)
from api_view.schemas import ChatRequest, ChatResponse
from api_view.sse import format_sse_event

campus_router = APIRouter(prefix="/campus", tags=["campus"])


@campus_router.post("/messages", response_model=ChatResponse)
def send_message(
    payload: ChatRequest,
    use_case: SendMessageUseCase = Depends(get_send_message_use_case),
) -> ChatResponse:
    """执行单轮非流式对话。"""
    result = use_case.execute(
        message=payload.message,
        thread_id=payload.thread_id,
    )
    return ChatResponse(thread_id=result.thread_id, answer=result.answer)


@campus_router.post("/messages/stream")
async def stream_message(
    payload: ChatRequest,
    use_case: StreamMessageUseCase = Depends(get_stream_message_use_case),
) -> StreamingResponse:
    """执行单轮流式对话，使用具名 SSE 事件。"""
    event_stream = use_case.execute(
        message=payload.message,
        thread_id=payload.thread_id,
    )

    async def iterator():
        async for event in event_stream:
            yield format_sse_event(event)

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
