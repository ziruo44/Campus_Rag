"""Chat and thread routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import StreamingResponse

from rag_agent.api.deps import get_chat_service
from rag_agent.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ThreadListItemResponse,
    ThreadResponse,
)
from rag_agent.api.services.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Execute a single chat turn."""
    result = chat_service.chat(
        message=payload.message,
        thread_id=payload.thread_id,
        precise_mode=payload.precise_mode,
    )
    return ChatResponse(thread_id=result.thread_id, answer=result.answer)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Execute a single chat turn and stream JSON SSE chunks."""
    event_stream = chat_service.stream_chat(
        message=payload.message,
        thread_id=payload.thread_id,
        precise_mode=payload.precise_mode,
    )

    async def iterator():
        async for chunk in event_stream:
            yield _format_sse_chunk(chunk)

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/threads", response_model=list[ThreadListItemResponse])
def list_threads(
    chat_service: ChatService = Depends(get_chat_service),
) -> list[ThreadListItemResponse]:
    """List thread summaries for history views."""
    return [
        ThreadListItemResponse.model_validate(item)
        for item in chat_service.list_threads()
    ]


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> ThreadResponse:
    """Fetch a thread by ID."""
    thread_payload = chat_service.get_thread(thread_id)
    return ThreadResponse.model_validate(thread_payload)


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> Response:
    """Delete an entire thread and its persisted JSON file."""
    chat_service.delete_thread(thread_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/threads/{thread_id}/turns/{turn_id}", response_model=ThreadResponse)
def delete_turn(
    thread_id: str,
    turn_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> ThreadResponse:
    """Delete a full conversation turn from a thread."""
    thread_payload = chat_service.delete_turn(thread_id, turn_id)
    return ThreadResponse.model_validate(thread_payload)


def _format_sse_chunk(chunk: ChatStreamChunk) -> str:
    """Format a JSON payload as one SSE data event."""
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
