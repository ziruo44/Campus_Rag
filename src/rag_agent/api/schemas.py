"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelProviderHealthResponse(BaseModel):
    """Serialized model-provider health probe details."""

    configured: bool
    checked: bool
    reachable: bool | None
    model: str | None
    base_url: str | None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    raw_data_exists: bool
    vector_index_exists: bool
    memory_dir_accessible: bool
    runtime_initialized: bool
    model_provider: ModelProviderHealthResponse


class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str = Field(min_length=1)
    thread_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response payload."""

    thread_id: str
    answer: str


class ThreadMessageDTO(BaseModel):
    """Serialized thread message."""

    role: str
    content: str
    timestamp: str


class ThreadTurnDTO(BaseModel):
    """Serialized conversation turn."""

    turn_id: str
    state: str
    user_message: ThreadMessageDTO | None
    assistant_message: ThreadMessageDTO | None
    updated_at: str


class ThreadResponse(BaseModel):
    """Serialized thread document."""

    thread_id: str
    title: str
    summary: str
    profile: dict[str, Any]
    turns: list[ThreadTurnDTO]


class ThreadListItemResponse(BaseModel):
    """Serialized thread summary for history listings."""

    thread_id: str
    title: str
    summary: str
    updated_at: str
    turn_count: int
    preview: str
