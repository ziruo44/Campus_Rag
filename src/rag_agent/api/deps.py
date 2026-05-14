"""Application dependencies for the FastAPI backend."""

from __future__ import annotations

from functools import lru_cache

from rag_agent.api.services.agent_runtime import AgentRuntime
from rag_agent.api.services.chat_service import ChatService


@lru_cache(maxsize=1)
def get_agent_runtime() -> AgentRuntime:
    """Return the shared cached agent runtime."""
    return AgentRuntime()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Return the shared chat service."""
    return ChatService(runtime=get_agent_runtime())
