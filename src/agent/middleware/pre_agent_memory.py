"""Pre-agent memory helpers and middleware."""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware

from memory.message_builder import build_agent_messages
from memory.session import ManagedThread


def build_memory_messages(thread: ManagedThread) -> list[Any]:
    """Return active-thread memory and dialogue history for the outer agent."""
    return list(build_agent_messages(thread=thread, include_reference_context=True))


class PreAgentMemoryMiddleware(AgentMiddleware[AgentState, None, Any]):
    """Inject thread-backed memory into the framework agent state."""

    def __init__(self, thread: ManagedThread) -> None:
        self._thread = thread

    def before_agent(
        self,
        state: AgentState,
        runtime,
    ) -> dict[str, Any]:
        del state, runtime
        return {"messages": build_memory_messages(self._thread)}
