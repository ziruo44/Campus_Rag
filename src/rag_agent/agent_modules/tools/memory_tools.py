"""Memory tools for thread-aware tool calling."""

from __future__ import annotations

import threading

from langchain_core.tools import tool

from rag_agent.memory_session.session import ManagedThread, SessionManager

_default_manager: SessionManager | None = None
_default_manager_lock = threading.Lock()


def get_memory_manager() -> SessionManager:
    """Get the compatibility default memory manager."""
    global _default_manager
    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                _default_manager = SessionManager()
    return _default_manager


def create_memory_tools(
    *,
    managed_thread: ManagedThread | None = None,
    session_manager: SessionManager | None = None,
) -> list:
    """Create memory tools bound to an explicit thread when available."""

    def resolve_thread() -> ManagedThread:
        if managed_thread is not None:
            return managed_thread
        if session_manager is not None:
            return session_manager.memory
        return get_memory_manager().memory

    @tool(description="Get the current thread context and referenced thread summaries.")
    def memory_get_history_tool() -> str:
        """Get the current thread context."""
        history = resolve_thread().build_context(include_references=True)
        if not history:
            return "No conversation history is available."
        return history

    @tool(
        description="Save a user preference or important fact in the active thread profile."
    )
    def memory_save_preference_tool(key: str, value: str) -> str:
        """Save a user preference or fact."""
        thread = resolve_thread()
        thread.set_profile(key, value)
        return f"Saved preference: {key} = {value}"

    @tool(description="Get user preferences stored in the active thread profile.")
    def memory_get_preferences_tool() -> str:
        """Get all saved user preferences."""
        prefs = resolve_thread().get_all_profile()
        if not prefs:
            return "No saved preferences."
        return "\n".join(f"{key}: {value}" for key, value in sorted(prefs.items()))

    return [
        memory_get_history_tool,
        memory_save_preference_tool,
        memory_get_preferences_tool,
    ]


memory_get_history_tool, memory_save_preference_tool, memory_get_preferences_tool = (
    create_memory_tools()
)
