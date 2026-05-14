"""Shared agent runtime reused by CLI and FastAPI."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Lock
from typing import Any

from langchain.agents import create_agent as create_langchain_agent

from rag_agent.agent_modules import create_agent as create_rag_agent
from rag_agent.agent_modules.message_builder import build_agent_messages
from rag_agent.agent_modules.tools.memory_tools import create_memory_tools
from rag_agent.data_processing import chunk_documents, load_documents
from rag_agent.indexing import IndexBuilder
from rag_agent.memory_session.session import ManagedThread
from rag_agent.utils.path import get_raw_data_dir

logger = logging.getLogger(__name__)


class RuntimeUnavailableError(RuntimeError):
    """Raised when the shared runtime cannot be initialized."""


@dataclass(slots=True)
class AgentInvocationResult:
    """Structured result returned by agent invocation."""

    answer: str
    messages: list[Any]
    raw_result: dict[str, Any]


class AgentRuntime:
    """Reusable runtime for loading data, index, and creating agents."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False
        self._chunks: list[Any] = []
        self._index_builder: IndexBuilder | None = None

    @property
    def is_initialized(self) -> bool:
        """Whether the runtime has been initialized."""
        return self._initialized

    def ensure_initialized(self) -> None:
        """Initialize data and vector index lazily."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                docs = load_documents(get_raw_data_dir())
                parents, children = chunk_documents(docs)
                self._chunks = parents + children

                builder = IndexBuilder()
                builder.load_or_build_index(self._chunks)
                self._index_builder = builder
                self._initialized = True
                logger.info("Agent runtime initialized with %s chunks", len(self._chunks))
            except Exception as exc:
                logger.exception("Failed to initialize agent runtime")
                raise RuntimeUnavailableError("Failed to initialize agent runtime.") from exc

    def create_langchain_agent(self, thread: ManagedThread):
        """Create a LangChain tool-calling agent bound to the given thread."""
        self.ensure_initialized()
        if self._index_builder is None:
            raise RuntimeUnavailableError("Agent runtime is not initialized.")

        memory_tools = create_memory_tools(managed_thread=thread)
        rag_agent = create_rag_agent(
            self._index_builder,
            self._chunks,
            memory_tools=memory_tools,
        )
        return create_langchain_agent(
            model=rag_agent.llm,
            tools=rag_agent.get_tools(),
            system_prompt=rag_agent.get_system_prompt(),
        )

    def invoke(self, thread: ManagedThread) -> AgentInvocationResult:
        """Invoke the agent for the current state of the thread."""
        lc_agent = self.create_langchain_agent(thread)
        result = lc_agent.invoke({"messages": build_agent_messages(thread=thread)})
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else ""
        return AgentInvocationResult(answer=answer, messages=messages, raw_result=result)
