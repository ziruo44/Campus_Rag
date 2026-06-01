"""对话兼容服务层。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Callable

from agent.workflows.life_guide_service import LifeGuideWorkflowService
from agent.workflows.service import MajorKnowledgeWorkflowService
from application.chat import (
    ChatCoordinator,
    ChatExecutionError,
    ChatTurnResult,
    DeleteThreadUseCase,
    DeleteTurnUseCase,
    GetThreadUseCase,
    ListThreadsUseCase,
    SendMessageUseCase,
    StreamMessageEvent,
    StreamMessageUseCase,
    ThreadNotFoundError,
    TurnNotFoundError,
)
from memory.session import ManagedThread, SessionManager


class ChatService:
    """保留给 CLI 与旧依赖使用的兼容封装。"""

    def __init__(
        self,
        major_workflow_service: MajorKnowledgeWorkflowService,
        life_guide_workflow_service: LifeGuideWorkflowService,
        session_manager_factory: Callable[[], SessionManager] | None = None,
    ) -> None:
        self.coordinator = ChatCoordinator(
            major_workflow_service=major_workflow_service,
            life_guide_workflow_service=life_guide_workflow_service,
            session_manager_factory=session_manager_factory,
        )
        self.send_message_use_case = SendMessageUseCase(self.coordinator)
        self.stream_message_use_case = StreamMessageUseCase(self.coordinator)
        self.list_threads_use_case = ListThreadsUseCase(self.coordinator)
        self.get_thread_use_case = GetThreadUseCase(self.coordinator)
        self.delete_thread_use_case = DeleteThreadUseCase(self.coordinator)
        self.delete_turn_use_case = DeleteTurnUseCase(self.coordinator)

    @property
    def agent(self):
        """兼容测试替换 agent 的旧写法。"""
        return self.coordinator.agent

    @agent.setter
    def agent(self, value) -> None:
        self.coordinator.agent = value

    def chat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> ChatTurnResult:
        """执行单轮非流式对话。"""
        return self.send_message_use_case.execute(
            message=message,
            thread_id=thread_id,
        )

    def stream_chat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> AsyncIterator[StreamMessageEvent]:
        """执行单轮流式对话。"""
        return self.stream_message_use_case.execute(
            message=message,
            thread_id=thread_id,
        )

    def invoke_thread(
        self,
        thread: ManagedThread,
        message: str,
    ) -> ChatTurnResult:
        """在显式线程上执行单轮对话。"""
        return self.coordinator.invoke_thread(
            thread,
            message,
        )

    def get_thread(self, thread_id: str) -> dict:
        """返回线程详情。"""
        return self.get_thread_use_case.execute(thread_id=thread_id)

    def list_threads(self) -> list[dict]:
        """返回线程摘要列表。"""
        return self.list_threads_use_case.execute()

    def delete_thread(self, thread_id: str) -> None:
        """删除线程。"""
        self.delete_thread_use_case.execute(thread_id=thread_id)

    def delete_turn(self, thread_id: str, turn_id: str) -> dict:
        """删除一轮问答。"""
        return self.delete_turn_use_case.execute(
            thread_id=thread_id,
            turn_id=turn_id,
        )


__all__ = [
    "ChatExecutionError",
    "ChatService",
    "ChatTurnResult",
    "StreamMessageEvent",
    "ThreadNotFoundError",
    "TurnNotFoundError",
]
