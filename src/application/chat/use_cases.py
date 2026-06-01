"""对话应用层用例。"""

from __future__ import annotations

from collections.abc import AsyncIterator
import logging
from typing import Any, Callable

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk
from openai import APIConnectionError, APITimeoutError

from agent.main_agent import CampusKnowledgeAgent
from agent.result_parser import build_knowledge_turn_result
from agent.workflows.life_guide_service import LifeGuideWorkflowService
from agent.workflows.service import MajorKnowledgeWorkflowService
from application.chat.errors import (
    ChatExecutionError,
    ThreadNotFoundError,
    TurnNotFoundError,
)
from application.chat.models import ChatTurnResult, StreamMessageEvent
from domain.life_guide_knowledge.runtime import (
    RuntimeUnavailableError as LifeGuideRuntimeUnavailableError,
)
from domain.major_knowledge.runtime import (
    RuntimeUnavailableError as MajorRuntimeUnavailableError,
)
from memory.session import ManagedThread, SessionManager
from shared.observability.performance import (
    finish_trace,
    mark_first_token,
    measure_stage,
    record_thread_turn_count,
    set_thread_id,
    start_trace,
)
from utils.errors import format_exception_message

logger = logging.getLogger(__name__)


def _extract_stream_chunk_text(message_chunk: AIMessageChunk) -> str:
    """提取流式消息块文本，并尽量保留模型原始空白。"""
    content = getattr(message_chunk, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _is_user_visible_stream_event(event: dict[str, Any]) -> bool:
    """只透传外层 agent 最终回答阶段的模型消息块。"""
    if event.get("type") != "messages":
        return False

    namespace = event.get("ns")
    if namespace not in ((), None):
        return False

    _message, metadata = event.get("data", (None, None))
    if not isinstance(metadata, dict):
        return False

    return metadata.get("langgraph_node") == "model"


class ChatCoordinator:
    """聚合线程与 agent 调用的内部协调器。"""

    def __init__(
        self,
        *,
        major_workflow_service: MajorKnowledgeWorkflowService,
        life_guide_workflow_service: LifeGuideWorkflowService,
        session_manager_factory: Callable[[], SessionManager] | None = None,
        agent: CampusKnowledgeAgent | None = None,
    ) -> None:
        self.major_workflow_service = major_workflow_service
        self.life_guide_workflow_service = life_guide_workflow_service
        self.session_manager_factory = session_manager_factory or SessionManager
        self._agent = agent or CampusKnowledgeAgent(
            major_workflow_service,
            life_guide_workflow_service,
        )

    @property
    def agent(self) -> CampusKnowledgeAgent:
        """返回当前使用的外层 agent。"""
        return self._agent

    @agent.setter
    def agent(self, value: CampusKnowledgeAgent) -> None:
        """允许兼容层或测试替换 agent。"""
        self._agent = value

    def send_message(
        self,
        *,
        message: str,
        thread_id: str | None = None,
    ) -> ChatTurnResult:
        """打开或创建线程，执行单轮非流式对话。"""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        with self.session_manager_factory() as manager:
            thread = self.resolve_thread(manager, normalized_thread_id)
            return self.invoke_thread(thread, message)

    def stream_message(
        self,
        *,
        message: str,
        thread_id: str | None = None,
    ) -> AsyncIterator[StreamMessageEvent]:
        """打开或创建线程，执行单轮流式对话。"""
        normalized_thread_id = self._normalize_thread_id(thread_id)

        async def iterator() -> AsyncIterator[StreamMessageEvent]:
            manager = self.session_manager_factory()
            trace_handle = start_trace(
                "chat.stream",
                thread_id=normalized_thread_id,
                query=message,
            )
            full_answer = ""
            collected_messages: list[Any] = []
            turn_id: str | None = None
            thread: ManagedThread | None = None

            try:
                thread = self.resolve_thread(manager, normalized_thread_id)
                set_thread_id(thread.thread_id)
                with measure_stage("memory.append_user_turn"):
                    turn_id = thread.append_user_turn(message)
                record_thread_turn_count(len(thread.turns))

                yield StreamMessageEvent(
                    event="start",
                    thread_id=thread.thread_id,
                    turn_id=turn_id,
                )

                async for event in self._agent.astream(thread, message):
                    if event.get("type") == "updates":
                        for node_output in event.get("data", {}).values():
                            if not isinstance(node_output, dict):
                                continue
                            messages = node_output.get("messages")
                            if isinstance(messages, list):
                                collected_messages.extend(messages)
                        continue

                    if not _is_user_visible_stream_event(event):
                        continue

                    message_chunk, _metadata = event.get("data", (None, None))
                    if not isinstance(message_chunk, AIMessageChunk):
                        continue
                    if getattr(message_chunk, "tool_call_chunks", None):
                        continue

                    chunk_text = _extract_stream_chunk_text(message_chunk)
                    if not chunk_text:
                        continue

                    if not full_answer:
                        mark_first_token()
                    full_answer += chunk_text
                    yield StreamMessageEvent(
                        event="delta",
                        thread_id=thread.thread_id,
                        turn_id=turn_id,
                        content=chunk_text,
                    )

                if not collected_messages and full_answer:
                    collected_messages.append(AIMessage(content=full_answer))

                result = build_knowledge_turn_result(collected_messages)
                final_answer = result["answer"]
                if final_answer and full_answer != final_answer:
                    suffix = final_answer[len(full_answer) :] if final_answer.startswith(full_answer) else final_answer
                    if suffix:
                        if not full_answer:
                            mark_first_token()
                        full_answer = final_answer
                        yield StreamMessageEvent(
                            event="delta",
                            thread_id=thread.thread_id,
                            turn_id=turn_id,
                            content=suffix,
                        )

                with measure_stage("memory.complete_turn"):
                    thread.complete_turn_with_artifacts(
                        turn_id,
                        full_answer,
                        artifacts=result["artifacts"],
                    )
                finish_trace(trace_handle, status="completed")
                yield StreamMessageEvent(
                    event="done",
                    thread_id=thread.thread_id,
                    turn_id=turn_id,
                )
            except Exception as exc:
                resolved_error = self.normalize_execution_exception(exc)
                if thread is not None and turn_id is not None:
                    with measure_stage("memory.fail_turn"):
                        thread.fail_turn(turn_id, str(resolved_error))
                finish_trace(trace_handle, status="failed", error=str(resolved_error))
                if thread is not None and turn_id is not None:
                    yield StreamMessageEvent(
                        event="error",
                        thread_id=thread.thread_id,
                        turn_id=turn_id,
                        error=str(resolved_error),
                    )
                    return
                raise resolved_error from exc
            finally:
                manager.close()

        return iterator()

    def invoke_thread(
        self,
        thread: ManagedThread,
        message: str,
    ) -> ChatTurnResult:
        """在显式线程上执行单轮对话。"""
        trace_handle = start_trace(
            "chat.invoke",
            thread_id=thread.thread_id,
            query=message,
        )
        with measure_stage("memory.append_user_turn"):
            turn_id = thread.append_user_turn(message)
        record_thread_turn_count(len(thread.turns))
        try:
            raw_result = self._agent.invoke(
                thread,
                message,
            )
            result = build_knowledge_turn_result(raw_result["messages"])
            with measure_stage("memory.complete_turn"):
                thread.complete_turn_with_artifacts(
                    turn_id,
                    result["answer"],
                    artifacts=result["artifacts"],
                )
            finish_trace(trace_handle, status="completed")
            return ChatTurnResult(
                thread_id=thread.thread_id,
                answer=result["answer"],
                messages=result["messages"],
                artifacts=result["artifacts"],
            )
        except Exception as exc:
            resolved_error = self.normalize_execution_exception(exc)
            with measure_stage("memory.fail_turn"):
                thread.fail_turn(turn_id, str(resolved_error))
            finish_trace(trace_handle, status="failed", error=str(resolved_error))
            raise resolved_error from exc

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        """按线程 ID 返回线程内容。"""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        if normalized_thread_id is None:
            raise ThreadNotFoundError("Thread not found.")
        with self.session_manager_factory() as manager:
            try:
                thread = manager.open_thread(normalized_thread_id)
            except FileNotFoundError as exc:
                raise ThreadNotFoundError(
                    f"Thread not found: {normalized_thread_id}"
                ) from exc
            return thread.to_dict()

    def list_threads(self) -> list[dict[str, Any]]:
        """返回线程摘要列表。"""
        with self.session_manager_factory() as manager:
            summaries: list[dict[str, Any]] = []
            for thread_id in manager.list_threads():
                thread = manager.open_thread(thread_id, read_only=True)
                payload = thread.to_dict()
                turns = payload.get("turns", [])
                summaries.append(
                    {
                        "thread_id": payload["thread_id"],
                        "title": payload.get("title", "") or "New Session",
                        "summary": payload.get("summary", ""),
                        "updated_at": payload.get("updated_at", ""),
                        "turn_count": len(turns),
                        "preview": self._build_preview(payload),
                    }
                )

            return sorted(
                summaries,
                key=lambda item: item["updated_at"],
                reverse=True,
            )

    def delete_thread(self, thread_id: str) -> None:
        """删除整条线程。"""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        if normalized_thread_id is None:
            raise ThreadNotFoundError("Thread not found.")

        with self.session_manager_factory() as manager:
            try:
                manager.delete_thread(normalized_thread_id)
            except FileNotFoundError as exc:
                raise ThreadNotFoundError(
                    f"Thread not found: {normalized_thread_id}"
                ) from exc

    def delete_turn(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        """删除线程中的一轮问答。"""
        normalized_thread_id = self._normalize_thread_id(thread_id)
        normalized_turn_id = self._normalize_turn_id(turn_id)
        if normalized_thread_id is None:
            raise ThreadNotFoundError("Thread not found.")
        if normalized_turn_id is None:
            raise TurnNotFoundError("Turn not found.")

        with self.session_manager_factory() as manager:
            try:
                thread = manager.open_thread(normalized_thread_id)
            except FileNotFoundError as exc:
                raise ThreadNotFoundError(
                    f"Thread not found: {normalized_thread_id}"
                ) from exc

            try:
                thread.delete_turn(normalized_turn_id)
            except ValueError as exc:
                raise TurnNotFoundError(
                    f"Turn not found: {normalized_turn_id}"
                ) from exc

            return thread.to_dict()

    def resolve_thread(
        self,
        manager: SessionManager,
        thread_id: str | None,
    ) -> ManagedThread:
        """打开已有线程，或创建新线程。"""
        if thread_id is None:
            return manager.create_new_thread(switch=False)
        try:
            return manager.open_thread(thread_id)
        except FileNotFoundError as exc:
            raise ThreadNotFoundError(f"Thread not found: {thread_id}") from exc

    def normalize_execution_exception(self, exc: Exception) -> Exception:
        """统一业务可见的执行异常。"""
        if isinstance(
            exc,
            (
                MajorRuntimeUnavailableError,
                LifeGuideRuntimeUnavailableError,
                ChatExecutionError,
            ),
        ):
            return exc

        if isinstance(exc, (APIConnectionError, APITimeoutError, httpx.HTTPError)):
            detail = self._describe_exception(exc)
            logger.exception("Model provider request failed")
            return MajorRuntimeUnavailableError(
                f"Model provider request failed: {detail}"
            )

        detail = self._describe_exception(exc)
        if detail:
            return ChatExecutionError(f"Failed to execute chat request: {detail}")
        return ChatExecutionError("Failed to execute chat request.")

    def _normalize_thread_id(self, thread_id: str | None) -> str | None:
        """标准化外部传入的线程 ID。"""
        if thread_id is None:
            return None
        normalized = thread_id.strip()
        return normalized or None

    def _normalize_turn_id(self, turn_id: str | None) -> str | None:
        """标准化外部传入的轮次 ID。"""
        if turn_id is None:
            return None
        normalized = turn_id.strip()
        return normalized or None

    def _build_preview(self, payload: dict[str, Any]) -> str:
        """生成线程列表中的预览文本。"""
        summary = str(payload.get("summary", "")).strip()
        if summary:
            return summary[:96]

        turns = list(payload.get("turns", []))
        for turn in reversed(turns):
            assistant = turn.get("assistant_message") or {}
            assistant_content = str(assistant.get("content", "")).strip()
            if assistant_content:
                return assistant_content.replace("\n", " ")[:96]

            user = turn.get("user_message") or {}
            user_content = str(user.get("content", "")).strip()
            if user_content:
                return user_content.replace("\n", " ")[:96]

        return "Empty session"

    def _describe_exception(self, exc: BaseException) -> str:
        """提取最深层、非空的异常描述。"""
        return format_exception_message(exc)


class SendMessageUseCase:
    """执行单轮非流式消息发送。"""

    def __init__(self, coordinator: ChatCoordinator) -> None:
        self._coordinator = coordinator

    def execute(
        self,
        *,
        message: str,
        thread_id: str | None = None,
    ) -> ChatTurnResult:
        return self._coordinator.send_message(
            message=message,
            thread_id=thread_id,
        )


class StreamMessageUseCase:
    """执行单轮流式消息发送。"""

    def __init__(self, coordinator: ChatCoordinator) -> None:
        self._coordinator = coordinator

    def execute(
        self,
        *,
        message: str,
        thread_id: str | None = None,
    ) -> AsyncIterator[StreamMessageEvent]:
        return self._coordinator.stream_message(
            message=message,
            thread_id=thread_id,
        )


class ListThreadsUseCase:
    """列出线程摘要。"""

    def __init__(self, coordinator: ChatCoordinator) -> None:
        self._coordinator = coordinator

    def execute(self) -> list[dict[str, Any]]:
        return self._coordinator.list_threads()


class GetThreadUseCase:
    """获取单个线程详情。"""

    def __init__(self, coordinator: ChatCoordinator) -> None:
        self._coordinator = coordinator

    def execute(self, *, thread_id: str) -> dict[str, Any]:
        return self._coordinator.get_thread(thread_id)


class DeleteThreadUseCase:
    """删除整条线程。"""

    def __init__(self, coordinator: ChatCoordinator) -> None:
        self._coordinator = coordinator

    def execute(self, *, thread_id: str) -> None:
        self._coordinator.delete_thread(thread_id)


class DeleteTurnUseCase:
    """删除线程中的一轮问答。"""

    def __init__(self, coordinator: ChatCoordinator) -> None:
        self._coordinator = coordinator

    def execute(self, *, thread_id: str, turn_id: str) -> dict[str, Any]:
        return self._coordinator.delete_turn(thread_id, turn_id)
