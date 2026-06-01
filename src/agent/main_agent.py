"""单 outer agent，负责工具选择与最终回答生成。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent

from agent.middleware import NavigationHumanReviewMiddleware
from agent.tools import (
    make_campus_navigation_tool,
    make_life_guide_retrieve_tool,
    make_major_retrieve_tool,
)
from agent.workflows.life_guide_service import LifeGuideWorkflowService
from agent.workflows.service import MajorKnowledgeWorkflowService
from llm.model import model as default_chat_model
from llm.prompt_loader import get_outer_agent_system_prompt
from memory.message_builder import build_agent_messages
from memory.session import ManagedThread

_OUTER_AGENT_SYSTEM_PROMPT = get_outer_agent_system_prompt().strip()


class CampusKnowledgeAgent:
    """外层 agent 的轻量封装。"""

    def __init__(
        self,
        major_workflow_service: MajorKnowledgeWorkflowService,
        life_guide_workflow_service: LifeGuideWorkflowService,
        *,
        chat_model: Any = default_chat_model,
    ) -> None:
        self.major_workflow_service = major_workflow_service
        self.life_guide_workflow_service = life_guide_workflow_service
        self.chat_model = chat_model
        self.agent = None

    def init_agent(self) -> Any:
        """创建带有三类校园工具的外层 agent。"""
        major_tool = make_major_retrieve_tool(self.major_workflow_service)
        life_guide_tool = make_life_guide_retrieve_tool(self.life_guide_workflow_service)
        navigation_tool = make_campus_navigation_tool()
        navigation_review_middleware = NavigationHumanReviewMiddleware()
        self.agent = create_agent(
            model=self.chat_model,
            tools=[major_tool, life_guide_tool, navigation_tool],
            system_prompt=_OUTER_AGENT_SYSTEM_PROMPT,
            middleware=[navigation_review_middleware],
            name="campus-baiduren-agent",
        )
        return self.agent

    def invoke(
        self,
        thread: ManagedThread,
        message: str,
    ) -> dict[str, Any]:
        """执行单轮对话并返回框架原始消息。"""
        del message
        agent = self.agent or self.init_agent()
        messages = build_agent_messages(thread=thread, include_reference_context=True)
        result = agent.invoke({"messages": messages})
        return {
            "messages": self._normalize_messages(result.get("messages", [])),
        }

    async def astream(
        self,
        thread: ManagedThread,
        message: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """以 LangGraph v2 流式协议返回模型消息块与节点更新。"""
        del message
        agent = self.agent or self.init_agent()
        messages = build_agent_messages(thread=thread, include_reference_context=True)
        async for event in agent.astream(
            {"messages": messages},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            yield event

    def _normalize_messages(self, messages: list[Any]) -> list[Any]:
        """将框架输出消息规范化为普通列表。"""
        return list(messages)
