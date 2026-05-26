"""Single outer agent for campus knowledge and navigation."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from agent.middleware import NavigationHumanReviewMiddleware
from agent.tools import make_campus_navigation_tool, make_knowledge_workflow_tool
from agent.workflows.service import KnowledgeWorkflowService
from llm.model import model as default_chat_model
from memory.message_builder import build_agent_messages
from memory.session import ManagedThread

_OUTER_AGENT_SYSTEM_PROMPT = """
你是温州商学院校园摆渡人，负责校园知识问答与校园导航。
对于涉及校园知识、专业、学院、课程、就业方向或招生相关的问题，
你可以调用 `knowledge_workflow_tool`。对于校园位置、路线、从一个地点到另一个地点怎么走的问题，
你可以调用 `campus_navigation_tool`。只要已经识别出起点、终点中的任意一项，就可以先调用导航工具：已知字段正常填写，未知字段留空字符串。调用校园导航前，需要先确认起点和终点；如果上一轮已经给出导航确认请求，而用户回复“确认”或直接修改起点/终点，再继续调用导航工具；如果用户回复“取消”，则结束本次导航。
使用工具返回的信息生成最终回答。当证据不足或地点无法识别时，主动说明现有信息不够充分。
回答要简洁、自然、清楚，优先直接解决用户问题。全程使用中文回答，不要提及工具、工作流、检索或模型内部机制。
""".strip()


class CampusKnowledgeAgent:
    """Thin wrapper around the framework agent definition."""

    def __init__(
        self,
        workflow_service: KnowledgeWorkflowService,
        *,
        chat_model: Any = default_chat_model,
    ) -> None:
        self.workflow_service = workflow_service
        self.chat_model = chat_model
        self.agent = None

    def init_agent(self) -> Any:
        """Create the framework agent with shared campus tools."""
        knowledge_tool = make_knowledge_workflow_tool(self.workflow_service)
        navigation_tool = make_campus_navigation_tool()
        navigation_review_middleware = NavigationHumanReviewMiddleware()
        self.agent = create_agent(
            model=self.chat_model,
            tools=[knowledge_tool, navigation_tool],
            system_prompt=_OUTER_AGENT_SYSTEM_PROMPT,
            middleware=[navigation_review_middleware],
            name="campus-baiduren-agent",
        )
        return self.agent

    def invoke(
        self,
        thread: ManagedThread,
        message: str,
        *,
        precise_mode: bool = False,
    ) -> dict[str, Any]:
        """Run one knowledge turn and return the raw framework output."""
        del message, precise_mode
        agent = self.agent or self.init_agent()
        messages = build_agent_messages(thread=thread, include_reference_context=True)
        result = agent.invoke({"messages": messages})
        return {
            "messages": self._normalize_messages(result.get("messages", [])),
        }

    def _normalize_messages(self, messages: list[Any]) -> list[Any]:
        """Normalize framework output messages to a plain list."""
        return list(messages)
