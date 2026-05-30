"""单 outer agent，负责工具选择与最终回答生成。"""

from __future__ import annotations

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
from memory.message_builder import build_agent_messages
from memory.session import ManagedThread

_OUTER_AGENT_SYSTEM_PROMPT = """
你是温州商学院校园问答 outer agent，负责判断用户问题应该走哪个工具，并基于工具结果生成最终回答。

工具边界必须严格遵守：
1. `major_retrieve_tool`
适用范围：专业、学院、培养方案、课程、专业特色、就业方向、毕业去向、招生相关问题。
示例：人工智能专业介绍、信息工程学院有哪些专业、培养目标是什么、就业方向怎么样。

2. `life_guide_retrieve_tool`
适用范围：宿舍、食堂、请假、校园网、智慧门户、办事流程、医务室、生活服务相关问题。
示例：怎么请假、食堂外卖、宿舍怎么选、校园网怎么用、门户在哪里、医务室在哪。

3. `campus_navigation_tool`
只用于校园地点、路线、从一个地点到另一个地点怎么走的问题，不和任何知识库工具混用。
只要已经识别出起点、终点中的任意一项，就可以先调用导航工具：已知字段正常填写，未知字段留空字符串。
调用校园导航前，需要先确认起点和终点；如果上一轮已经给出导航确认请求，而用户回复“确认”或直接修改起点/终点，再继续调用导航工具；如果用户回复“取消”，则结束本次导航。

回答要求：
- 必须优先依据工具返回的信息作答。
- 证据不足或地点无法识别时，直接说明信息不足。
- 全程使用中文。
- 不要提及工具、工作流、检索、向量库或模型内部机制。
""".strip()


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
        *,
        precise_mode: bool = False,
    ) -> dict[str, Any]:
        """执行单轮对话并返回框架原始消息。"""
        del message, precise_mode
        agent = self.agent or self.init_agent()
        messages = build_agent_messages(thread=thread, include_reference_context=True)
        result = agent.invoke({"messages": messages})
        return {
            "messages": self._normalize_messages(result.get("messages", [])),
        }

    def _normalize_messages(self, messages: list[Any]) -> list[Any]:
        """将框架输出消息规范化为普通列表。"""
        return list(messages)
