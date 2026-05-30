"""专业知识检索工具。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.workflows.service import MajorKnowledgeWorkflowService


class MajorRetrieveToolInput(BaseModel):
    query: str = Field(description="用户的专业、学院、培养方案、就业方向相关问题")


def make_major_retrieve_tool(workflow_service: MajorKnowledgeWorkflowService):
    @tool(args_schema=MajorRetrieveToolInput)
    def major_retrieve_tool(query: str) -> dict[str, Any]:
        """检索专业、学院、培养方案、课程与就业方向相关知识。"""
        return workflow_service.execute(user_query=query)

    return major_retrieve_tool
