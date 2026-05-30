"""生活指南检索工具。"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.workflows.life_guide_service import LifeGuideWorkflowService


class LifeGuideRetrieveToolInput(BaseModel):
    query: str = Field(description="用户的宿舍、食堂、请假、校园网、门户、医务室等生活指南问题")


def make_life_guide_retrieve_tool(workflow_service: LifeGuideWorkflowService):
    @tool(args_schema=LifeGuideRetrieveToolInput)
    def life_guide_retrieve_tool(query: str) -> dict[str, Any]:
        """检索宿舍、食堂、请假、校园网、门户、医务室等生活指南知识。"""
        return workflow_service.execute(user_query=query)

    return life_guide_retrieve_tool
