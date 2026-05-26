"""Agent-facing knowledge workflow tool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent.workflows.service import KnowledgeWorkflowService



class KnowledgeWorkflowInput(BaseModel):
    query: str = Field(description="User question for campus knowledge retrieval")


def make_knowledge_workflow_tool(workflow_service: KnowledgeWorkflowService):
    @tool(args_schema=KnowledgeWorkflowInput)
    def knowledge_workflow_tool(query: str) -> dict[str, Any]:
        """Retrieve grounded campus knowledge and supporting evidence for the query."""
        return workflow_service.execute(user_query=query)

    return knowledge_workflow_tool