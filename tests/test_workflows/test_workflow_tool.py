"""Tests for the knowledge workflow tool factory."""

from __future__ import annotations

from agent.tools import make_knowledge_workflow_tool


class StubWorkflowService:
    def execute(self, *, user_query, retrieval_context_strategy="compressed"):
        del user_query, retrieval_context_strategy
        evidence = {
            "content": "The library is near the center of campus.",
            "source": "campus.md",
            "metadata": {"source": "campus.md", "section": "location"},
        }
        trace = {
            "step": "retrieval",
            "source": "retrieval",
            "tool_name": "list_retrieval_tool",
            "tool_args": {"query": "library location"},
            "tool_output": "route=list\nquery=library location\nresult_count=1",
        }
        return {
            "retrieval_context": "The library is near the center of campus.",
            "evidence_bundle": [evidence],
            "resolved_queries": [{"route": "list", "source_query": "library location"}],
            "route_trace": ["list"],
            "workflow_trace": [trace],
        }


def test_workflow_tool_returns_structured_result_contract() -> None:
    tool_callable = make_knowledge_workflow_tool(
        StubWorkflowService(),
    )

    output = tool_callable.invoke({"query": "Where is the library?"})
    assert tool_callable.args_schema is not None
    assert output["retrieval_context"] == "The library is near the center of campus."
    assert output["route_trace"] == ["list"]
    assert output["evidence_bundle"][0]["source"] == "campus.md"
    assert output["resolved_queries"][0]["source_query"] == "library location"
    assert output["workflow_trace"][0]["tool_name"] == "list_retrieval_tool"
