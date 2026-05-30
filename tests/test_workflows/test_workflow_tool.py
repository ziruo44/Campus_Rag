"""Tests for the retrieve tool factories."""

from __future__ import annotations

from agent.tools import make_life_guide_retrieve_tool, make_major_retrieve_tool


class StubWorkflowService:
    def __init__(self, *, route: str, tool_name: str, source: str) -> None:
        self.route = route
        self.tool_name = tool_name
        self.source = source

    def execute(self, *, user_query, retrieval_context_strategy="compressed"):
        del retrieval_context_strategy
        evidence = {
            "content": "stub retrieval context",
            "source": self.source,
            "metadata": {"source": self.source},
        }
        trace = {
            "step": "retrieval",
            "source": "retrieval",
            "tool_name": self.tool_name,
            "tool_args": {"query": user_query},
            "tool_output": f"route={self.route}\nquery={user_query}\nresult_count=1",
        }
        return {
            "retrieval_context": "stub retrieval context",
            "evidence_bundle": [evidence],
            "resolved_queries": [{"route": self.route, "source_query": user_query}],
            "route_trace": [self.route],
            "workflow_trace": [trace],
        }


def test_major_retrieve_tool_returns_structured_result_contract() -> None:
    tool_callable = make_major_retrieve_tool(
        StubWorkflowService(
            route="detail",
            tool_name="detail_retrieval_tool",
            source="majors.md",
        ),
    )

    output = tool_callable.invoke({"query": "人工智能专业介绍"})
    assert tool_callable.args_schema is not None
    assert output["retrieval_context"] == "stub retrieval context"
    assert output["route_trace"] == ["detail"]
    assert output["evidence_bundle"][0]["source"] == "majors.md"
    assert output["resolved_queries"][0]["source_query"] == "人工智能专业介绍"
    assert output["workflow_trace"][0]["tool_name"] == "detail_retrieval_tool"


def test_life_guide_retrieve_tool_returns_structured_result_contract() -> None:
    tool_callable = make_life_guide_retrieve_tool(
        StubWorkflowService(
            route="life_guide",
            tool_name="life_guide_retrieval_tool",
            source="生活指南.md",
        ),
    )

    output = tool_callable.invoke({"query": "校园网怎么用"})
    assert tool_callable.args_schema is not None
    assert output["retrieval_context"] == "stub retrieval context"
    assert output["route_trace"] == ["life_guide"]
    assert output["evidence_bundle"][0]["source"] == "生活指南.md"
    assert output["resolved_queries"][0]["source_query"] == "校园网怎么用"
    assert output["workflow_trace"][0]["tool_name"] == "life_guide_retrieval_tool"
