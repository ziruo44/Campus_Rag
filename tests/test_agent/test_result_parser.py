"""Tests for raw agent result parsing."""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from agent.result_parser import build_knowledge_turn_result


def test_build_knowledge_turn_result_extracts_answer_and_artifacts() -> None:
    workflow_result = {
        "retrieval_context": "retrieval context",
        "evidence_bundle": [
            {
                "content": "Artificial Intelligence covers intelligent systems.",
                "source": "majors.md",
                "metadata": {"source": "majors.md", "major": "AI"},
            }
        ],
        "resolved_queries": [{"route": "detail", "source_query": "AI major"}],
        "route_trace": ["detail"],
        "workflow_trace": [
            {
                "step": "retrieval",
                "source": "retrieval",
                "tool_name": "detail_retrieval_tool",
                "tool_args": {"query": "AI major"},
                "tool_output": "retrieval context",
            }
        ],
    }
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_framework",
                    "name": "major_retrieve_tool",
                    "args": {"query": "AI major"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content=workflow_result, tool_call_id="call_framework"),
        AIMessage(content="final summarized answer"),
    ]

    result = build_knowledge_turn_result(messages)

    assert result["answer"] == "final summarized answer"
    assert result["capability_type"] == "knowledge"
    assert result["artifacts"]["workflow_summary"]["route_trace"] == ["detail"]
    assert result["artifacts"]["evidence_bundle"][0]["source"] == "majors.md"
