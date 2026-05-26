"""Tests for the framework-backed outer agent and memory helpers."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.main_agent import CampusKnowledgeAgent
from agent.middleware import build_memory_messages
from memory.config import MemorySettings
from memory.session import SessionManager


def build_settings(tmp_path: Path) -> MemorySettings:
    return MemorySettings(
        session_dir=tmp_path / "sessions",
        current_session_file=tmp_path / ".current_session",
        max_turns=5,
        retention_days=30,
        lock_timeout_seconds=2.0,
        max_references_per_thread=4,
        reference_recent_turns_limit=1,
        reference_summary_char_limit=128,
        backup_corrupt_files=True,
    )


class StubWorkflowService:
    def __init__(self) -> None:
        self._chat_model = object()

    def execute(self, *, user_query, retrieval_context_strategy="compressed"):
        del retrieval_context_strategy
        evidence = {
            "content": "Artificial Intelligence covers intelligent systems and data analysis.",
            "source": "majors.md",
            "metadata": {"source": "majors.md", "major": "AI"},
        }
        trace = {
            "step": "retrieval",
            "source": "retrieval",
            "tool_name": "detail_retrieval_tool",
            "tool_args": {"query": user_query},
            "tool_output": "retrieval context",
        }
        return {
            "retrieval_context": "retrieval context",
            "evidence_bundle": [evidence],
            "resolved_queries": [
                {"route": "detail", "source_query": user_query}
            ],
            "route_trace": ["detail"],
            "workflow_trace": [trace],
        }

    @property
    def chat_model(self):
        return self._chat_model


def test_memory_builder_builds_messages_with_reference_context(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    with SessionManager(settings) as manager:
        thread = manager.thread
        thread.set_summary("User is asking about majors.")
        thread.set_profile("name", "Alice")
        first_turn = thread.append_user_turn("Tell me about AI")
        thread.complete_turn(first_turn, "AI major overview")

        reference = manager.create_new_thread(switch=False)
        reference.set_summary("Referenced admissions context.")
        reference.set_profile("topic", "admissions")
        ref_turn = reference.append_user_turn("Referenced user")
        reference.complete_turn(ref_turn, "Referenced assistant")
        thread.attach_reference(reference.thread_id, alias="background", recent_turns_limit=1)

        messages = build_memory_messages(thread)

    assert isinstance(messages[0], SystemMessage)
    assert "User is asking about majors." in messages[0].content
    assert "name: Alice" in messages[0].content
    assert "Referenced admissions context." in messages[0].content
    assert "Referenced user" not in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Tell me about AI"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "AI major overview"


def test_memory_builder_replays_only_active_window_and_keeps_summary(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    with SessionManager(settings) as manager:
        thread = manager.thread
        for index in range(6):
            turn_id = thread.append_user_turn(f"question-{index}")
            thread.complete_turn(turn_id, f"answer-{index}")

        messages = build_memory_messages(thread)

    assert isinstance(messages[0], SystemMessage)
    assert "Older conversation context:" in messages[0].content
    replayed_human_messages = [
        message.content for message in messages if isinstance(message, HumanMessage)
    ]
    assert replayed_human_messages == ["question-5"]


def test_memory_builder_keeps_pending_turn_in_active_window(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    with SessionManager(settings) as manager:
        thread = manager.thread
        for index in range(5):
            turn_id = thread.append_user_turn(f"question-{index}")
            thread.complete_turn(turn_id, f"answer-{index}")
        thread.append_user_turn("pending-question")

        messages = build_memory_messages(thread)

    replayed_human_messages = [
        message.content for message in messages if isinstance(message, HumanMessage)
    ]
    assert replayed_human_messages[-1] == "pending-question"
    assert replayed_human_messages == ["pending-question"]


def test_outer_agent_uses_framework_agent_and_tool_output(monkeypatch, tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    workflow_service = StubWorkflowService()
    chat_model = object()

    class FakeFrameworkAgent:
        def __init__(self, tool_callable):
            self.tool_callable = tool_callable

        def invoke(self, payload):
            messages = list(payload["messages"])
            tool_text = self.tool_callable.invoke({"query": "What are the job prospects?"})
            return {
                "messages": [
                    *messages,
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_framework",
                                "name": "knowledge_workflow_tool",
                                "args": {"query": "What are the job prospects?"},
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(content=tool_text, tool_call_id="call_framework"),
                    AIMessage(content="final summarized answer"),
                ]
            }

    captured: dict[str, object] = {}

    def fake_create_agent(*, model, tools, system_prompt, middleware, name):
        captured["model"] = model
        captured["tool_names"] = [tool.name for tool in tools]
        captured["middleware_names"] = [
            item.__class__.__name__ for item in middleware
        ]
        captured["system_prompt"] = system_prompt
        captured["name"] = name
        return FakeFrameworkAgent(tools[0])

    import agent.main_agent as ma
    monkeypatch.setattr(ma, "create_agent", fake_create_agent)

    with SessionManager(settings) as manager:
        thread = manager.thread
        prior_turn = thread.append_user_turn("Tell me about the AI major")
        thread.complete_turn(prior_turn, "The AI major covers intelligent systems.")
        latest_turn = thread.append_user_turn("What are the job prospects?")
        del latest_turn

        agent = CampusKnowledgeAgent(
            workflow_service,
            chat_model=chat_model,
        )
        result = agent.invoke(
            thread,
            "What are the job prospects?",
        )

    assert captured["model"] is chat_model
    assert captured["name"] == "campus-baiduren-agent"
    assert captured["tool_names"] == [
        "knowledge_workflow_tool",
        "campus_navigation_tool",
    ]
    assert captured["middleware_names"] == ["NavigationHumanReviewMiddleware"]
    assert "校园摆渡人" in str(captured["system_prompt"])
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "final summarized answer"
    assert any(
        isinstance(message, ToolMessage)
        and "retrieval context" in str(message.content)
        for message in result["messages"]
    )


def test_outer_agent_does_not_duplicate_latest_user_message(monkeypatch, tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    workflow_service = StubWorkflowService()

    class InspectingFrameworkAgent:
        def invoke(self, payload):
            messages = list(payload["messages"])
            user_messages = [
                message.content for message in messages if isinstance(message, HumanMessage)
            ]
            assert user_messages.count("latest-question") == 1
            return {"messages": [*messages, AIMessage(content="ok")]}

    def fake_create_agent(*, model, tools, system_prompt, middleware, name):
        del model, tools, system_prompt, middleware, name
        return InspectingFrameworkAgent()

    import agent.main_agent as ma

    monkeypatch.setattr(ma, "create_agent", fake_create_agent)

    with SessionManager(settings) as manager:
        thread = manager.thread
        old_turn = thread.append_user_turn("old-question")
        thread.complete_turn(old_turn, "old-answer")
        thread.append_user_turn("latest-question")

        agent = CampusKnowledgeAgent(workflow_service, chat_model=object())
        result = agent.invoke(thread, "latest-question")

    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == "ok"
