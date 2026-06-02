"""Tests for the FastAPI backend."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

import api_view.web_main as web_main_module
from agent.main_agent import CampusKnowledgeAgent
from app_bootstrap import (
    get_chat_service,
    get_life_guide_runtime,
    get_life_guide_workflow_service,
    get_major_knowledge_runtime,
    get_major_workflow_service,
    get_model_health_probe,
)
from api_view.services.chat_service import ChatService
from api_view.web_main import create_app
from llm.health import ModelProviderHealthResult
from memory.config import MemorySettings
from memory.session import SessionManager


def build_settings(tmp_path: Path) -> MemorySettings:
    """Create isolated memory settings for API tests."""
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
    """Small workflow-service stub that avoids real model calls."""

    def __init__(
        self,
        *,
        answer: str = "stub answer",
        should_fail: bool = False,
        provider_health: ModelProviderHealthResult | None = None,
        route_trace: list[str] | None = None,
        tool_name: str = "general_retrieval_tool",
        source: str = "stub.md",
    ) -> None:
        self.answer = answer
        self.should_fail = should_fail
        self.is_initialized = True
        self._chat_model = object()
        self.route_trace = route_trace or ["detail"]
        self.tool_name = tool_name
        self.source = source
        self.provider_health = provider_health or ModelProviderHealthResult(
            configured=True,
            checked=False,
            reachable=None,
            model="stub-model",
            base_url="https://example.com/v1",
            detail=None,
        )

    def ensure_initialized(self) -> None:
        return None

    def execute(self, *, user_query):
        if self.should_fail:
            raise RuntimeError("runtime boom")
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
            "tool_output": "stub retrieval context",
        }
        return {
            "retrieval_context": "stub retrieval context",
            "evidence_bundle": [evidence],
            "resolved_queries": [],
            "route_trace": self.route_trace,
            "workflow_trace": [trace],
        }

    @property
    def chat_model(self):
        return self._chat_model


class StubKnowledgeRuntime:
    def __init__(self, is_initialized: bool = True) -> None:
        self.is_initialized = is_initialized
        self.ensure_initialized_calls = 0

    def ensure_initialized(self) -> None:
        self.ensure_initialized_calls += 1
        self.is_initialized = True


class StubModelHealthProbe:
    def __init__(self, provider_health: ModelProviderHealthResult) -> None:
        self.provider_health = provider_health

    def probe(self, check_connection: bool = False):
        if not check_connection:
            return ModelProviderHealthResult(
                configured=self.provider_health.configured,
                checked=False,
                reachable=None,
                model=self.provider_health.model,
                base_url=self.provider_health.base_url,
                detail=self.provider_health.detail if not self.provider_health.configured else None,
            )
        return self.provider_health


class FakeFrameworkAgent:
    def __init__(self, workflow_service: StubWorkflowService, tool_callable):
        self.workflow_service = workflow_service
        self.tool_callable = tool_callable

    def invoke(self, payload):
        messages = list(payload["messages"])
        tool_text = self.tool_callable.invoke({"query": "hello"})
        return {
            "messages": [
                *messages,
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_framework",
                            "name": "major_retrieve_tool",
                            "args": {},
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content=tool_text, tool_call_id="call_framework"),
                AIMessage(content=self.workflow_service.answer),
            ]
        }

    async def astream(self, payload, stream_mode=None, version="v2"):
        del stream_mode, version
        tool_text = self.tool_callable.invoke({"query": "hello"})
        tool_call_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_framework",
                    "name": "major_retrieve_tool",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        )
        tool_message = ToolMessage(content=tool_text, tool_call_id="call_framework")
        final_message = AIMessage(content=self.workflow_service.answer)

        yield {
            "type": "updates",
            "ns": (),
            "data": {"model": {"messages": [tool_call_message]}},
        }
        yield {
            "type": "updates",
            "ns": (),
            "data": {"tools": {"messages": [tool_message]}},
        }
        yield {
            "type": "messages",
            "ns": ("major_retrieve_tool", "query_decomposition"),
            "data": (
                AIMessageChunk(content="DECOMPOSABLE: false"),
                {"langgraph_node": "model"},
            ),
        }

        for chunk_text in ("stub ", "answer"):
            yield {
                "type": "messages",
                "ns": (),
                "data": (
                    AIMessageChunk(content=chunk_text),
                    {"langgraph_node": "model"},
                ),
            }

        yield {
            "type": "updates",
            "ns": (),
            "data": {"model": {"messages": [final_message]}},
        }


def create_test_client(
    tmp_path: Path,
    major_workflow_service: StubWorkflowService | None = None,
    life_guide_workflow_service: StubWorkflowService | None = None,
) -> TestClient:
    """Create a TestClient with dependency overrides."""
    import agent.main_agent as ma

    app = create_app(prewarm_runtime=False)
    settings = build_settings(tmp_path)
    resolved_major_workflow_service = major_workflow_service or StubWorkflowService()
    resolved_life_guide_workflow_service = life_guide_workflow_service or StubWorkflowService(
        route_trace=["life_guide"],
        tool_name="life_guide_retrieval_tool",
        source="生活指南.md",
    )
    stub_major_runtime = StubKnowledgeRuntime(
        is_initialized=resolved_major_workflow_service.is_initialized
    )
    stub_life_runtime = StubKnowledgeRuntime(
        is_initialized=resolved_life_guide_workflow_service.is_initialized
    )
    stub_health_probe = StubModelHealthProbe(resolved_major_workflow_service.provider_health)

    def fake_create_agent(*, model, tools, system_prompt, middleware, name):
        del model, system_prompt, middleware, name
        return FakeFrameworkAgent(resolved_major_workflow_service, tools[0])

    ma.create_agent = fake_create_agent

    service = ChatService(
        major_workflow_service=resolved_major_workflow_service,
        life_guide_workflow_service=resolved_life_guide_workflow_service,
        session_manager_factory=lambda: SessionManager(settings),
    )
    service.agent = CampusKnowledgeAgent(
        resolved_major_workflow_service,
        resolved_life_guide_workflow_service,
    )
    app.dependency_overrides[get_major_workflow_service] = lambda: resolved_major_workflow_service
    app.dependency_overrides[get_life_guide_workflow_service] = lambda: resolved_life_guide_workflow_service
    app.dependency_overrides[get_major_knowledge_runtime] = lambda: stub_major_runtime
    app.dependency_overrides[get_life_guide_runtime] = lambda: stub_life_runtime
    app.dependency_overrides[get_model_health_probe] = lambda: stub_health_probe
    app.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(app)


def test_app_startup_warms_both_knowledge_runtimes(tmp_path: Path) -> None:
    stub_major_runtime = StubKnowledgeRuntime(is_initialized=False)
    stub_life_runtime = StubKnowledgeRuntime(is_initialized=False)
    original_major_factory = web_main_module.get_major_knowledge_runtime
    original_life_factory = web_main_module.get_life_guide_runtime
    web_main_module.get_major_knowledge_runtime = lambda: stub_major_runtime
    web_main_module.get_life_guide_runtime = lambda: stub_life_runtime
    app = create_app(prewarm_runtime=True)

    try:
        with TestClient(app):
            pass
    finally:
        web_main_module.get_major_knowledge_runtime = original_major_factory
        web_main_module.get_life_guide_runtime = original_life_factory

    assert stub_major_runtime.ensure_initialized_calls == 1
    assert stub_life_runtime.ensure_initialized_calls == 1


def test_health_returns_basic_status(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.get("/campus/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "runtime_initialized" in payload
    assert payload["model_provider"]["checked"] is False
    assert payload["model_provider"]["reachable"] is None


def test_health_can_probe_model_provider_connectivity(tmp_path: Path) -> None:
    workflow_service = StubWorkflowService(
        provider_health=ModelProviderHealthResult(
            configured=True,
            checked=True,
            reachable=True,
            model="qwen-test",
            base_url="https://example.com/v1",
            detail="Model provider responded successfully.",
        )
    )
    client = create_test_client(tmp_path, major_workflow_service=workflow_service)

    response = client.get("/campus/health?check_model=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_provider"]["checked"] is True
    assert payload["model_provider"]["reachable"] is True


def test_health_reports_degraded_when_model_probe_fails(tmp_path: Path) -> None:
    workflow_service = StubWorkflowService(
        provider_health=ModelProviderHealthResult(
            configured=True,
            checked=True,
            reachable=False,
            model="qwen-test",
            base_url="https://example.com/v1",
            detail="tls handshake failed",
        )
    )
    client = create_test_client(tmp_path, major_workflow_service=workflow_service)

    response = client.get("/campus/health?check_model=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["model_provider"]["detail"] == "tls handshake failed"


def test_campus_chat_creates_new_thread_when_thread_id_missing(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.post("/campus/messages", json={"message": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "stub answer"
    assert payload["thread_id"]


def test_chat_reuses_existing_thread(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/campus/messages", json={"message": "hello"})
    thread_id = first.json()["thread_id"]

    second = client.post(
        "/campus/messages",
        json={"message": "follow up", "thread_id": thread_id},
    )

    assert second.status_code == 200
    assert second.json()["thread_id"] == thread_id


def test_chat_accepts_request_without_optional_flags(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.post("/campus/messages", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["answer"] == "stub answer"


def test_campus_stream_returns_named_sse_events(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.post("/campus/messages/stream", json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in response.text
    assert "event: delta" in response.text
    assert response.text.count("event: delta") >= 2
    assert "event: done" in response.text
    assert '"thread_id"' in response.text
    assert '"content":"stub "' in response.text
    assert '"content":"answer"' in response.text
    assert "DECOMPOSABLE" not in response.text


def test_retired_legacy_endpoints_return_404(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    thread_id = client.post("/campus/messages", json={"message": "hello"}).json()["thread_id"]
    thread_payload = client.get(f"/campus/threads/{thread_id}").json()
    turn_id = thread_payload["turns"][0]["turn_id"]

    responses = [
        client.get("/health"),
        client.post("/api/chat", json={"message": "hello"}),
        client.post("/api/chat/stream", json={"message": "hello"}),
        client.get("/api/threads"),
        client.get(f"/api/threads/{thread_id}"),
        client.delete(f"/api/threads/{thread_id}"),
        client.delete(f"/api/threads/{thread_id}/turns/{turn_id}"),
    ]

    for response in responses:
        assert response.status_code == 404


def test_campus_get_thread_returns_persisted_turns(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    chat_response = client.post("/campus/messages", json={"message": "hello"})
    thread_id = chat_response.json()["thread_id"]

    response = client.get(f"/campus/threads/{thread_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == thread_id
    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["user_message"]["content"] == "hello"


def test_thread_reload_keeps_full_history_after_more_than_five_rounds(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/campus/messages", json={"message": "hello-0"})
    thread_id = first.json()["thread_id"]

    for index in range(1, 6):
        client.post(
            "/campus/messages",
            json={"message": f"hello-{index}", "thread_id": thread_id},
        )

    response = client.get(f"/campus/threads/{thread_id}")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["turns"]) == 6
    assert payload["context_compacted_turn_count"] == 5
    assert payload["context_summary"]


def test_campus_list_threads_returns_history_summaries(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/campus/messages", json={"message": "hello"})
    thread_id = first.json()["thread_id"]
    client.post(
        "/campus/messages",
        json={"message": "follow up", "thread_id": thread_id},
    )

    response = client.get("/campus/threads")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 1
    assert payload[0]["thread_id"] == thread_id
    assert payload[0]["title"] == "hello"
    assert payload[0]["turn_count"] == 2
    assert "preview" in payload[0]


def test_get_thread_strips_trailing_whitespace_in_thread_id(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    chat_response = client.post("/campus/messages", json={"message": "hello"})
    thread_id = chat_response.json()["thread_id"]

    response = client.get(f"/campus/threads/{thread_id}%0A")

    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id


def test_missing_thread_returns_404(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.get("/campus/threads/missing-thread")

    assert response.status_code == 404


def test_chat_failure_marks_turn_failed(tmp_path: Path) -> None:
    client = create_test_client(
        tmp_path,
        major_workflow_service=StubWorkflowService(should_fail=True),
    )

    response = client.post("/campus/messages", json={"message": "hello"})

    assert response.status_code == 500

    settings = build_settings(tmp_path)
    with SessionManager(settings) as manager:
        thread_ids = manager.list_threads()
        failed_turn_found = False
        for thread_id in thread_ids:
            thread = manager.open_thread(thread_id)
            if thread.turns and thread.turns[-1].state == "failed":
                failed_turn_found = True
                break
        assert failed_turn_found


def test_chat_returns_503_for_model_connection_errors(tmp_path: Path) -> None:
    class ConnectionErrorWorkflowService(StubWorkflowService):
        def execute(self, *, user_query):
            del user_query
            request = httpx.Request("POST", "https://example.com/v1/chat/completions")
            raise httpx.ConnectError("tls handshake failed", request=request)

    client = create_test_client(
        tmp_path,
        major_workflow_service=ConnectionErrorWorkflowService(),
    )

    response = client.post("/campus/messages", json={"message": "hello"})

    assert response.status_code == 503
    assert "Model provider request failed" in response.json()["detail"]
    assert "tls handshake failed" in response.json()["detail"]


def test_delete_turn_removes_user_and_assistant_messages(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/campus/messages", json={"message": "first question"})
    thread_id = first.json()["thread_id"]
    client.post(
        "/campus/messages",
        json={"message": "second question", "thread_id": thread_id},
    )

    thread_payload = client.get(f"/campus/threads/{thread_id}").json()
    turn_id = thread_payload["turns"][0]["turn_id"]

    response = client.delete(f"/campus/threads/{thread_id}/turns/{turn_id}")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["user_message"]["content"] == "second question"


def test_delete_thread_removes_persisted_json(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    thread_id = client.post("/campus/messages", json={"message": "hello"}).json()["thread_id"]

    response = client.delete(f"/campus/threads/{thread_id}")

    assert response.status_code == 204

    settings = build_settings(tmp_path)
    assert not settings.get_session_path(thread_id).exists()


def test_chat_persists_turn_artifacts_for_unified_agent_results(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    thread_id = client.post("/campus/messages", json={"message": "人工智能专业介绍"}).json()["thread_id"]

    settings = build_settings(tmp_path)
    with SessionManager(settings) as manager:
        thread = manager.open_thread(thread_id)
        artifacts = thread.turns[-1].to_dict()["artifacts"]

    assert artifacts["capability_type"] == "knowledge"
    assert "workflow_summary" in artifacts
    assert "workflow_trace" in artifacts["workflow_summary"]
    assert "evidence_bundle" in artifacts
