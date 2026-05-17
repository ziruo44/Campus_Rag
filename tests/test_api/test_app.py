"""Tests for the FastAPI backend."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import httpx

from rag_agent.api.app import create_app
from rag_agent.api.deps import get_agent_runtime, get_chat_service
from rag_agent.api.services.agent_runtime import ModelProviderHealthResult
from rag_agent.api.services.chat_service import ChatService
from rag_agent.memory_session.config import MemorySettings
from rag_agent.memory_session.session import SessionManager


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


class StubRuntime:
    """Small runtime stub that avoids real model calls."""

    def __init__(
        self,
        answer: str = "stub answer",
        should_fail: bool = False,
        provider_health: ModelProviderHealthResult | None = None,
    ) -> None:
        self.answer = answer
        self.should_fail = should_fail
        self.is_initialized = True
        self.chunks = []
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

    def invoke(self, thread, extra_system_context=None, retrieval_context_strategy="compressed"):
        del thread, extra_system_context, retrieval_context_strategy
        if self.should_fail:
            raise RuntimeError("runtime boom")
        from rag_agent.api.services.agent_runtime import AgentInvocationResult

        return AgentInvocationResult(
            answer=self.answer,
            messages=[],
            raw_result={},
        )

    def invoke_with_strategy(self, thread, retrieval_context_strategy="compressed"):
        return self.invoke(thread, retrieval_context_strategy=retrieval_context_strategy)

    def stream(self, thread, retrieval_context_strategy="compressed"):
        del thread, retrieval_context_strategy

        async def iterator():
            if False:
                yield ""

        return iterator()

    def probe_model_provider(self, check_connection: bool = False):
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


def create_test_client(tmp_path: Path, runtime: StubRuntime | None = None) -> TestClient:
    """Create a TestClient with dependency overrides."""
    app = create_app()
    settings = build_settings(tmp_path)
    resolved_runtime = runtime or StubRuntime()
    service = ChatService(
        runtime=resolved_runtime,
        session_manager_factory=lambda: SessionManager(settings),
    )
    app.dependency_overrides[get_agent_runtime] = lambda: resolved_runtime
    app.dependency_overrides[get_chat_service] = lambda: service
    return TestClient(app)


def test_health_returns_basic_status(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "runtime_initialized" in payload
    assert payload["model_provider"]["checked"] is False
    assert payload["model_provider"]["reachable"] is None


def test_health_can_probe_model_provider_connectivity(tmp_path: Path) -> None:
    runtime = StubRuntime(
        provider_health=ModelProviderHealthResult(
            configured=True,
            checked=True,
            reachable=True,
            model="qwen-test",
            base_url="https://example.com/v1",
            detail="Model provider responded successfully.",
        )
    )
    client = create_test_client(tmp_path, runtime=runtime)

    response = client.get("/health?check_model=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_provider"]["checked"] is True
    assert payload["model_provider"]["reachable"] is True


def test_health_reports_degraded_when_model_probe_fails(tmp_path: Path) -> None:
    runtime = StubRuntime(
        provider_health=ModelProviderHealthResult(
            configured=True,
            checked=True,
            reachable=False,
            model="qwen-test",
            base_url="https://example.com/v1",
            detail="tls handshake failed",
        )
    )
    client = create_test_client(tmp_path, runtime=runtime)

    response = client.get("/health?check_model=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["model_provider"]["detail"] == "tls handshake failed"


def test_chat_creates_new_thread_when_thread_id_missing(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "stub answer"
    assert payload["thread_id"]


def test_chat_reuses_existing_thread(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/api/chat", json={"message": "hello"})
    thread_id = first.json()["thread_id"]

    second = client.post(
        "/api/chat",
        json={"message": "follow up", "thread_id": thread_id},
    )

    assert second.status_code == 200
    assert second.json()["thread_id"] == thread_id


def test_chat_accepts_precise_mode_flag(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.post(
        "/api/chat",
        json={"message": "hello", "precise_mode": True},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "stub answer"


def test_get_thread_returns_persisted_turns(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    chat_response = client.post("/api/chat", json={"message": "hello"})
    thread_id = chat_response.json()["thread_id"]

    response = client.get(f"/api/threads/{thread_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread_id"] == thread_id
    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["user_message"]["content"] == "hello"


def test_list_threads_returns_history_summaries(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/api/chat", json={"message": "hello"})
    thread_id = first.json()["thread_id"]
    client.post(
        "/api/chat",
        json={"message": "follow up", "thread_id": thread_id},
    )

    response = client.get("/api/threads")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 1
    assert payload[0]["thread_id"] == thread_id
    assert payload[0]["title"] == "hello"
    assert payload[0]["turn_count"] == 2
    assert "preview" in payload[0]


def test_get_thread_strips_trailing_whitespace_in_thread_id(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    chat_response = client.post("/api/chat", json={"message": "hello"})
    thread_id = chat_response.json()["thread_id"]

    response = client.get(f"/api/threads/{thread_id}%0A")

    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id


def test_missing_thread_returns_404(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)

    response = client.get("/api/threads/missing-thread")

    assert response.status_code == 404


def test_chat_failure_marks_turn_failed(tmp_path: Path) -> None:
    client = create_test_client(tmp_path, runtime=StubRuntime(should_fail=True))

    response = client.post("/api/chat", json={"message": "hello"})

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
    class ConnectionErrorRuntime(StubRuntime):
        def invoke(self, thread, extra_system_context=None, retrieval_context_strategy="compressed"):
            del thread, extra_system_context, retrieval_context_strategy
            request = httpx.Request("POST", "https://example.com/v1/chat/completions")
            raise httpx.ConnectError("tls handshake failed", request=request)

    client = create_test_client(tmp_path, runtime=ConnectionErrorRuntime())

    response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert "Model provider request failed" in response.json()["detail"]
    assert "tls handshake failed" in response.json()["detail"]


def test_delete_turn_removes_user_and_assistant_messages(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    first = client.post("/api/chat", json={"message": "first question"})
    thread_id = first.json()["thread_id"]
    client.post(
        "/api/chat",
        json={"message": "second question", "thread_id": thread_id},
    )

    thread_payload = client.get(f"/api/threads/{thread_id}").json()
    turn_id = thread_payload["turns"][0]["turn_id"]

    response = client.delete(f"/api/threads/{thread_id}/turns/{turn_id}")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["turns"]) == 1
    assert payload["turns"][0]["user_message"]["content"] == "second question"


def test_delete_thread_removes_persisted_json(tmp_path: Path) -> None:
    client = create_test_client(tmp_path)
    thread_id = client.post("/api/chat", json={"message": "hello"}).json()["thread_id"]

    response = client.delete(f"/api/threads/{thread_id}")

    assert response.status_code == 204

    settings = build_settings(tmp_path)
    assert not settings.get_session_path(thread_id).exists()
