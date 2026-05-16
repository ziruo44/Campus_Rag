"""Tests for the hardened memory session subsystem."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import threading

from rag_agent.memory_session.config import MemorySettings
from rag_agent.memory_session.models import THREAD_DOCUMENT_VERSION
from rag_agent.memory_session.session import SessionManager, ThreadStore


def build_settings(tmp_path: Path, **overrides) -> MemorySettings:
    """Create isolated memory settings for tests."""
    values = {
        "session_dir": tmp_path / "sessions",
        "current_session_file": tmp_path / ".current_session",
        "max_turns": 3,
        "retention_days": 30,
        "lock_timeout_seconds": 2.0,
        "max_references_per_thread": 4,
        "reference_recent_turns_limit": 2,
        "reference_summary_char_limit": 32,
        "backup_corrupt_files": True,
    }
    values.update(overrides)
    return MemorySettings(**values)


def load_create_memory_tools():
    """Load the memory_tools module without importing the whole agent package."""
    tools_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "rag_agent"
        / "agent_modules"
        / "tools"
        / "memory_tools.py"
    )
    spec = spec_from_file_location("memory_tools_for_test", tools_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.create_memory_tools


def test_legacy_json_is_migrated_and_resaved(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    legacy_thread_id = "legacy01"
    legacy_path = settings.get_session_path(legacy_thread_id)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "messages": [
            {"role": "user", "content": "Hello", "timestamp": "2026-05-01T00:00:00+00:00"},
            {"role": "assistant", "content": "Hi", "timestamp": "2026-05-01T00:00:01+00:00"},
            {"role": "user", "content": "Pending", "timestamp": "2026-05-01T00:00:02+00:00"},
        ],
        "profile": {"name": "Alice"},
        "max_turns": 4,
    }
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    store = ThreadStore(settings)
    thread = store.open_thread(legacy_thread_id)

    assert thread.to_dict()["version"] == THREAD_DOCUMENT_VERSION
    assert len(thread.to_dict()["turns"]) == 2
    assert thread.to_dict()["turns"][-1]["state"] == "pending"

    persisted = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert persisted["version"] == THREAD_DOCUMENT_VERSION
    assert "messages" not in persisted
    assert persisted["thread_id"] == legacy_thread_id


def test_turn_window_keeps_pending_tail_and_trims_old_terminal_turns(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, max_turns=2)
    thread = SessionManager(settings).memory

    first_turn = thread.append_user_turn("u1")
    thread.complete_turn(first_turn, "a1")
    second_turn = thread.append_user_turn("u2")
    thread.complete_turn(second_turn, "a2")
    third_turn = thread.append_user_turn("u3")
    thread.complete_turn(third_turn, "a3")
    pending_turn = thread.append_user_turn("u4")

    payload = thread.refresh().to_dict()

    assert [turn["turn_id"] for turn in payload["turns"]] == [third_turn, pending_turn]
    assert payload["turns"][-1]["state"] == "pending"


def test_session_switch_flushes_previous_thread_and_updates_pointer(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    manager = SessionManager(settings)
    original_thread_id = manager.current_thread_id

    manager.memory.set_profile("project", "rag")
    new_thread = manager.create_new_thread(switch=True)

    original_payload = json.loads(
        settings.get_session_path(original_thread_id).read_text(encoding="utf-8")
    )
    pointer_value = settings.current_session_file.read_text(encoding="utf-8").strip()

    assert original_payload["profile"]["project"] == "rag"
    assert pointer_value == new_thread.thread_id
    assert manager.current_thread_id == new_thread.thread_id


def test_references_merge_summary_profile_and_optional_recent_turns(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    manager = SessionManager(settings)
    primary = manager.memory
    referenced = manager.create_new_thread(switch=False)

    referenced.set_profile("topic", "admissions")
    referenced.set_summary("Summary text for the referenced thread.")
    referenced_turn = referenced.append_user_turn("Referenced user")
    referenced.complete_turn(referenced_turn, "Referenced assistant")

    primary.attach_reference(
        referenced.thread_id,
        alias="background",
        recent_turns_limit=2,
    )

    default_context = primary.build_context(include_references=True)
    full_context = primary.build_context(
        include_references=True,
        include_reference_turns=True,
    )

    assert "Referenced Thread: background" in default_context
    assert "topic: admissions" in default_context
    assert "Referenced user" not in default_context
    assert "Referenced user" in full_context
    assert "Referenced assistant" in full_context


def test_reference_cycle_detection_rejects_bidirectional_links(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    manager = SessionManager(settings)
    first = manager.memory
    second = manager.create_new_thread(switch=False)

    first.attach_reference(second.thread_id)

    try:
        second.attach_reference(first.thread_id)
    except ValueError as exc:
        assert "cycle" in str(exc).lower()
    else:
        raise AssertionError("Expected cycle detection to raise ValueError")


def test_concurrent_updates_preserve_all_profile_and_turn_changes(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, max_turns=16)
    manager = SessionManager(settings)
    thread_id = manager.current_thread_id
    store = ThreadStore(settings)

    def worker(index: int) -> None:
        thread = store.open_thread(thread_id)
        turn_id = thread.append_user_turn(f"user-{index}")
        thread.complete_turn(turn_id, f"assistant-{index}")
        thread.set_profile(f"key-{index}", f"value-{index}")

    workers = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    payload = store.open_thread(thread_id).to_dict()

    assert len(payload["turns"]) == 6
    for index in range(6):
        assert payload["profile"][f"key-{index}"] == f"value-{index}"


def test_corrupt_json_is_backed_up_and_recovered(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    broken_thread_id = "broken01"
    broken_path = settings.get_session_path(broken_thread_id)
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("{not-json", encoding="utf-8")

    store = ThreadStore(settings)
    recovered = store.open_thread(broken_thread_id)

    assert recovered.to_dict()["thread_id"] == broken_thread_id
    backups = list(broken_path.parent.glob("broken01.corrupt-*.json"))
    assert backups
    persisted = json.loads(broken_path.read_text(encoding="utf-8"))
    assert persisted["version"] == THREAD_DOCUMENT_VERSION


def test_thread_bound_memory_tools_operate_on_bound_thread(tmp_path: Path) -> None:
    try:
        import langchain_core.tools  # noqa: F401
    except ModuleNotFoundError:
        return

    settings = build_settings(tmp_path)
    manager = SessionManager(settings)
    thread = manager.memory
    thread.set_profile("name", "Bob")
    turn_id = thread.append_user_turn("hello")
    thread.complete_turn(turn_id, "hi")

    create_memory_tools = load_create_memory_tools()
    history_tool, save_tool, prefs_tool = create_memory_tools(managed_thread=thread)

    save_result = save_tool.invoke({"key": "city", "value": "Wenzhou"})
    prefs_result = prefs_tool.invoke({})
    history_result = history_tool.invoke({})

    assert "Saved preference" in save_result
    assert "city: Wenzhou" in prefs_result
    assert "Current Thread:" in history_result
    assert "hello" in history_result
