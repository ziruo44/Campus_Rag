"""Tests for the hardened memory session subsystem."""

from __future__ import annotations

from pathlib import Path
import json
import threading

from memory.config import MemorySettings
from memory.models import THREAD_DOCUMENT_VERSION
from memory.compaction import split_turns_for_context
from memory.session import SessionManager, ThreadStore


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


def test_turn_window_keeps_pending_tail_without_deleting_history(tmp_path: Path) -> None:
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
    slices = split_turns_for_context(
        thread.turns,
        thread.max_turns,
        payload["context_compacted_turn_count"],
    )

    assert [turn["turn_id"] for turn in payload["turns"]] == [
        first_turn,
        second_turn,
        third_turn,
        pending_turn,
    ]
    assert [turn.turn_id for turn in slices.active_turns] == [third_turn, pending_turn]
    assert payload["turns"][-1]["state"] == "pending"
    assert payload["context_compacted_turn_count"] == 2


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


def test_older_turns_are_preserved_after_exceeding_window(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, max_turns=5)
    manager = SessionManager(settings)
    thread = manager.memory

    for index in range(6):
        turn_id = thread.append_user_turn(f"user-{index}")
        thread.complete_turn(turn_id, f"assistant-{index}")

    payload = thread.refresh().to_dict()

    assert len(payload["turns"]) == 6
    assert payload["turns"][0]["user_message"]["content"] == "user-0"
    assert payload["context_compacted_turn_count"] == 5
    assert payload["context_summary"]


def test_delete_turn_recomputes_context_summary(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, max_turns=2)
    manager = SessionManager(settings)
    thread = manager.memory
    turn_ids: list[str] = []

    for index in range(4):
        turn_id = thread.append_user_turn(f"user-{index}")
        thread.complete_turn(turn_id, f"assistant-{index}")
        turn_ids.append(turn_id)

    before_delete = thread.refresh().to_dict()
    assert before_delete["context_compacted_turn_count"] == 2

    thread.delete_turn(turn_ids[0])

    payload = thread.refresh().to_dict()
    assert len(payload["turns"]) == 3
    assert payload["context_compacted_turn_count"] == 2


def test_legacy_migration_initializes_context_summary_fields(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    legacy_thread_id = "legacy02"
    legacy_path = settings.get_session_path(legacy_thread_id)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "version": 3,
        "thread_id": legacy_thread_id,
        "title": "Legacy",
        "profile": {},
        "turns": [
            {
                "turn_id": "turn_1",
                "user_message": {
                    "role": "user",
                    "content": "one",
                    "timestamp": "2026-05-01T00:00:00+00:00",
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": "reply one",
                    "timestamp": "2026-05-01T00:00:01+00:00",
                },
                "state": "completed",
                "started_at": "2026-05-01T00:00:00+00:00",
                "updated_at": "2026-05-01T00:00:01+00:00",
                "error": None,
                "artifacts": {},
            }
        ],
        "summary": "",
        "references": [],
        "max_turns": 4,
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:01+00:00",
    }
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    store = ThreadStore(settings)
    payload = store.open_thread(legacy_thread_id).to_dict()

    assert payload["version"] == THREAD_DOCUMENT_VERSION
    assert payload["context_summary"] == ""
    assert payload["context_compacted_turn_count"] == 0
    assert payload["context_summary_updated_at"]


def test_compaction_rolls_previous_summary_into_next_summary(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, max_turns=2)
    manager = SessionManager(settings)
    thread = manager.memory

    for index in range(5):
        turn_id = thread.append_user_turn(f"user-{index}")
        thread.complete_turn(turn_id, f"assistant-{index}")

    payload = thread.refresh().to_dict()

    assert payload["context_compacted_turn_count"] == 4
    assert "Previously compacted context:" in payload["context_summary"]


def test_compaction_callback_is_triggered_when_segment_is_compacted(tmp_path: Path) -> None:
    callback_events: list[dict] = []
    settings = build_settings(
        tmp_path,
        max_turns=2,
        compaction_notice_callback=lambda payload: callback_events.append(dict(payload)),
    )
    manager = SessionManager(settings)
    thread = manager.memory

    for index in range(3):
        turn_id = thread.append_user_turn(f"user-{index}")
        thread.complete_turn(turn_id, f"assistant-{index}")

    assert callback_events
    assert callback_events[-1]["type"] == "context_compacted"
    assert callback_events[-1]["compacted_delta"] == 2


def test_compaction_callback_fires_before_summary_generation(tmp_path: Path, monkeypatch) -> None:
    import memory.compaction as compaction

    event_order: list[str] = []
    settings = build_settings(
        tmp_path,
        max_turns=2,
        compaction_notice_callback=lambda payload: (
            payload,
            event_order.append("callback"),
        ),
    )
    manager = SessionManager(settings)
    thread = manager.memory

    def fake_generate_summary(*, compacted_turns, previous_summary, chat_model):
        del compacted_turns, previous_summary, chat_model
        event_order.append("summary")
        return "summary"

    monkeypatch.setattr(compaction, "_generate_summary", fake_generate_summary)

    for index in range(3):
        turn_id = thread.append_user_turn(f"user-{index}")
        thread.complete_turn(turn_id, f"assistant-{index}")

    assert event_order[:2] == ["callback", "summary"]


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


