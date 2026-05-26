"""Tests for file-backed memory locking."""

from __future__ import annotations

from pathlib import Path
import json

from memory.locks import FileMutex


def test_file_mutex_cleans_up_lock_from_dead_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "dead.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 999999,
                "created_at": 0,
            }
        ),
        encoding="utf-8",
    )

    mutex = FileMutex(path=lock_path, timeout_seconds=0.1)
    mutex.acquire()

    assert lock_path.exists()
    mutex.release()
    assert not lock_path.exists()
