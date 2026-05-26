"""Locking helpers for concurrent thread persistence."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from ctypes import wintypes
import ctypes
import json
import os
import threading
import time


class LockTimeoutError(TimeoutError):
    """Raised when a memory lock cannot be acquired in time."""


class LockRegistry:
    """Shared in-process lock registry keyed by file path."""

    _registry_lock = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    @classmethod
    def get_lock(cls, key: Path) -> threading.RLock:
        normalized_key = str(key.resolve())
        with cls._registry_lock:
            lock = cls._locks.get(normalized_key)
            if lock is None:
                lock = threading.RLock()
                cls._locks[normalized_key] = lock
            return lock


class FileMutex:
    """Cross-process mutex implemented with lock files."""

    def __init__(self, path: Path, timeout_seconds: float):
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._acquired = False

    def acquire(self) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            self._cleanup_stale_lock()
            try:
                fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(f"Timed out waiting for lock: {self.path}")
                time.sleep(0.05)
                continue

            payload = {
                "pid": os.getpid(),
                "created_at": time.time(),
            }
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            self._acquired = True
            return

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self.path.unlink(missing_ok=True)
        finally:
            self._acquired = False

    def _cleanup_stale_lock(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        pid = payload.get("pid")
        if isinstance(pid, int) and not _pid_exists(pid):
            self.path.unlink(missing_ok=True)
            return
        created_at = float(payload.get("created_at", 0.0))
        stale_after_seconds = max(self.timeout_seconds * 5, 30.0)
        if created_at and (time.time() - created_at) > stale_after_seconds:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "FileMutex":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()


@contextmanager
def locked_path(path: Path, timeout_seconds: float):
    """Acquire the in-process and file lock for a path."""
    process_lock = LockRegistry.get_lock(path)
    with process_lock:
        with FileMutex(path=path, timeout_seconds=timeout_seconds):
            yield


def _pid_exists(pid: int) -> bool:
    """Return whether a process ID appears to still exist."""
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            wintypes.DWORD(process_query_limited_information),
            wintypes.BOOL(False),
            wintypes.DWORD(pid),
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True
