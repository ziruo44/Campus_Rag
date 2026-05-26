"""Memory system settings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.paths import get_data_dir


class MemorySettings(BaseSettings):
    """Settings for the file-backed memory system."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    max_tokens: int = 4000
    max_turns: int = 5
    retention_days: int = 30
    lock_timeout_seconds: float = 10.0
    max_references_per_thread: int = 5
    reference_recent_turns_limit: int = 1
    reference_summary_char_limit: int = 500
    backup_corrupt_files: bool = True
    session_dir: Path = Field(
        default_factory=lambda: get_data_dir() / "memory" / "sessions"
    )
    current_session_file: Path = Field(
        default_factory=lambda: get_data_dir() / "memory" / ".current_session"
    )
    compaction_notice_callback: Callable[[dict], None] | None = None

    def get_session_path(self, session_id: str) -> Path:
        """Get the file path for a thread ID."""
        return self.session_dir / f"{session_id}.json"

    def get_lock_path(self, session_id: str) -> Path:
        """Get the lock file path for a thread ID."""
        return self.session_dir / f"{session_id}.lock"

    def get_current_session_lock_path(self) -> Path:
        """Get the lock file path for the current-thread pointer."""
        return self.current_session_file.with_suffix(".lock")
