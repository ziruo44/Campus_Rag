"""Shared pytest configuration."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def stub_memory_compaction(monkeypatch):
    """Avoid real model calls in sliding-window summary tests."""
    import memory.compaction as compaction

    monkeypatch.setattr(
        compaction,
        "_generate_summary",
        lambda *, compacted_turns, previous_summary, chat_model: compaction.merge_summary_with_fallback(
            previous_summary=previous_summary,
            compacted_turns=compacted_turns,
        ),
    )


def pytest_addoption(parser) -> None:
    """Register manual workflow debug options."""
    parser.addoption(
        "--workflow-query",
        action="store",
        default="",
        help="Run the manual workflow debug test with the given query.",
    )
    parser.addoption(
        "--workflow-precise",
        action="store_true",
        default=False,
        help="Use passthrough retrieval context in the manual workflow debug test.",
    )
