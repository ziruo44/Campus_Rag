"""Prompt loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR: Path | None = None


def _get_prompts_dir() -> Path:
    """Get the prompts directory path."""
    global _PROMPTS_DIR
    if _PROMPTS_DIR is None:
        _PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
    return _PROMPTS_DIR


def load_prompt(filename: str) -> str:
    """Load a prompt template from file."""
    prompts_dir = _get_prompts_dir()
    filepath = prompts_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")
    logger.debug(f"Loaded prompt from {filename}")
    return content


def get_router_prompt() -> str:
    """Get router prompt."""
    return load_prompt("router.txt")


def get_query_rewrite_prompt() -> str:
    """Get query rewrite prompt."""
    return load_prompt("query_rewrite.txt")


def get_query_decomposition_prompt() -> str:
    """Get query decomposition prompt."""
    return load_prompt("query_decomposition.txt")


def get_memory_compaction_prompt() -> str:
    """Get memory compaction prompt."""
    return load_prompt("memory_compaction.txt")
