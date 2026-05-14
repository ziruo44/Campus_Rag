"""Prompt loading utilities."""

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_PROMPTS_DIR: Path | None = None


def _get_prompts_dir() -> Path:
    """Get the prompts directory path."""
    global _PROMPTS_DIR
    if _PROMPTS_DIR is None:
        from rag_agent.utils.path import get_project_root
        _PROMPTS_DIR = get_project_root() / "src" / "rag_agent" / "prompts"
    return _PROMPTS_DIR


def load_prompt(filename: str) -> str:
    """Load a prompt template from file.

    Args:
        filename: Name of the prompt file (e.g., "router.txt")

    Returns:
        The prompt template string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompts_dir = _get_prompts_dir()
    filepath = prompts_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    content = filepath.read_text(encoding="utf-8")
    logger.debug(f"Loaded prompt from {filename}")
    return content


def load_prompt_template(filename: str) -> Callable:
    """Load a prompt template and return a function that formats it.

    Args:
        filename: Name of the prompt file (e.g., "router.txt")

    Returns:
        A function that takes keyword arguments and returns the formatted prompt
    """
    def template_func(**kwargs) -> str:
        content = load_prompt(filename)
        return content.format(**kwargs)
    return template_func


def get_router_prompt() -> str:
    """Get router prompt."""
    return load_prompt("router.txt")


def get_query_rewrite_prompt() -> str:
    """Get query rewrite prompt."""
    return load_prompt("query_rewrite.txt")


def get_query_decomposition_prompt() -> str:
    """Get query decomposition prompt."""
    return load_prompt("query_decomposition.txt")


def get_system_prompt() -> str:
    """Get system prompt."""
    return load_prompt("system_prompt.txt")