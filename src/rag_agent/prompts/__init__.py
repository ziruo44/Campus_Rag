"""Prompts module - loads prompts from txt files."""

from rag_agent.utils.prompt_loader import (
    load_prompt,
    load_prompt_template,
    get_router_prompt,
    get_query_rewrite_prompt,
    get_query_decomposition_prompt,
    get_system_prompt,
)

__all__ = [
    "load_prompt",
    "load_prompt_template",
    "get_router_prompt",
    "get_query_rewrite_prompt",
    "get_query_decomposition_prompt",
    "get_system_prompt",
]