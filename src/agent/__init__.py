"""Unified agent entrypoint and supporting types."""

from agent.main_agent import CampusKnowledgeAgent
from agent.middleware import build_memory_messages

__all__ = [
    "CampusKnowledgeAgent",
    "build_memory_messages",
]
