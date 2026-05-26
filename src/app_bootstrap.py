"""Centralized application bootstrap factories."""

from __future__ import annotations

from functools import lru_cache

from agent.workflows.service import KnowledgeWorkflowService
from api_view.services.chat_service import ChatService
from domain.knowledge.runtime import KnowledgeRuntime
from llm.health import ModelProviderHealthProbe


@lru_cache(maxsize=1)
def get_knowledge_runtime() -> KnowledgeRuntime:
    """Return the shared cached knowledge runtime."""
    return KnowledgeRuntime()


@lru_cache(maxsize=1)
def get_model_health_probe() -> ModelProviderHealthProbe:
    """Return the shared model-health probe."""
    return ModelProviderHealthProbe()


@lru_cache(maxsize=1)
def get_workflow_service() -> KnowledgeWorkflowService:
    """Return the shared cached knowledge workflow service."""
    return KnowledgeWorkflowService(
        knowledge_runtime=get_knowledge_runtime(),
        health_probe=get_model_health_probe(),
    )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Return the shared chat service."""
    return ChatService(workflow_service=get_workflow_service())
