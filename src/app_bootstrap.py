"""应用启动工厂。"""

from __future__ import annotations

from functools import lru_cache

from agent.workflows.life_guide_service import LifeGuideWorkflowService
from agent.workflows.service import MajorKnowledgeWorkflowService
from api_view.services.chat_service import ChatService
from domain.life_guide_knowledge.runtime import LifeGuideKnowledgeRuntime
from domain.major_knowledge.runtime import KnowledgeRuntime
from llm.health import ModelProviderHealthProbe


@lru_cache(maxsize=1)
def get_major_knowledge_runtime() -> KnowledgeRuntime:
    """返回共享的专业知识库运行时。"""
    return KnowledgeRuntime()


@lru_cache(maxsize=1)
def get_life_guide_runtime() -> LifeGuideKnowledgeRuntime:
    """返回共享的生活指南运行时。"""
    return LifeGuideKnowledgeRuntime()


@lru_cache(maxsize=1)
def get_model_health_probe() -> ModelProviderHealthProbe:
    """返回共享的模型健康探针。"""
    return ModelProviderHealthProbe()


@lru_cache(maxsize=1)
def get_major_workflow_service() -> MajorKnowledgeWorkflowService:
    """返回共享的专业知识工作流服务。"""
    return MajorKnowledgeWorkflowService(
        knowledge_runtime=get_major_knowledge_runtime(),
        health_probe=get_model_health_probe(),
    )


@lru_cache(maxsize=1)
def get_life_guide_workflow_service() -> LifeGuideWorkflowService:
    """返回共享的生活指南工作流服务。"""
    return LifeGuideWorkflowService(
        knowledge_runtime=get_life_guide_runtime(),
        health_probe=get_model_health_probe(),
    )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """返回共享的对话服务。"""
    return ChatService(
        major_workflow_service=get_major_workflow_service(),
        life_guide_workflow_service=get_life_guide_workflow_service(),
    )


def get_knowledge_runtime() -> KnowledgeRuntime:
    """兼容旧依赖名，返回专业知识库运行时。"""
    return get_major_knowledge_runtime()


def get_workflow_service() -> MajorKnowledgeWorkflowService:
    """兼容旧依赖名，返回专业知识工作流服务。"""
    return get_major_workflow_service()
