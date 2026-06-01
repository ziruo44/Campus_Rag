"""API 依赖装配。"""

from __future__ import annotations

from fastapi import Depends

from application.chat import (
    DeleteThreadUseCase,
    DeleteTurnUseCase,
    GetThreadUseCase,
    ListThreadsUseCase,
    SendMessageUseCase,
    StreamMessageUseCase,
)
from app_bootstrap import (
    get_chat_service,
    get_major_knowledge_runtime,
    get_model_health_probe,
)
from api_view.services.chat_service import ChatService
from domain.major_knowledge.runtime import KnowledgeRuntime
from llm.health import ModelProviderHealthProbe


def get_send_message_use_case(
    chat_service: ChatService = Depends(get_chat_service),
) -> SendMessageUseCase:
    """返回发送消息用例。"""
    return chat_service.send_message_use_case


def get_stream_message_use_case(
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamMessageUseCase:
    """返回流式消息用例。"""
    return chat_service.stream_message_use_case


def get_list_threads_use_case(
    chat_service: ChatService = Depends(get_chat_service),
) -> ListThreadsUseCase:
    """返回线程列表用例。"""
    return chat_service.list_threads_use_case


def get_get_thread_use_case(
    chat_service: ChatService = Depends(get_chat_service),
) -> GetThreadUseCase:
    """返回线程详情用例。"""
    return chat_service.get_thread_use_case


def get_delete_thread_use_case(
    chat_service: ChatService = Depends(get_chat_service),
) -> DeleteThreadUseCase:
    """返回删除线程用例。"""
    return chat_service.delete_thread_use_case


def get_delete_turn_use_case(
    chat_service: ChatService = Depends(get_chat_service),
) -> DeleteTurnUseCase:
    """返回删除轮次用例。"""
    return chat_service.delete_turn_use_case


def get_health_runtime(
    knowledge_runtime: KnowledgeRuntime = Depends(get_major_knowledge_runtime),
) -> KnowledgeRuntime:
    """返回健康检查所用的专业知识运行时。"""
    return knowledge_runtime


def get_health_probe(
    model_health_probe: ModelProviderHealthProbe = Depends(get_model_health_probe),
) -> ModelProviderHealthProbe:
    """返回健康检查所用模型探针。"""
    return model_health_probe
