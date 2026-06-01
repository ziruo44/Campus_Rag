"""健康检查路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api_view.dependencies import get_health_probe, get_health_runtime
from api_view.schemas import HealthResponse, ModelProviderHealthResponse
from domain.major_knowledge.runtime import KnowledgeRuntime
from llm.health import ModelProviderHealthProbe
from memory.config import MemorySettings
from utils.paths import get_chroma_db_dir, get_raw_data_dir

campus_router = APIRouter(prefix="/campus", tags=["campus"])


def _build_health_response(
    *,
    check_model: bool,
    knowledge_runtime: KnowledgeRuntime,
    model_health_probe: ModelProviderHealthProbe,
) -> HealthResponse:
    memory_settings = MemorySettings()
    vector_index_path = get_chroma_db_dir()
    raw_data_exists = get_raw_data_dir().exists()
    vector_index_exists = vector_index_path.exists()
    memory_dir_accessible = memory_settings.session_dir.parent.exists()
    model_provider = model_health_probe.probe(check_connection=check_model)

    status = "ok"
    if not raw_data_exists or not vector_index_exists or not memory_dir_accessible:
        status = "degraded"
    if check_model and not model_provider.reachable:
        status = "degraded"

    return HealthResponse(
        status=status,
        raw_data_exists=raw_data_exists,
        vector_index_exists=vector_index_exists,
        memory_dir_accessible=memory_dir_accessible,
        runtime_initialized=knowledge_runtime.is_initialized,
        model_provider=ModelProviderHealthResponse(
            configured=model_provider.configured,
            checked=model_provider.checked,
            reachable=model_provider.reachable,
            model=model_provider.model,
            base_url=model_provider.base_url,
            detail=model_provider.detail,
        ),
    )


@campus_router.get("/health", response_model=HealthResponse)
def health(
    check_model: bool = Query(
        default=False,
        description="Run a live upstream model-provider connectivity probe.",
    ),
    knowledge_runtime: KnowledgeRuntime = Depends(get_health_runtime),
    model_health_probe: ModelProviderHealthProbe = Depends(get_health_probe),
) -> HealthResponse:
    """返回应用健康状态。"""
    return _build_health_response(
        check_model=check_model,
        knowledge_runtime=knowledge_runtime,
        model_health_probe=model_health_probe,
    )

