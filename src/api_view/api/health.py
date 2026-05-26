"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api_view.web_config import HealthResponse, ModelProviderHealthResponse
from app_bootstrap import get_knowledge_runtime, get_model_health_probe
from domain.knowledge.runtime import KnowledgeRuntime
from llm.health import ModelProviderHealthProbe
from memory.config import MemorySettings
from utils.paths import get_data_dir, get_raw_data_dir

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    check_model: bool = Query(
        default=False,
        description="Run a live upstream model-provider connectivity probe.",
    ),
    knowledge_runtime: KnowledgeRuntime = Depends(get_knowledge_runtime),
    model_health_probe: ModelProviderHealthProbe = Depends(get_model_health_probe),
) -> HealthResponse:
    """Return application health and optional upstream model connectivity."""
    memory_settings = MemorySettings()
    vector_index_path = get_data_dir() / "vector_index"
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
