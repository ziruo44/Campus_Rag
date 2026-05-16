"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi import Query

from rag_agent.api.deps import get_agent_runtime
from rag_agent.api.schemas import HealthResponse, ModelProviderHealthResponse
from rag_agent.api.services.agent_runtime import AgentRuntime
from rag_agent.memory_session.config import MemorySettings
from rag_agent.utils.path import get_data_dir, get_raw_data_dir

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    check_model: bool = Query(
        default=False,
        description="Run a live upstream model-provider connectivity probe.",
    ),
    runtime: AgentRuntime = Depends(get_agent_runtime),
) -> HealthResponse:
    """Return application health and optional upstream model connectivity."""
    memory_settings = MemorySettings()
    vector_index_path = get_data_dir() / "vector_index"
    raw_data_exists = get_raw_data_dir().exists()
    vector_index_exists = vector_index_path.exists()
    memory_dir_accessible = memory_settings.session_dir.parent.exists()
    model_provider = runtime.probe_model_provider(check_connection=check_model)

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
        runtime_initialized=runtime.is_initialized,
        model_provider=ModelProviderHealthResponse(
            configured=model_provider.configured,
            checked=model_provider.checked,
            reachable=model_provider.reachable,
            model=model_provider.model,
            base_url=model_provider.base_url,
            detail=model_provider.detail,
        ),
    )
