"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_agent.api.deps import get_agent_runtime
from rag_agent.api.schemas import HealthResponse
from rag_agent.api.services.agent_runtime import AgentRuntime
from rag_agent.memory_session.config import MemorySettings
from rag_agent.utils.path import get_data_dir, get_raw_data_dir

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(runtime: AgentRuntime = Depends(get_agent_runtime)) -> HealthResponse:
    """Return application health without triggering model calls."""
    memory_settings = MemorySettings()
    vector_index_path = get_data_dir() / "vector_index"
    return HealthResponse(
        status="ok",
        raw_data_exists=get_raw_data_dir().exists(),
        vector_index_exists=vector_index_path.exists(),
        memory_dir_accessible=memory_settings.session_dir.parent.exists(),
        runtime_initialized=runtime.is_initialized,
    )
