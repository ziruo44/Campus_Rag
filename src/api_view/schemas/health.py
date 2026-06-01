"""健康检查 schema。"""

from __future__ import annotations

from pydantic import BaseModel


class ModelProviderHealthResponse(BaseModel):
    """模型提供方健康状态。"""

    configured: bool
    checked: bool
    reachable: bool | None
    model: str | None
    base_url: str | None
    detail: str | None = None


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    raw_data_exists: bool
    vector_index_exists: bool
    memory_dir_accessible: bool
    runtime_initialized: bool
    model_provider: ModelProviderHealthResponse
