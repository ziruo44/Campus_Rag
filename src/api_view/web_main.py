"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api_view.errors import register_exception_handlers
from api_view.routers import (
    campus_chat_router,
    campus_health_router,
    campus_threads_router,
)
from app_bootstrap import get_life_guide_runtime, get_major_knowledge_runtime
from shared.logging_setup import configure_logging


def create_app(*, prewarm_runtime: bool = True) -> FastAPI:
    """Create the FastAPI application."""
    configure_logging()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if prewarm_runtime:
            get_major_knowledge_runtime().ensure_initialized()
            get_life_guide_runtime().ensure_initialized()
        yield

    app = FastAPI(title="RAG Agent API", version="0.2.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(campus_health_router)
    app.include_router(campus_chat_router)
    app.include_router(campus_threads_router)
    return app


app = create_app()
