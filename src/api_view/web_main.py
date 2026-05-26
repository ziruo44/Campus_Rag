"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.workflows.service import RuntimeUnavailableError
from api_view.api.chat import router as chat_router
from api_view.api.health import router as health_router
from api_view.services.chat_service import (
    ChatExecutionError,
    ThreadNotFoundError,
    TurnNotFoundError,
)
from app_bootstrap import get_knowledge_runtime
from memory.locks import LockTimeoutError


def create_app(*, prewarm_runtime: bool = True) -> FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if prewarm_runtime:
            get_knowledge_runtime().ensure_initialized()
        yield

    app = FastAPI(title="RAG Agent API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ThreadNotFoundError)
    async def handle_thread_not_found(
        request: Request,
        exc: ThreadNotFoundError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(TurnNotFoundError)
    async def handle_turn_not_found(
        request: Request,
        exc: TurnNotFoundError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(RuntimeUnavailableError)
    async def handle_runtime_unavailable(
        request: Request,
        exc: RuntimeUnavailableError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ChatExecutionError)
    async def handle_chat_execution_error(
        request: Request,
        exc: ChatExecutionError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(LockTimeoutError)
    async def handle_lock_timeout(
        request: Request,
        exc: LockTimeoutError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=503,
            content={"detail": f"Memory store is busy: {exc}"},
        )

    app.include_router(health_router)
    app.include_router(chat_router)
    return app


app = create_app()
