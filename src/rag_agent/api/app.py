"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_agent.api.routes.chat import router as chat_router
from rag_agent.api.routes.health import router as health_router
from rag_agent.api.services.agent_runtime import RuntimeUnavailableError
from rag_agent.api.services.chat_service import (
    ChatExecutionError,
    ThreadNotFoundError,
    TurnNotFoundError,
)
from rag_agent.memory_session.locks import LockTimeoutError


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="RAG Agent API", version="0.1.0")

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
