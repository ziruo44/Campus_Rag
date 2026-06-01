"""FastAPI 异常处理。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from application.chat import ChatExecutionError, ThreadNotFoundError, TurnNotFoundError
from domain.life_guide_knowledge.runtime import (
    RuntimeUnavailableError as LifeGuideRuntimeUnavailableError,
)
from domain.major_knowledge.runtime import (
    RuntimeUnavailableError as MajorRuntimeUnavailableError,
)
from memory.locks import LockTimeoutError


def register_exception_handlers(app: FastAPI) -> None:
    """注册 API 统一异常处理。"""

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

    @app.exception_handler(MajorRuntimeUnavailableError)
    @app.exception_handler(LifeGuideRuntimeUnavailableError)
    async def handle_runtime_unavailable(
        request: Request,
        exc: MajorRuntimeUnavailableError | LifeGuideRuntimeUnavailableError,
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
