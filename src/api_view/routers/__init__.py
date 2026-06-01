"""API 路由导出。"""

from api_view.routers.chat import campus_router as campus_chat_router
from api_view.routers.health import campus_router as campus_health_router
from api_view.routers.threads import campus_router as campus_threads_router

__all__ = [
    "campus_chat_router",
    "campus_health_router",
    "campus_threads_router",
]
