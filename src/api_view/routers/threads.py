"""线程路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from application.chat import (
    DeleteThreadUseCase,
    DeleteTurnUseCase,
    GetThreadUseCase,
    ListThreadsUseCase,
)
from api_view.dependencies import (
    get_delete_thread_use_case,
    get_delete_turn_use_case,
    get_get_thread_use_case,
    get_list_threads_use_case,
)
from api_view.schemas import ThreadListItemResponse, ThreadResponse

campus_router = APIRouter(prefix="/campus", tags=["campus"])


@campus_router.get("/threads", response_model=list[ThreadListItemResponse])
def list_threads(
    use_case: ListThreadsUseCase = Depends(get_list_threads_use_case),
) -> list[ThreadListItemResponse]:
    """列出线程摘要。"""
    return [ThreadListItemResponse.model_validate(item) for item in use_case.execute()]


@campus_router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: str,
    use_case: GetThreadUseCase = Depends(get_get_thread_use_case),
) -> ThreadResponse:
    """获取线程详情。"""
    return ThreadResponse.model_validate(use_case.execute(thread_id=thread_id))


@campus_router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: str,
    use_case: DeleteThreadUseCase = Depends(get_delete_thread_use_case),
) -> Response:
    """删除整条线程。"""
    use_case.execute(thread_id=thread_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@campus_router.delete("/threads/{thread_id}/turns/{turn_id}", response_model=ThreadResponse)
def delete_turn(
    thread_id: str,
    turn_id: str,
    use_case: DeleteTurnUseCase = Depends(get_delete_turn_use_case),
) -> ThreadResponse:
    """删除线程中的一轮问答。"""
    return ThreadResponse.model_validate(
        use_case.execute(thread_id=thread_id, turn_id=turn_id)
    )

