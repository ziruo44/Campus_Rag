"""导航人工确认中间件测试。"""

from __future__ import annotations

from unittest.mock import Mock

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.middleware import NavigationHumanReviewMiddleware


def build_navigation_tool_call(
    *,
    start_location: str = "",
    end_location: str = "",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "campus_navigation_tool",
                "args": {
                    "start_location": start_location,
                    "end_location": end_location,
                },
                "id": "call_navigation",
                "type": "tool_call",
            }
        ],
    )


def test_navigation_review_blocks_initial_tool_call_with_confirmation_prompt() -> None:
    middleware = NavigationHumanReviewMiddleware()
    state = {
        "messages": [
            HumanMessage(content="从北门去图书馆怎么走"),
            build_navigation_tool_call(start_location="北门", end_location="图书馆"),
        ]
    }

    result = middleware.after_model(state, runtime=None)

    assert result is not None
    prompt = result["messages"][0]
    assert isinstance(prompt, AIMessage)
    assert "导航工具需要你确认后再执行" in prompt.content
    assert "起点：北门" in prompt.content
    assert "终点：图书馆" in prompt.content


def test_navigation_review_requests_only_missing_start_location() -> None:
    middleware = NavigationHumanReviewMiddleware()
    state = {
        "messages": [
            HumanMessage(content="图书馆怎么走"),
            build_navigation_tool_call(end_location="图书馆"),
        ]
    }

    result = middleware.after_model(state, runtime=None)

    assert result is not None
    prompt = result["messages"][0]
    assert isinstance(prompt, AIMessage)
    assert "导航信息还不完整" in prompt.content
    assert "起点：待补充" in prompt.content
    assert "终点：图书馆" in prompt.content
    assert "我还缺少起点" in prompt.content


def test_navigation_review_merges_user_supplement_into_confirmation_prompt() -> None:
    middleware = NavigationHumanReviewMiddleware()
    state = {
        "messages": [
            HumanMessage(content="图书馆怎么走"),
            AIMessage(
                content=(
                    "导航信息还不完整。\n"
                    "起点：待补充\n"
                    "终点：图书馆\n\n"
                    "我还缺少起点，请直接补充。\n"
                    "例如：\n"
                    "1. 起点是北门\n"
                    "2. 从北门出发\n"
                    "3. 取消"
                )
            ),
            HumanMessage(content="起点是北门"),
            build_navigation_tool_call(start_location="北门"),
        ]
    }

    result = middleware.after_model(state, runtime=None)

    assert result is not None
    confirmation_message = result["messages"][0]
    assert isinstance(confirmation_message, AIMessage)
    assert "导航工具需要你确认后再执行" in confirmation_message.content
    assert "起点：北门" in confirmation_message.content
    assert "终点：图书馆" in confirmation_message.content


def test_navigation_review_edit_reply_updates_confirmation_instead_of_executing() -> None:
    middleware = NavigationHumanReviewMiddleware()
    state = {
        "messages": [
            HumanMessage(content="从北门去图书馆怎么走"),
            AIMessage(
                content=(
                    "导航工具需要你确认后再执行。\n"
                    "起点：北门\n"
                    "终点：图书馆\n\n"
                    "你可以直接回复以下操作：\n"
                    "1. 确认\n"
                    "2. 修改起点和/或终点，例如：起点改为南门，终点改为图书馆\n"
                    "3. 取消"
                )
            ),
            HumanMessage(content="起点改为南门"),
            build_navigation_tool_call(start_location="南门"),
        ]
    }

    result = middleware.after_model(state, runtime=None)

    assert result is not None
    confirmation_message = result["messages"][0]
    assert isinstance(confirmation_message, AIMessage)
    assert "导航工具需要你确认后再执行" in confirmation_message.content
    assert "起点：南门" in confirmation_message.content
    assert "终点：图书馆" in confirmation_message.content


def test_navigation_review_converts_rejected_tool_call_into_cancellation_message() -> None:
    middleware = NavigationHumanReviewMiddleware()
    state = {
        "messages": [
            HumanMessage(content="从北门去图书馆怎么走"),
            AIMessage(
                content=(
                    "导航工具需要你确认后再执行。\n"
                    "起点：北门\n"
                    "终点：图书馆\n\n"
                    "你可以直接回复以下操作：\n"
                    "1. 确认\n"
                    "2. 修改起点和/或终点，例如：起点改为南门，终点改为图书馆\n"
                    "3. 取消"
                )
            ),
            HumanMessage(content="取消"),
            build_navigation_tool_call(start_location="北门", end_location="图书馆"),
        ]
    }

    result = middleware.after_model(state, runtime=None)

    assert result is not None
    response = result["messages"][0]
    assert isinstance(response, AIMessage)
    assert "已取消本次校园导航" in response.content


def test_pending_navigation_review_short_circuits_model_for_missing_info_follow_up() -> None:
    middleware = NavigationHumanReviewMiddleware()
    request = ModelRequest(
        model=Mock(),
        messages=[
            HumanMessage(content="图书馆怎么走"),
            AIMessage(
                content=(
                    "导航信息还不完整。\n"
                    "起点：待补充\n"
                    "终点：图书馆\n\n"
                    "我还缺少起点，请直接补充。"
                )
            ),
            HumanMessage(content="起点是北门"),
        ],
        state={"messages": []},
        runtime=None,
    )
    handler = Mock()

    result = middleware.wrap_model_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, AIMessage)
    assert "导航工具需要你确认后再执行" in result.content
    assert "起点：北门" in result.content
    assert "终点：图书馆" in result.content


def test_pending_navigation_review_short_circuits_model_for_confirm_reply() -> None:
    middleware = NavigationHumanReviewMiddleware()
    request = ModelRequest(
        model=Mock(),
        messages=[
            HumanMessage(content="从北门去图书馆怎么走"),
            AIMessage(
                content=(
                    "导航工具需要你确认后再执行。\n"
                    "起点：北门\n"
                    "终点：图书馆\n\n"
                    "你可以直接回复以下操作：\n"
                    "1. 确认\n"
                    "2. 修改起点和/或终点，例如：起点改为南门，终点改为图书馆\n"
                    "3. 取消"
                )
            ),
            HumanMessage(content="确认"),
        ],
        state={"messages": []},
        runtime=None,
    )
    handler = Mock()

    result = middleware.wrap_model_call(request, handler)

    handler.assert_not_called()
    assert isinstance(result, AIMessage)
    assert result.tool_calls[0]["args"] == {
        "start_location": "北门",
        "end_location": "图书馆",
    }


def test_pending_navigation_review_does_not_short_circuit_after_tool_output() -> None:
    middleware = NavigationHumanReviewMiddleware()
    request = ModelRequest(
        model=Mock(),
        messages=[
            HumanMessage(content="从北门去图书馆怎么走"),
            AIMessage(
                content=(
                    "导航工具需要你确认后再执行。\n"
                    "起点：北门\n"
                    "终点：图书馆\n\n"
                    "你可以直接回复以下操作：\n"
                    "1. 确认\n"
                    "2. 修改起点和/或终点，例如：起点改为南门，终点改为图书馆\n"
                    "3. 取消"
                )
            ),
            HumanMessage(content="确认"),
            ToolMessage(content="导航工具输出", tool_call_id="call_navigation_review"),
        ],
        state={"messages": []},
        runtime=None,
    )
    handler = Mock(return_value=AIMessage(content="最终导航说明"))

    result = middleware.wrap_model_call(request, handler)

    handler.assert_called_once()
    assert isinstance(result, AIMessage)
    assert result.content == "最终导航说明"
