"""Tests for the navigation human-review middleware."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

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


def test_navigation_review_merges_user_supplement_and_replays_tool_call() -> None:
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
    replay_message = result["messages"][0]
    assert isinstance(replay_message, AIMessage)
    assert replay_message.tool_calls[0]["args"] == {
        "start_location": "北门",
        "end_location": "图书馆",
    }


def test_navigation_review_allows_confirmed_tool_call_with_edited_start() -> None:
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
    replay_message = result["messages"][0]
    assert isinstance(replay_message, AIMessage)
    assert replay_message.tool_calls[0]["args"] == {
        "start_location": "南门",
        "end_location": "图书馆",
    }


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
