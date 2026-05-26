"""Tests for the campus navigation tool."""

from __future__ import annotations

from agent.tools import make_campus_navigation_tool
from agent.tools.campus_navigation_tool import plan_campus_route


def test_plan_campus_route_supports_fuzzy_location_match() -> None:
    result = plan_campus_route("北门", "图书馆")

    assert result["status"] == "ok"
    assert result["resolved_start"] == "温州商学院北校区北门"
    assert result["resolved_end"] == "博闻楼和图书馆"
    assert result["path"][0] == "温州商学院北校区北门"
    assert result["path"][-1] == "博闻楼和图书馆"
    assert "博闻楼和图书馆" in result["key0"]


def test_plan_campus_route_returns_location_failure_for_unknown_place() -> None:
    result = plan_campus_route("火星基地", "图书馆")

    assert result["status"] == "location_not_found"
    assert result["key0"] == "地点识别失败"
    assert result["resolved_start"] is None
    assert result["resolved_end"] == "博闻楼和图书馆"
    assert result["path"] == []


def test_campus_navigation_tool_exposes_structured_result() -> None:
    tool_callable = make_campus_navigation_tool()

    output = tool_callable.invoke(
        {"start_location": "思源楼", "end_location": "南门"}
    )

    assert tool_callable.args_schema is not None
    assert output["status"] == "ok"
    assert output["resolved_start"] == "思源楼"
    assert output["resolved_end"] == "温州商学院南校区南门"
    assert output["path"][0] == "思源楼"
    assert output["path"][-1] == "温州商学院南校区南门"


def test_campus_navigation_tool_accepts_partial_args_for_hitl_flow() -> None:
    tool_callable = make_campus_navigation_tool()

    output = tool_callable.invoke({"end_location": "图书馆"})

    assert output["status"] == "location_not_found"
    assert output["start_input"] == ""
    assert output["end_input"] == "图书馆"
