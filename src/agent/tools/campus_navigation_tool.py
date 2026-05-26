"""Agent-facing campus navigation tool."""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_CAMPUS_LOCATIONS = (
    "思源楼",
    "思创楼",
    "思贤楼",
    "致用楼",
    "经世楼",
    "温州商学院北校区北门",
    "博爱楼",
    "商院桥",
    "南院桥",
    "北校区体育场",
    "温州商学院南校区北门",
    "一食堂",
    "习德楼和立德楼",
    "文博和水心生活区",
    "二食堂",
    "博雅楼",
    "励行桥",
    "竞进桥",
    "博闻楼和图书馆",
    "三食堂",
    "德涵生活区",
    "超豪和罗山生活区",
    "综合体",
    "体育馆",
    "南校区操场",
    "温州商学院南校区南门",
)

_CAMPUS_EDGES = (
    ("思源楼", "思创楼"),
    ("思源楼", "致用楼"),
    ("思源楼", "经世楼"),
    ("思创楼", "思贤楼"),
    ("思创楼", "致用楼"),
    ("思创楼", "温州商学院北校区北门"),
    ("思贤楼", "温州商学院北校区北门"),
    ("致用楼", "温州商学院北校区北门"),
    ("致用楼", "经世楼"),
    ("温州商学院北校区北门", "博爱楼"),
    ("经世楼", "博爱楼"),
    ("经世楼", "商院桥"),
    ("博爱楼", "商院桥"),
    ("南院桥", "北校区体育场"),
    ("商院桥", "温州商学院南校区北门"),
    ("温州商学院南校区北门", "一食堂"),
    ("温州商学院南校区北门", "习德楼和立德楼"),
    ("习德楼和立德楼", "博雅楼"),
    ("一食堂", "文博和水心生活区"),
    ("文博和水心生活区", "博雅楼"),
    ("文博和水心生活区", "二食堂"),
    ("博雅楼", "励行桥"),
    ("博雅楼", "竞进桥"),
    ("二食堂", "竞进桥"),
    ("励行桥", "博闻楼和图书馆"),
    ("励行桥", "德涵生活区"),
    ("竞进桥", "博闻楼和图书馆"),
    ("竞进桥", "体育馆"),
    ("博闻楼和图书馆", "三食堂"),
    ("博闻楼和图书馆", "德涵生活区"),
    ("德涵生活区", "超豪和罗山生活区"),
    ("三食堂", "综合体"),
    ("体育馆", "南校区操场"),
    ("南校区操场", "综合体"),
    ("综合体", "超豪和罗山生活区"),
    ("超豪和罗山生活区", "温州商学院南校区南门"),
)


class CampusNavigationInput(BaseModel):
    start_location: str = Field(
        default="",
        description=(
            "用户当前所在地点，例如北门、图书馆、思源楼。"
            "如果起点暂时未知，可以留空字符串。"
        ),
    )
    end_location: str = Field(
        default="",
        description=(
            "用户想去的目标地点，例如图书馆、食堂、南门。"
            "如果终点暂时未知，可以留空字符串。"
        ),
    )


def _build_graph() -> dict[str, list[str]]:
    graph = {location: [] for location in _CAMPUS_LOCATIONS}
    for start, end in _CAMPUS_EDGES:
        graph[start].append(end)
        graph[end].append(start)
    return graph


_CAMPUS_GRAPH = _build_graph()


def _edit_distance(left: str, right: str) -> int:
    left_length = len(left)
    right_length = len(right)
    dp = [[0] * (right_length + 1) for _ in range(left_length + 1)]
    for index in range(left_length + 1):
        dp[index][0] = index
    for index in range(right_length + 1):
        dp[0][index] = index
    for left_index in range(1, left_length + 1):
        for right_index in range(1, right_length + 1):
            cost = 0 if left[left_index - 1] == right[right_index - 1] else 1
            dp[left_index][right_index] = min(
                dp[left_index - 1][right_index] + 1,
                dp[left_index][right_index - 1] + 1,
                dp[left_index - 1][right_index - 1] + cost,
            )
    return dp[left_length][right_length]


def _resolve_location_name(user_input: str) -> tuple[str | None, bool]:
    normalized_input = user_input.strip()
    if not normalized_input:
        return None, False
    if normalized_input in _CAMPUS_LOCATIONS:
        return normalized_input, True

    best_match: str | None = None
    min_distance = float("inf")
    for location in _CAMPUS_LOCATIONS:
        if normalized_input in location:
            return location, True
        distance = _edit_distance(normalized_input, location)
        threshold = max(len(normalized_input), len(location)) * 0.6
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            best_match = location
    return (best_match, True) if best_match else (None, False)


def _find_shortest_path(start_location: str, end_location: str) -> list[str] | None:
    queue: deque[list[str]] = deque([[start_location]])
    visited: set[str] = set()
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current in visited:
            continue
        visited.add(current)
        if current == end_location:
            return path
        for neighbor in _CAMPUS_GRAPH.get(current, []):
            if neighbor not in visited:
                queue.append([*path, neighbor])
    return None


def plan_campus_route(start_location: str, end_location: str) -> dict[str, Any]:
    """Resolve two campus locations and compute a shortest navigation route."""
    resolved_start, start_found = _resolve_location_name(start_location)
    resolved_end, end_found = _resolve_location_name(end_location)

    if not start_found or not end_found:
        return {
            "status": "location_not_found",
            "key0": "地点识别失败",
            "start_input": start_location,
            "end_input": end_location,
            "resolved_start": resolved_start,
            "resolved_end": resolved_end,
            "path": [],
            "path_text": "",
        }

    path = _find_shortest_path(resolved_start, resolved_end)
    if not path:
        return {
            "status": "route_not_found",
            "key0": "无法连通",
            "start_input": start_location,
            "end_input": end_location,
            "resolved_start": resolved_start,
            "resolved_end": resolved_end,
            "path": [],
            "path_text": "",
        }

    path_text = " -> ".join(path)
    return {
        "status": "ok",
        "key0": path_text,
        "start_input": start_location,
        "end_input": end_location,
        "resolved_start": resolved_start,
        "resolved_end": resolved_end,
        "path": path,
        "path_text": path_text,
    }


def make_campus_navigation_tool():
    @tool(args_schema=CampusNavigationInput)
    def campus_navigation_tool(
        start_location: str = "",
        end_location: str = "",
    ) -> dict[str, Any]:
        """Find a shortest campus route between two Wenzhou Business College locations."""
        return plan_campus_route(
            start_location=start_location,
            end_location=end_location,
        )

    return campus_navigation_tool
