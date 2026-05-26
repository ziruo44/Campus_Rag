"""Human-review middleware for campus navigation without a graph checkpointer."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall

ReviewAction = Literal["approve", "edit", "reject"]

_NAVIGATION_TOOL_NAME = "campus_navigation_tool"
_CONFIRM_PROMPT_HEADER = "导航工具需要你确认后再执行。"
_MISSING_PROMPT_HEADER = "导航信息还不完整。"
_UNKNOWN_LOCATION = "待补充"

_APPROVE_KEYWORDS = (
    "确认",
    "可以",
    "好的",
    "好",
    "是的",
    "没问题",
    "对",
    "继续",
    "开始导航",
)
_REJECT_KEYWORDS = (
    "取消",
    "不用了",
    "算了",
    "拒绝",
    "不需要",
    "不要查了",
)
_EDIT_KEYWORDS = (
    "改",
    "修改",
    "起点",
    "终点",
    "从",
    "到",
    "出发",
    "去",
    "我在",
)

_ROUTE_PATTERNS = (
    re.compile(
        r"我在(?P<start>.+?)[，, ]*(?:想去|去|到)(?P<end>.+?)(?:怎么走|怎么去|怎么到|路线)?$"
    ),
    re.compile(r"从(?P<start>.+?)(?:到|去)(?P<end>.+?)(?:怎么走|怎么去|怎么到|路线)?$"),
    re.compile(r"(?P<start>.+?)到(?P<end>.+?)(?:怎么走|怎么去|怎么到|路线)?$"),
    re.compile(r"(?:去|到)(?P<end>.+?)(?:怎么走|怎么去|怎么到|路线)?$"),
    re.compile(r"(?P<end>.+?)(?:怎么走|怎么去|怎么到|路线)$"),
)


@dataclass(slots=True, frozen=True)
class ToolHumanReviewConfig:
    """Configuration for one tool that requires explicit human review."""

    allowed_actions: tuple[ReviewAction, ...] = ("approve", "edit", "reject")


def _extract_message_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _normalize_location_value(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    normalized = normalized.strip("，。！？；：,.!?;: ")
    if normalized == _UNKNOWN_LOCATION:
        return ""
    return normalized


def _find_last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )


def _find_last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )


def _find_latest_navigation_prompt(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _extract_message_text(message)
        if text.startswith(_CONFIRM_PROMPT_HEADER) or text.startswith(_MISSING_PROMPT_HEADER):
            return text
    return ""


def _classify_review_reply(user_reply: str) -> ReviewAction | None:
    normalized_reply = user_reply.strip()
    if not normalized_reply:
        return None
    if any(keyword in normalized_reply for keyword in _REJECT_KEYWORDS):
        return "reject"
    if any(keyword in normalized_reply for keyword in _APPROVE_KEYWORDS):
        return "approve"
    if any(keyword in normalized_reply for keyword in _EDIT_KEYWORDS):
        return "edit"
    return None


def _extract_slots_from_prompt(prompt: str) -> dict[str, str]:
    result = {"start_location": "", "end_location": ""}
    for line in prompt.splitlines():
        if line.startswith("起点："):
            result["start_location"] = _normalize_location_value(line.removeprefix("起点："))
        if line.startswith("终点："):
            result["end_location"] = _normalize_location_value(line.removeprefix("终点："))
    return result


def _extract_slots_from_text(text: str) -> dict[str, str]:
    normalized = text.strip()
    if not normalized:
        return {"start_location": "", "end_location": ""}

    explicit_start = re.search(
        r"起点(?:改为|改成|改一下|是|为)?(?P<start>[^，。,；;\n]+)",
        normalized,
    )
    explicit_end = re.search(
        r"终点(?:改为|改成|改一下|是|为)?(?P<end>[^，。,；;\n]+)",
        normalized,
    )
    result = {
        "start_location": _normalize_location_value(
            explicit_start.group("start") if explicit_start else ""
        ),
        "end_location": _normalize_location_value(
            explicit_end.group("end") if explicit_end else ""
        ),
    }
    if result["start_location"] or result["end_location"]:
        return result

    for pattern in _ROUTE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        return {
            "start_location": _normalize_location_value(match.groupdict().get("start", "")),
            "end_location": _normalize_location_value(match.groupdict().get("end", "")),
        }

    return {"start_location": "", "end_location": ""}


def _resolve_tool_args(
    tool_call: ToolCall,
    user_reply: str,
    latest_prompt: str,
) -> dict[str, str]:
    prompt_slots = _extract_slots_from_prompt(latest_prompt)
    tool_args = dict(tool_call.get("args") or {})
    text_slots = _extract_slots_from_text(user_reply)

    return {
        "start_location": (
            text_slots["start_location"]
            or _normalize_location_value(tool_args.get("start_location"))
            or prompt_slots["start_location"]
        ),
        "end_location": (
            text_slots["end_location"]
            or _normalize_location_value(tool_args.get("end_location"))
            or prompt_slots["end_location"]
        ),
    }


def _missing_fields(resolved_args: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if not resolved_args["start_location"]:
        missing.append("起点")
    if not resolved_args["end_location"]:
        missing.append("终点")
    return missing


def _build_confirmation_prompt(
    resolved_args: dict[str, str],
    allowed_actions: tuple[ReviewAction, ...],
) -> str:
    lines = [
        _CONFIRM_PROMPT_HEADER,
        f"起点：{resolved_args['start_location'] or _UNKNOWN_LOCATION}",
        f"终点：{resolved_args['end_location'] or _UNKNOWN_LOCATION}",
        "",
        "你可以直接回复以下操作：",
    ]
    if "approve" in allowed_actions:
        lines.append("1. 确认")
    if "edit" in allowed_actions:
        lines.append("2. 修改起点和/或终点，例如：起点改为南门，终点改为图书馆")
    if "reject" in allowed_actions:
        lines.append("3. 取消")
    return "\n".join(lines)


def _build_missing_info_prompt(
    resolved_args: dict[str, str],
    missing_fields: list[str],
) -> str:
    missing_text = "和".join(missing_fields)
    lines = [
        _MISSING_PROMPT_HEADER,
        f"起点：{resolved_args['start_location'] or _UNKNOWN_LOCATION}",
        f"终点：{resolved_args['end_location'] or _UNKNOWN_LOCATION}",
        "",
        f"我还缺少{missing_text}，请直接补充。",
        "例如：",
    ]
    if "起点" in missing_fields and "终点" in missing_fields:
        lines.append("1. 从北门到图书馆")
        lines.append("2. 起点是北门，终点是图书馆")
    elif "起点" in missing_fields:
        lines.append("1. 起点是北门")
        lines.append("2. 从北门出发")
    else:
        lines.append("1. 终点是图书馆")
        lines.append("2. 去图书馆")
    lines.append("3. 取消")
    return "\n".join(lines)


def _build_navigation_tool_message(tool_call: ToolCall, resolved_args: dict[str, str]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_call["name"],
                "args": resolved_args,
                "id": tool_call["id"],
                "type": tool_call.get("type", "tool_call"),
            }
        ],
    )


class NavigationHumanReviewMiddleware(AgentMiddleware[AgentState, None, Any]):
    """Intercept navigation tool calls and require a human confirmation turn."""

    def __init__(
        self,
        interrupt_on: dict[str, ToolHumanReviewConfig] | None = None,
    ) -> None:
        self.interrupt_on = interrupt_on or {
            _NAVIGATION_TOOL_NAME: ToolHumanReviewConfig(),
        }

    def after_model(
        self,
        state: AgentState,
        runtime,
    ) -> dict[str, Any] | None:
        del runtime
        messages = list(state.get("messages", []))
        if not messages:
            return None

        last_ai_message = _find_last_ai_message(messages)
        if last_ai_message is None or not last_ai_message.tool_calls:
            return None

        message_index = messages.index(last_ai_message)
        prior_messages = messages[:message_index]
        last_human_message = _find_last_human_message(prior_messages)
        last_human_text = _extract_message_text(last_human_message)
        latest_prompt = _find_latest_navigation_prompt(prior_messages)
        has_pending_prompt = bool(latest_prompt)
        review_action = _classify_review_reply(last_human_text)
        if has_pending_prompt and review_action is None and last_human_text:
            review_action = "edit"

        for tool_call in last_ai_message.tool_calls:
            config = self.interrupt_on.get(tool_call["name"])
            if config is None:
                continue

            resolved_args = _resolve_tool_args(
                tool_call=tool_call,
                user_reply=last_human_text,
                latest_prompt=latest_prompt,
            )
            missing_fields = _missing_fields(resolved_args)

            if has_pending_prompt and review_action == "reject":
                return {
                    "messages": [
                        AIMessage(
                            content="已取消本次校园导航。需要重新查询时，直接告诉我新的起点和终点即可。"
                        )
                    ]
                }

            if has_pending_prompt and review_action in {"approve", "edit"}:
                if missing_fields:
                    return {
                        "messages": [
                            AIMessage(
                                content=_build_missing_info_prompt(
                                    resolved_args=resolved_args,
                                    missing_fields=missing_fields,
                                )
                            )
                        ]
                    }
                return {
                    "messages": [
                        _build_navigation_tool_message(
                            tool_call=tool_call,
                            resolved_args=resolved_args,
                        )
                    ]
                }

            if missing_fields:
                return {
                    "messages": [
                        AIMessage(
                            content=_build_missing_info_prompt(
                                resolved_args=resolved_args,
                                missing_fields=missing_fields,
                            )
                        )
                    ]
                }

            return {
                "messages": [
                    AIMessage(
                        content=_build_confirmation_prompt(
                            resolved_args=resolved_args,
                            allowed_actions=config.allowed_actions,
                        )
                    )
                ]
            }

        return None
