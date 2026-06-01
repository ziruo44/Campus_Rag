"""对话应用层数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class ChatTurnResult:
    """单轮对话完成后的结构化结果。"""

    thread_id: str
    answer: str
    messages: list[Any]
    artifacts: dict[str, Any]


@dataclass(slots=True)
class StreamMessageEvent:
    """流式对话事件。"""

    event: Literal["start", "delta", "done", "error"]
    thread_id: str
    turn_id: str
    content: str = ""
    error: str | None = None
