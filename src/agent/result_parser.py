"""Parse raw framework-agent output into stable application results."""

from __future__ import annotations

import ast
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def build_knowledge_turn_result(messages: list[Any]) -> dict[str, Any]:
    """Build the final answer and persisted artifacts from raw agent messages."""
    workflow_result = extract_workflow_result(messages)
    return {
        "answer": extract_final_answer(messages),
        "messages": list(messages),
        "artifacts": build_knowledge_artifacts(workflow_result),
        "capability_type": "knowledge",
    }


def build_knowledge_artifacts(
    workflow_result: dict[str, Any] | None,
    notices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build persisted knowledge artifacts from one workflow result."""
    if workflow_result is None:
        return {"notices": list(notices or [])} if notices else {}
    artifacts = {
        "capability_type": "knowledge",
        "workflow_summary": {
            "resolved_queries": workflow_result.get("resolved_queries", []),
            "route_trace": workflow_result.get("route_trace", []),
            "workflow_trace": workflow_result.get("workflow_trace", []),
        },
        "evidence_bundle": workflow_result.get("evidence_bundle", []),
    }
    if notices:
        artifacts["notices"] = list(notices)
    return artifacts


def extract_workflow_result(messages: list[Any]) -> dict[str, Any] | None:
    """Extract one workflow result from tool messages."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        parsed = parse_tool_message_content(message.content)
        if isinstance(parsed, dict) and "retrieval_context" in parsed:
            return parsed
    return None


def extract_final_answer(messages: list[Any]) -> str:
    """Extract the last non-empty AI message as the final answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = extract_message_text(message)
            if text:
                return text
    return ""


def parse_tool_message_content(content: Any) -> Any:
    """Parse structured tool content from LangChain tool messages."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        normalized = content.strip()
        if not normalized:
            return None
        try:
            return ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            return None
    return None


def extract_message_text(message: Any) -> str:
    """Best-effort text extraction from LangChain message content."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip() if content else ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                parts.append(normalized)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()
