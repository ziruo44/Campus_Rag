"""Sliding-window memory compaction helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm.model import model as default_chat_model
from llm.prompt_loader import get_memory_compaction_prompt
from memory.models import ConversationTurn, ThreadDocument, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompactionSlices:
    """Partition a thread into compacted history and active turns."""

    compacted_turns: list[ConversationTurn]
    active_turns: list[ConversationTurn]


def split_turns_for_context(
    turns: list[ConversationTurn],
    active_window_size: int,
    compacted_turn_count: int = 0,
) -> CompactionSlices:
    """Return the compacted prefix and the active post-compaction segment."""
    del active_window_size
    active_start = min(max(compacted_turn_count, 0), len(turns))
    active_turns = list(turns[active_start:])
    compacted_turns = list(turns[:active_start])
    return CompactionSlices(
        compacted_turns=compacted_turns,
        active_turns=active_turns,
    )


def refresh_thread_context_summary(
    document: ThreadDocument,
    *,
    chat_model=default_chat_model,
    notice_callback=None,
) -> None:
    """Compact full segments of terminal turns into the rolling summary."""
    changed = False

    if document.context_compacted_turn_count > len(document.turns):
        document.context_compacted_turn_count = len(document.turns)
        changed = True

    while True:
        active_turns = document.turns[document.context_compacted_turn_count :]
        compaction_size = _next_compaction_size(
            active_turns=active_turns,
            active_window_size=document.max_turns,
        )
        if compaction_size == 0:
            break

        turns_to_compact = active_turns[:compaction_size]
        next_compacted_turn_count = (
            document.context_compacted_turn_count + compaction_size
        )
        if notice_callback is not None:
            notice_callback(
                {
                    "type": "context_compacted",
                    "compacted_delta": compaction_size,
                    "compacted_turn_count": next_compacted_turn_count,
                }
            )
        document.context_summary = _generate_summary(
            compacted_turns=turns_to_compact,
            previous_summary=document.context_summary,
            chat_model=chat_model,
        )
        document.context_compacted_turn_count += compaction_size
        document.context_summary_updated_at = utc_now_iso()
        changed = True

    if (
        not document.turns
        and (
            document.context_summary
            or document.context_compacted_turn_count != 0
        )
    ):
        document.context_summary = ""
        document.context_compacted_turn_count = 0
        document.context_summary_updated_at = utc_now_iso()
        changed = True

    if changed and not document.context_summary_updated_at:
        document.context_summary_updated_at = utc_now_iso()


def build_context_summary_fallback(compacted_turns: list[ConversationTurn]) -> str:
    """Build a bounded plaintext summary when model summarization is unavailable."""
    lines = [
        "Older conversation summary:",
    ]
    for turn in compacted_turns[-8:]:
        if turn.user_message and turn.user_message.content:
            lines.append(f"User: {_clip(turn.user_message.content)}")
        if turn.assistant_message and turn.assistant_message.content:
            lines.append(f"Assistant: {_clip(turn.assistant_message.content)}")
        if turn.state == "failed" and turn.error:
            lines.append(f"Turn error: {_clip(turn.error)}")
    return "\n".join(lines)


def merge_summary_with_fallback(
    *,
    previous_summary: str,
    compacted_turns: list[ConversationTurn],
) -> str:
    """Merge the previous summary with the next compacted segment deterministically."""
    next_segment = build_context_summary_fallback(compacted_turns)
    if previous_summary.strip():
        return "\n\n".join(
            [
                "Previously compacted context:",
                previous_summary.strip(),
                next_segment,
            ]
        )
    return next_segment


def _generate_summary(
    *,
    compacted_turns: list[ConversationTurn],
    previous_summary: str,
    chat_model,
) -> str:
    transcript = _render_turns(compacted_turns)
    prompt = get_memory_compaction_prompt().format(
        previous_summary=previous_summary.strip() or "(none)",
        transcript=transcript,
    )

    try:
        response = chat_model.invoke(
            [
                SystemMessage(
                    content="You summarize older conversation state for a coding and RAG assistant."
                ),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content if item)
        normalized = str(content).strip()
        if normalized:
            return normalized
    except Exception as exc:
        logger.warning("Failed to summarize compacted thread context: %s", exc)
    return merge_summary_with_fallback(
        previous_summary=previous_summary,
        compacted_turns=compacted_turns,
    )


def _render_turns(turns: list[ConversationTurn]) -> str:
    lines: list[str] = []
    for turn in turns:
        lines.append(f"Turn ID: {turn.turn_id}")
        lines.append(f"State: {turn.state}")
        if turn.user_message and turn.user_message.content:
            lines.append(f"User: {turn.user_message.content}")
        if turn.assistant_message and turn.assistant_message.content:
            lines.append(f"Assistant: {turn.assistant_message.content}")
        if turn.state == "failed" and turn.error:
            lines.append(f"Error: {turn.error}")
        if turn.artifacts:
            lines.append(f"Artifacts: {turn.artifacts}")
        lines.append("")
    return "\n".join(lines).strip()


def _clip(value: str, limit: int = 180) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."


def _next_compaction_size(
    *,
    active_turns: list[ConversationTurn],
    active_window_size: int,
) -> int:
    if active_window_size <= 0:
        return len(active_turns)

    terminal_positions = [
        index for index, turn in enumerate(active_turns) if turn.state in {"completed", "failed"}
    ]
    if len(terminal_positions) < active_window_size:
        return 0

    threshold_position = terminal_positions[active_window_size - 1]
    if threshold_position == len(active_turns) - 1 and len(terminal_positions) == active_window_size:
        return 0
    return threshold_position + 1
