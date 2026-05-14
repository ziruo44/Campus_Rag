"""Build LangChain chat messages from persisted thread state."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_agent.memory_session.models import ThreadDocument, ThreadReference
from rag_agent.memory_session.session import ManagedThread


def build_agent_messages(
    thread: ManagedThread,
    include_reference_context: bool = True,
) -> list[HumanMessage | AIMessage | SystemMessage]:
    """Build the full message list for the current agent invocation."""
    thread.refresh()
    messages: list[HumanMessage | AIMessage | SystemMessage] = []

    memory_context = _build_memory_context(
        thread=thread,
        include_reference_context=include_reference_context,
    )
    if memory_context:
        messages.append(SystemMessage(content=memory_context))

    for turn in thread.turns:
        if turn.user_message and turn.user_message.content:
            messages.append(HumanMessage(content=turn.user_message.content))
        if turn.assistant_message and turn.assistant_message.content:
            messages.append(AIMessage(content=turn.assistant_message.content))

    return messages


def _build_memory_context(
    thread: ManagedThread,
    include_reference_context: bool,
) -> str:
    """Build auxiliary non-dialogue memory context for the active thread."""
    lines: list[str] = []

    if thread.summary:
        lines.extend(
            [
                "Active thread summary:",
                thread.summary,
            ]
        )

    profile = thread.get_all_profile()
    if profile:
        lines.append("Active thread profile:")
        for key, value in sorted(profile.items()):
            lines.append(f"- {key}: {value}")

    if include_reference_context:
        reference_sections = _build_reference_sections(thread)
        if reference_sections:
            if lines:
                lines.append("")
            lines.append("Attached thread references:")
            lines.extend(reference_sections)

    return "\n".join(lines).strip()


def _build_reference_sections(thread: ManagedThread) -> list[str]:
    """Build summary/profile snippets for attached reference threads."""
    sections: list[str] = []

    for reference in thread.list_references():
        document = thread.store.load_thread_document(reference.thread_id)
        section = _format_reference_section(reference, document)
        if section:
            sections.append(section)

    return sections


def _format_reference_section(reference: ThreadReference, document: ThreadDocument) -> str:
    """Format one referenced thread without replaying its chat turns."""
    lines = [f"[{reference.display_name()} | {reference.thread_id}]"]

    if reference.include_summary and document.summary:
        lines.append(f"Summary: {document.summary}")

    if reference.include_profile and document.profile:
        lines.append("Profile:")
        for key, value in sorted(document.profile.items()):
            lines.append(f"- {key}: {value}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
