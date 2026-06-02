"""Command-line interface for the campus knowledge agent."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app_bootstrap import (
    get_life_guide_workflow_service,
    get_major_workflow_service,
)
from agent.workflows.life_guide_service import LifeGuideWorkflowService
from agent.workflows.service import MajorKnowledgeWorkflowService
from api_view.services.chat_service import ChatService
from domain.life_guide_knowledge.retrieval.bm25_index import (
    prewarm_jieba as prewarm_life_guide_jieba,
)
from domain.major_knowledge.retrieval.bm25_index import prewarm_jieba
from memory.config import MemorySettings
from memory.session import ManagedThread, SessionManager
from shared.logging_setup import configure_logging

_WELCOME_BANNER = """
Campus Knowledge CLI
Type your question and press Enter.
Commands: /help, /thread, /threads, /new, /switch <thread_id>, /attach <thread_id>, /refs, /history, /artifacts, /exit
""".strip()

_HELP_TEXT = """
/help                 Show available commands
/thread               Show the current thread status
/threads              List persisted threads
/new                  Create and switch to a new thread
/switch <thread_id>   Switch to an existing thread
/attach <thread_id>   Attach a reference thread to the current thread
/refs                 List attached reference threads
/history              Show recent turns in the current thread
/artifacts            Show artifacts from the latest completed turn
/exit                 Exit the CLI
""".strip()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the campus knowledge agent from the command line.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="One-shot user query. If omitted, start an interactive session.",
    )
    parser.add_argument(
        "--thread-id",
        help="Resume an existing thread by ID.",
    )
    parser.add_argument(
        "--new-thread",
        action="store_true",
        help="Create and switch to a new thread before running.",
    )
    parser.add_argument(
        "--list-threads",
        action="store_true",
        help="List persisted thread IDs.",
    )
    parser.add_argument(
        "--attach-thread",
        help="Attach a reference thread to the active thread before running.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = MemorySettings(
        compaction_notice_callback=_print_compaction_notice,
    )
    major_workflow_service = get_major_workflow_service()
    life_guide_workflow_service = get_life_guide_workflow_service()
    chat_service = ChatService(
        major_workflow_service,
        life_guide_workflow_service,
        session_manager_factory=lambda: SessionManager(settings),
    )

    with SessionManager(settings) as manager:
        if args.list_threads:
            _print_threads(manager)
            return 0

        thread = _resolve_thread(
            manager,
            thread_id=args.thread_id,
            create_new=args.new_thread,
        )
        if args.attach_thread:
            thread.attach_reference(args.attach_thread)

        if args.message:
            _prewarm_cli_runtime(
                major_workflow_service,
                life_guide_workflow_service,
            )
            return _run_one_shot(chat_service, thread, args.message)

        _prewarm_cli_runtime(
            major_workflow_service,
            life_guide_workflow_service,
        )
        return _run_interactive(chat_service, manager, thread)


def _resolve_thread(
    manager: SessionManager,
    *,
    thread_id: str | None,
    create_new: bool,
):
    """Resolve the target thread for the current CLI run."""
    if create_new:
        return manager.create_new_thread(switch=True)
    if thread_id:
        return manager.switch_thread(thread_id)
    return manager.thread


def _print_threads(manager: SessionManager) -> None:
    """Print thread IDs, newest first according to store ordering."""
    thread_ids = manager.list_threads()
    if not thread_ids:
        print("No threads found.")
        return

    for thread_id in thread_ids:
        marker = "*" if thread_id == manager.current_thread_id else " "
        print(f"{marker} {thread_id}")


def _run_one_shot(
    chat_service: ChatService,
    thread: ManagedThread,
    message: str,
) -> int:
    """Run one non-interactive CLI turn."""
    try:
        result = chat_service.invoke_thread(thread, message)
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print(f"[thread_id] {thread.thread_id}", file=sys.stderr)
        return 1

    print(result.answer)
    if result.artifacts:
        print(f"\n[artifacts] {', '.join(sorted(result.artifacts.keys()))}")
    print(f"\n[thread_id] {result.thread_id}")
    return 0


def _run_interactive(
    chat_service: ChatService,
    manager: SessionManager,
    thread: ManagedThread,
) -> int:
    """Run the interactive REPL-like CLI."""
    print(_WELCOME_BANNER)
    print(f"Current thread: {thread.thread_id}")
    _print_history(thread)

    while True:
        try:
            raw_message = input(f"\nYou [{thread.thread_id}]> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        message = raw_message.strip()
        if not message:
            continue

        if message.startswith("/"):
            thread, should_exit = _handle_command(manager, thread, message)
            if should_exit:
                print("Bye.")
                return 0
            continue

        print("Assistant is thinking...")
        try:
            result = chat_service.invoke_thread(thread, message)
        except Exception as exc:
            print(f"Request failed: {exc}")
            continue

        print(f"\nAssistant [{result.thread_id}]>\n{result.answer}")
        _print_latest_turn_summary(thread, result.artifacts)


def _handle_command(
    manager: SessionManager,
    thread: ManagedThread,
    command_line: str,
) -> tuple[ManagedThread, bool]:
    """Handle one interactive CLI slash command."""
    command, _, argument = command_line.partition(" ")
    arg = argument.strip()

    if command == "/help":
        print(_HELP_TEXT)
        return thread, False

    if command == "/thread":
        _print_thread_status(thread)
        return thread, False

    if command == "/threads":
        _print_threads(manager)
        return thread, False

    if command == "/new":
        thread = manager.create_new_thread(switch=True)
        print(f"Switched to new thread: {thread.thread_id}")
        _print_history(thread)
        return thread, False

    if command == "/switch":
        if not arg:
            print("Usage: /switch <thread_id>")
            return thread, False
        try:
            thread = manager.switch_thread(arg)
        except FileNotFoundError:
            print(f"Thread not found: {arg}")
            return thread, False
        print(f"Switched to thread: {thread.thread_id}")
        _print_history(thread)
        return thread, False

    if command == "/attach":
        if not arg:
            print("Usage: /attach <thread_id>")
            return thread, False
        try:
            thread.attach_reference(arg)
        except FileNotFoundError:
            print(f"Thread not found: {arg}")
            return thread, False
        print(f"Attached reference thread: {arg}")
        return thread, False

    if command == "/refs":
        references = thread.list_references()
        if not references:
            print("No attached references.")
            return thread, False
        print("Attached references:")
        for reference in references:
            print(f"- {reference.display_name()} ({reference.thread_id})")
        return thread, False

    if command == "/history":
        _print_history(thread)
        return thread, False

    if command == "/artifacts":
        _print_latest_artifacts(thread)
        return thread, False

    if command == "/exit":
        return thread, True

    print(f"Unknown command: {command}. Type /help for commands.")
    return thread, False


def _print_history(thread: ManagedThread, limit: int = 6) -> None:
    """Print recent turns from the current thread."""
    thread.refresh()
    turns = thread.turns[-limit:]
    if not turns:
        print("Current thread has no turns yet.")
        return

    print(f"Recent turns in {thread.thread_id}:")
    for turn in turns:
        if turn.user_message and turn.user_message.content:
            print(f"You: {turn.user_message.content}")
        if turn.assistant_message and turn.assistant_message.content:
            print(f"Assistant: {turn.assistant_message.content}")
        if turn.state == "failed" and turn.error:
            print(f"Error: {turn.error}")


def _print_latest_turn_summary(
    thread: ManagedThread,
    artifacts: dict[str, object] | None = None,
) -> None:
    """Print a compact status line for the latest turn."""
    artifact_payload = dict(artifacts or {})
    if not artifact_payload:
        thread.refresh()
        if not thread.turns:
            return
        artifact_payload = dict(thread.turns[-1].artifacts or {})

    artifact_keys = sorted(artifact_payload.keys())
    if not artifact_keys:
        return

    print(f"[latest turn artifacts] {', '.join(artifact_keys)}")


def _print_latest_artifacts(thread: ManagedThread) -> None:
    """Print artifacts from the latest completed turn."""
    thread.refresh()
    latest_completed_turn = next(
        (
            turn
            for turn in reversed(thread.turns)
            if turn.state == "completed" and turn.artifacts
        ),
        None,
    )
    if latest_completed_turn is None:
        print("No persisted artifacts found in the current thread.")
        return

    artifacts = latest_completed_turn.artifacts
    workflow_summary = artifacts.get("workflow_summary") or {}
    evidence_bundle = artifacts.get("evidence_bundle") or []

    print(f"Artifacts for turn: {latest_completed_turn.turn_id}")
    capability_type = artifacts.get("capability_type")
    if capability_type:
        print(f"Capability: {capability_type}")

    route_trace = workflow_summary.get("route_trace") or []
    resolved_queries = workflow_summary.get("resolved_queries") or []
    workflow_trace = workflow_summary.get("workflow_trace") or []

    print(f"Resolved queries: {len(resolved_queries)}")
    print(f"Route trace: {', '.join(route_trace) if route_trace else 'none'}")
    print(f"Workflow trace events: {len(workflow_trace)}")
    print(f"Evidence count: {len(evidence_bundle)}")

    if evidence_bundle:
        first_evidence = evidence_bundle[0]
        source = first_evidence.get("source") or "unknown"
        print(f"Top evidence source: {source}")


def _print_thread_status(thread: ManagedThread) -> None:
    """Print a compact status view for the current thread."""
    thread.refresh()
    references = thread.list_references()
    turns = thread.turns

    print(f"Current thread: {thread.thread_id}")
    print(f"Title: {thread.title or 'New Session'}")
    print(f"Turn count: {len(turns)}")
    print(f"Reference count: {len(references)}")
    if thread.summary:
        print(f"Summary: {thread.summary}")


def _prewarm_cli_runtime(
    major_workflow_service: MajorKnowledgeWorkflowService,
    life_guide_workflow_service: LifeGuideWorkflowService,
) -> None:
    """预热两套知识库运行时，降低 CLI 首轮延迟。"""
    prewarm_jieba()
    prewarm_life_guide_jieba()
    if major_workflow_service.is_initialized and life_guide_workflow_service.is_initialized:
        return

    print("Prewarming knowledge runtimes...")
    try:
        if not major_workflow_service.is_initialized:
            major_workflow_service.ensure_initialized()
        if not life_guide_workflow_service.is_initialized:
            life_guide_workflow_service.ensure_initialized()
    except Exception as exc:
        print(f"Runtime prewarm failed: {exc}")
    else:
        print("Knowledge runtimes ready.")


def _print_compaction_notice(payload: dict) -> None:
    """Print a CLI-visible notice when thread memory is compacted."""
    compacted_delta = int(payload.get("compacted_delta", 0))
    if compacted_delta <= 0:
        print("Compressing older context...")
        return
    print(f"Compressing older context... merged {compacted_delta} turn(s) into memory summary.")
