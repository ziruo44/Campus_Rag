"""CLI entry point for the RAG Agent."""

from __future__ import annotations

import argparse
import logging

from rag_agent.api.services.agent_runtime import AgentRuntime
from rag_agent.api.services.chat_service import ChatService
from rag_agent.memory_session.session import ManagedThread, SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_query(chat_service: ChatService, thread: ManagedThread, query: str) -> str:
    """Run a single query while persisting turn lifecycle updates."""
    result = chat_service.invoke_thread(thread, query)

    print("\n" + "=" * 60)
    for msg in result.messages:
        role = msg.type.upper() if hasattr(msg, "type") else msg.__class__.__name__
        print(f"\n--- [{role}] ---")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  => Tool: {tc['name']}({tc['args']})")
        elif hasattr(msg, "content") and msg.content:
            print(f"  {msg.content[:2000]}")

    ai_response = result.answer
    print("\n" + "=" * 60)
    print("Answer:\n" + ai_response)
    return ai_response


def configure_active_thread(args: argparse.Namespace, manager: SessionManager) -> ManagedThread:
    """Open or create the active thread based on CLI arguments."""
    if args.new_thread:
        thread = manager.create_new_thread(switch=True)
    elif args.thread_id:
        thread = manager.switch_thread(args.thread_id)
    else:
        thread = manager.memory

    for target_thread_id in args.attach_thread:
        thread.attach_reference(target_thread_id)

    return thread


def main():
    parser = argparse.ArgumentParser(description="RAG Agent for Wenzhou Business College")
    parser.add_argument("query", nargs="*", help="Optional one-shot query")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--thread-id", help="Resume an existing memory thread")
    parser.add_argument("--new-thread", action="store_true", help="Create and switch to a new memory thread")
    parser.add_argument(
        "--attach-thread",
        action="append",
        default=[],
        help="Attach an existing thread as a read-only reference to the active thread",
    )
    parser.add_argument(
        "--list-threads",
        action="store_true",
        help="List stored memory thread IDs and exit",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    with SessionManager() as manager:
        if args.list_threads:
            for thread_id in manager.list_threads():
                print(thread_id)
            return

        active_thread = configure_active_thread(args, manager)
        logger.info("Initializing RAG Agent for thread %s", active_thread.thread_id)
        runtime = AgentRuntime()
        runtime.ensure_initialized()
        logger.info("Agent runtime is ready")
        chat_service = ChatService(runtime=runtime)
        logger.info("Agent initialized with thread %s", active_thread.thread_id)

        if args.query:
            run_query(chat_service, active_thread, " ".join(args.query))
            return

        print("\n=== Wenzhou Business College RAG Agent ===")
        print(f"Active thread: {active_thread.thread_id}")
        print("Enter a question, or type q to quit.\n")

        while True:
            query = input("Question: ").strip()
            if not query:
                continue
            if query.lower() == "q":
                print("Bye.")
                break
            run_query(chat_service, active_thread, query)


if __name__ == "__main__":
    main()
