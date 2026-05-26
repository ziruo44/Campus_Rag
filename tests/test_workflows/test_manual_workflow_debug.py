"""Manual workflow debug entrypoint for inspecting retrieval output."""

from __future__ import annotations

import json

import pytest

from app_bootstrap import get_workflow_service


def pytest_addoption(parser):
    """This module relies on options registered in the root conftest."""


@pytest.mark.manual
def test_manual_workflow_debug(pytestconfig) -> None:
    """Run one real workflow query and print retrieval artifacts for inspection."""
    query = pytestconfig.getoption("workflow_query")
    if not query:
        pytest.skip("Pass --workflow-query to run manual workflow inspection.")

    precise = pytestconfig.getoption("workflow_precise")
    strategy = "passthrough" if precise else "compressed"

    result = get_workflow_service().execute(
        user_query=query,
        retrieval_context_strategy=strategy,
    )

    print("\n=== query ===")
    print(query)
    print("\n=== retrieval_context ===")
    print(result["retrieval_context"])
    print("\n=== evidence_bundle ===")
    print(json.dumps(result["evidence_bundle"], ensure_ascii=False, indent=2))
    print("\n=== resolved_queries ===")
    print(json.dumps(result["resolved_queries"], ensure_ascii=False, indent=2))
    print("\n=== workflow_trace ===")
    print(json.dumps(result["workflow_trace"], ensure_ascii=False, indent=2))
