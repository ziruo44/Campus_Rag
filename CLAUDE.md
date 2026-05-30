# CLAUDE.md

This file provides guidance to coding agents working in this repository. It has two parts:
**behavioral guidelines** (how to work) and **project reference** (what to know).

---

## Behavioral Guidelines

These rules reduce common LLM coding mistakes. They bias toward caution over speed — for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project Reference

Campus knowledge RAG agent for Wenzhou Business College.
Stack: single outer agent + Qwen LLM + ChromaDB + hybrid retrieval (BM25 + vector) + FastAPI + Vue frontend.

### Commands

```bash
uv sync                          # Install dependencies
uv add <pkg>                     # Add a dependency
uv run python main.py            # Run CLI agent
uv run python main.py "<query>"  # One-shot query
uv run python -m domain.major_knowledge.indexing.index_builder  # Rebuild index
uv run uvicorn api_view.web_main:app --reload             # Backend
cd frontend && npm run dev       # Frontend
uv run pytest tests/             # Tests
uv run ruff check/format src/ tests/  # Lint & format
```

### Architecture Flow

```
user → outer agent → knowledge_workflow_tool → KnowledgeWorkflowService
  ├─ decomposition → routing → (rewrite if general) → hybrid retrieval
  └─ returns retrieval_context + evidence_bundle to agent for final answer
```

Entry points: `main.py` (CLI), `src/agent/__main__.py` (package CLI), `api_view/web_main.py` (FastAPI), `frontend/src/App.vue` (Vue).

### Module Map

| Module | Purpose |
|--------|---------|
| `agent/` | Outer agent, workflow orchestration, tool boundary |
| `agent/middleware/` | Inject thread memory into agent |
| `api_view/` | FastAPI routes and chat service |
| `domain/major_knowledge/ingestion/` | Document loading, chunking, metadata |
| `domain/major_knowledge/indexing/` | ChromaDB indexing and embeddings |
| `domain/major_knowledge/retrieval/` | Hybrid search, BM25, academy map, config |
| `memory/` | Persistent thread memory, session, compaction |
| `llm/` | Model setup, health probes, prompts |
| `utils/` | Paths, errors, text, time helpers |

### Code Standards

- Keep files under 500 lines when practical.
- Code, comments, docstrings in `src/` and `tests/` in chinese.
- Use path helpers from `src/utils/paths.py`.
- Use `uv` for package management.
- Do not hardcode API keys, model names, or absolute paths.
- Copy `.env.example` to `.env` — do not add new vars without updating it.

### Architectural Constraints

- No `domain -> agent` imports.
- `knowledge_workflow_tool` is the outer agent's sole business tool boundary.
- Thread memory owned inside `memory/`.
- Workflow returns retrieval evidence + structured trace; outer agent owns final answer.

### Testing

Prefer focused unit tests, API-level tests, workflow orchestration tests, memory injection tests.
Avoid brittle end-to-end LLM assertions.

```bash
uv run pytest tests/test_workflows/ tests/test_agent/ tests/test_app/ tests/test_memory/
```