# AGENTS.md

This file provides guidance to Codex when working in this repository.

## Project Overview

This project is a RAG-based campus knowledge agent for Wenzhou Business College.

Current stack:

- LangChain tool-calling agent
- Qwen-compatible chat model via OpenAI-compatible API
- ChromaDB vector index
- BM25 + vector hybrid retrieval with RRF reranking
- Persistent thread-based conversational memory
- FastAPI backend
- Vue3 demo frontend
- CLI entry point for local debugging and one-shot queries

This is not just a basic retrieval demo. The current codebase already includes:

- query routing
- query rewrite
- query decomposition
- hybrid retrieval
- multi-turn memory
- thread references and profile storage
- structured major-comparison presentation output
- ambiguity clarification for follow-up major queries

## Commands

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>

# Run the CLI agent
uv run python main.py

# Run a one-shot query
uv run python main.py "计算机科学与技术和人工智能有什么区别"

# List memory threads
uv run python main.py --list-threads

# Start a new thread
uv run python main.py --new-thread

# Resume a thread
uv run python main.py --thread-id <thread_id>

# Attach a reference thread
uv run python main.py --thread-id <thread_id> --attach-thread <other_thread_id>

# Run indexing module directly
uv run python -m src.rag_agent.indexing.index_builder

# Run backend locally
uv run uvicorn rag_agent.api.app:app --reload

# Run frontend locally
cd frontend
npm run dev

# Run tests
uv run pytest tests/

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Environment Configuration

Copy `.env.example` to `.env`.

Important variables used by the current codebase:

- `QWEN_API_KEY`
- `QWEN_MODEL`
- `QWEN_BASE_URL`
- `DASHSCOPE_API_KEY`
- `EMBEDDING_MODEL`

Memory settings are loaded from `src/rag_agent/memory_session/config.py`.
Retrieval settings are loaded from `src/rag_agent/retrieval/config.py`.

Do not add new environment variables without updating `.env.example`.

## Architecture

High-level flow:

1. Load markdown documents from `data/raw/`
2. Chunk them into parent and child documents with metadata
3. Build or load the ChromaDB vector index
4. Build hybrid retrieval on top of vector search + BM25
5. Create the tool-calling agent
6. Persist turns into thread memory
7. Post-process selected answers into structured presentation data for the frontend

Current runtime entry point:

- `main.py`

Current web app entry points:

- backend: `src/rag_agent/api/app.py`
- frontend: `frontend/src/App.vue`

If reusable app logic is needed later for FastAPI, extract it from `main.py` instead of duplicating it.

## Module Map

| Module | Key Files | Purpose |
|--------|-----------|---------|
| `agent_modules/` | `agent.py`, `model.py`, `message_builder.py` | Agent assembly, model setup, memory-aware message building |
| `agent_modules/tools/` | `router.py`, `query_rewrite.py`, `query_decomposition.py`, `retrieval.py`, `memory_tools.py` | Tool definitions used by the agent |
| `api/` | `app.py`, `schemas.py`, `routes/chat.py`, `services/*` | FastAPI app, response contracts, chat orchestration |
| `api/services/` | `agent_runtime.py`, `chat_service.py`, `major_comparison.py` | Shared runtime, chat orchestration, comparison/clarification logic |
| `data_processing/` | `loader.py`, `chunker.py`, `metadata.py` | Document loading, chunking, metadata enrichment |
| `indexing/` | `index_builder.py`, `embeddings.py` | ChromaDB indexing and embedding wrapper |
| `retrieval/` | `hybrid_search.py`, `bm25_index.py`, `academy_map.py`, `config.py` | Hybrid retrieval and retrieval settings |
| `memory_session/` | `config.py`, `models.py`, `store.py`, `session.py`, `locks.py` | Persistent thread memory and locking |
| `prompts/` | `system_prompt.txt`, `router.txt`, `query_rewrite.txt`, `query_decomposition.txt` | Prompt assets |
| `frontend/` | `src/App.vue`, `src/components/*`, `src/composables/useChatSession.ts` | Vue chat UI and rendering logic |
| `utils/` | `path.py`, `prompt_loader.py` | Path and prompt helpers |

## Retrieval Design

The retrieval stack is hybrid, not vector-only.

Current retrieval behavior:

- vector search via `IndexBuilder.similarity_search()`
- BM25 retrieval via `BM25Indexer`
- hybrid retrieval via `HybridRetriever.hybrid_search()`
- RRF reranking in `HybridRetriever._rrf_rerank()`
- optional metadata filtering by detected college / major

Current route-specific retrieval tools:

- `list_retrieval_tool`
- `detail_retrieval_tool`
- `general_retrieval_tool`

The agent also uses:

- `router_tool`
- `query_rewrite_tool`
- `query_decomposition_tool`

Do not collapse this back into a single plain retrieval call unless explicitly requested.

## Major Comparison Design

The current "major comparison" capability is not implemented as a standalone agent tool.

It is implemented as a global chat-layer capability around the existing agent flow:

- pre-generation intent and boundary handling in `chat_service.py`
- comparison and ambiguity helpers in `major_comparison.py`
- normal retrieval still goes through the existing routed agent tools
- post-generation parsing extracts structured comparison presentation data

Important constraints:

- do not add a separate `comparison_tool` unless explicitly requested
- do not replace retrieval with hardcoded field-by-field comparisons
- use programmatic boundary control for ambiguous follow-up queries
- use prompt/output shaping only to improve answer format, not to own boundary decisions

Current comparison behavior should follow this rule order:

- explicit comparison intent: treat as comparison
- explicit single-major intent: treat as single-major explanation
- short ambiguous follow-up after a comparison: ask for clarification first

The preferred clarification text is:

- `你是想了解该专业，还是想和上一轮做对比？`

## Memory System

The memory system is a first-class subsystem and should be preserved.

Implemented capabilities:

- persistent thread documents on disk
- per-thread conversation turns
- per-thread profile storage
- per-thread summaries
- thread references
- assistant-message presentation metadata
- concurrent update locking
- corrupt file recovery
- switching active threads in CLI

Important classes:

- `MemorySettings`
- `ThreadDocument`
- `ConversationTurn`
- `ManagedThread`
- `SessionManager`
- `ThreadStore`

When adding features:

- reuse the existing memory model
- bind tools to an explicit `ManagedThread` when possible
- preserve autosave semantics in `ManagedThread` and `ThreadStore`
- if frontend rendering must survive page refresh or thread restore, persist the data in memory instead of returning it only from `/api/chat`

## Data, Paths, and Metadata

Always use helpers from [path.py](D:/ziruo_project/rag_project/src/rag_agent/utils/path.py) for project paths.

Available helpers include:

- `get_project_root()`
- `get_data_dir()`
- `get_raw_data_dir()`
- `get_chroma_db_dir()`

Do not hardcode repository paths, Windows absolute paths, or direct `data/...` strings in application logic.

Important directories:

- `data/raw/`
- `data/vector_index/`
- `data/memory/`

Chunk metadata is relied on by retrieval and filtering. Preserve fields such as:

- `parent_id`
- `doc_type`
- `doc_level`
- `college`
- `major`
- `section`
- `chunk_index`
- `source`
- `filename`

## Code Style

- Keep files under 500 lines; split by responsibility if needed.
- Code, comments, and docstrings in `src/` and `tests/` should be in English.
- Knowledge source markdown under `data/raw/` can remain Chinese.
- Do not hardcode API keys, model names, or absolute local paths.
- If path handling needs to be standardized, use the helpers in [path.py](D:/ziruo_project/rag_project/src/rag_agent/utils/path.py).
- Prefer extending existing modules over creating parallel implementations of retrieval or memory logic.
- For frontend demo UX, prefer compact and focused layouts; avoid oversized top-of-page showcase blocks unless explicitly requested.

## Testing Guidance

Current test coverage is strongest in the memory subsystem.

When adding features, prefer:

- focused unit tests
- API-level tests for future FastAPI work
- avoiding brittle end-to-end LLM assertions unless tightly scoped or mocked

For comparison and clarification features, test at three layers when possible:

- pure helper tests for parsing / intent detection / clarification resolution
- API tests for response shape and thread persistence
- frontend build verification for rendering and type safety

## Near-Term Direction

Planned evolution:

- better citation / evidence display
- more stable structured output for major-comparison questions
- cleaner frontend presentation for demo and resume use

When implementing these:

- extract reusable services from `main.py`
- do not bypass `memory_session`
- do not replace hybrid retrieval with a placeholder implementation
- optimize for demoability and resume value
- treat structured comparison display as a supporting demo feature, not the primary system capability
