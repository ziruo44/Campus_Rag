# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Agent for Wenzhou Business College campus knowledge base (温州商学院学院介绍). Built with LangChain + Qwen (DashScope) + ChromaDB, featuring ReAct-based tool calling.

## Commands

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package>

# Run indexing test
uv run python -m src.rag_agent.indexing.index_builder

# Run tests
uv run pytest tests/

# Lint & format
uv run ruff check src/
uv run ruff format src/

# Run with specific Python version
uv run python main.py <args>
```

## Environment Configuration

Copy `.env.example` to `.env` and set required API keys:
- `QWEN_API_KEY` - Qwen/DashScope API key (required)
- `DASHSCOPE_API_KEY` - Alternative API key (optional)
- `EMBEDDING_MODEL` - Embedding model (default: `tongyi-embedding-vision-flash-2026-03-06`)

All settings are in `src/rag_agent/config.py` via Pydantic Settings — never hardcode config values.

## Architecture

```
用户查询
    │
    ▼
数据处理 (data_processing/) ───► 文档加载 ─► 分块 ─► 元数据增强
    │
    ▼
索引构建 (indexing/) ───► DashScope 嵌入 ─► ChromaDB 持久化
    │
    ▼
向量检索 ─► 生成回答
```

### Module Map

| Module | File | Purpose |
|--------|------|---------|
| `data_processing/` | `loader.py` | Document loading (Markdown) |
| `data_processing/` | `chunker.py` | Two-level chunking (college → major → section) |
| `data_processing/` | `metadata.py` | Metadata extraction and enrichment |
| `indexing/` | `index_builder.py` | ChromaDB index builder with caching, deduplication, metadata filtering |
| `indexing/` | `embeddings.py` | DashScope MultiModalEmbedding wrapper |
| `utils/` | `path.py` | Path utilities (get_project_root, get_data_dir, etc.) |
| `config.py` | Settings | Pydantic Settings singleton |

### File Structure

```
src/rag_agent/
├── __init__.py
├── config.py              # Settings (API keys, paths)
├── data_processing/
│   ├── __init__.py
│   ├── loader.py          # load_document(), load_documents()
│   ├── chunker.py         # split_by_college(), split_by_major(), chunk_documents()
│   └── metadata.py        # extract_*(), compute_parent_id(), enrich_*()
├── indexing/
│   ├── __init__.py
│   ├── index_builder.py   # IndexBuilder class (build, load, search, metadata_filtered_search)
│   └── embeddings.py     # DashScopeEmbeddings class
└── utils/
    ├── __init__.py
    └── path.py            # get_project_root(), get_data_dir(), get_raw_data_dir()
```

### Critical Dependency Chains

- `index_builder.py` → `embeddings.py` → `dashscope`
- `index_builder.py` → `utils/path.py` → `get_data_dir()`
- `data_processing/` → Document metadata (parent_id, college, major, section)

### Data Flow

```python
# 1. Load documents
docs = load_documents(get_raw_data_dir())

# 2. Chunk documents (college + major with children)
parents, children = chunk_documents(docs)
all_chunks = parents + children

# 3. Build or load index (with deduplication)
builder = IndexBuilder()
builder.load_or_build_index(all_chunks)

# 4. Similarity search
results = builder.similarity_search("计算机科学与技术", k=5)

# 5. Metadata filtered search
results = builder.metadata_filtered_search(
    query="培养目标",
    filters={"college": "信息工程学院"},
    k=5
)
```

### Chunk Metadata Convention

Each document chunk carries metadata for deduplication and filtering:

| Key | Description |
|-----|-------------|
| `parent_id` | MD5 hash of `college` or `college:major` (stable across re-runs) |
| `doc_type` | `"parent"` or `"child"` |
| `doc_level` | `"college"` or `"major"` |
| `college` | College name (extracted from ## header) |
| `major` | Major name (extracted from ### header) |
| `section` | Section name (extracted from #### header) |
| `chunk_index` | Index within parent document (for children) |
| `source` | Source file path |
| `filename` | Source filename |

### IndexBuilder Features

1. **Caching**: Loads existing index from `data/vector_index/` if present
2. **Deduplication**: Skips chunks with duplicate `parent_id`
3. **Auto-persist**: ChromaDB auto-saves via `persist_directory`
4. **Metadata filtering**: `metadata_filtered_search(query, filters, k)` for field-based filtering

### DashScope Embeddings

Using `tongyi-embedding-vision-flash-2026-03-06` (multimodal model, text-only mode):

```python
DashScopeEmbeddings(
    model="tongyi-embedding-vision-flash-2026-03-06",  # or from EMBEDDING_MODEL env
    dimension=768,  # Supported: 64, 128, 256, 512, 768
    api_key=None    # Auto-read from DASHSCOPE_API_KEY or QWEN_API_KEY
)
```

### Path Utilities (Required)

**必须**使用 `src/rag_agent/utils/path.py` 中的工具函数处理路径：

```python
from rag_agent.utils.path import get_project_root, get_data_dir, get_raw_data_dir

get_project_root()  # → D:\ziruo_project\rag_project
get_data_dir()       # → .../data
get_raw_data_dir()   # → .../data/raw
```

**禁止硬编码路径**，所有路径操作必须通过工具函数

### ChromaDB Patterns

- Use `persist_directory` for persistence
- Use `collection_name` to organize data (default: `rag_collection`)
- Metadata filtering: `vectorstore.get(where={"college": "信息工程学院"})`
- Deduplication: Check `parent_id` before adding

## Code Standards

### File Line Limit

- No single code file shall exceed **500 lines**
- Files exceeding 500 lines must be split by functionality into separate files (e.g., into `nodes/`, `tools/`, `utils/` subdirectories)
- Splitting principle: one file per clearly-defined functional module

### Class-Based Modules

Follow the `all-in-rag` pattern:

```python
class MyModule:
    def __init__(self, deps):
        self.deps = deps
        self.setup_xxx()

    def setup_xxx(self):
        logger.info("Setting up...")
```

### Configuration

All runtime config lives in `.env`, loaded via `pydantic_settings.BaseSettings` in `config.py`. Do not add new environment variables without updating `.env.example`.

### UV Package Management

This project uses **UV** for Python package management.

```bash
uv add <package>           # Add dependency
uv sync                    # Install from pyproject.toml
uv lock                    # Update lock file
```

### Language

All code, comments, and documentation in `src/` and `scripts/` use **English**. Knowledge base source files (`*.md` in `data/raw/`) use Chinese.