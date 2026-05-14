# RAG Agent with LangChain

基于 LangChain 实现的 RAG Agent，使用阿里云 Qwen + ChromaDB + 工具调用。

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│  LangChain ReAct Agent                   │
│                                          │
│  Tools:                                   │
│  - rag_retriever (RAG 知识库检索)         │
│  - web_search (网络搜索)                  │
│  - python_repl (Python 代码执行)          │
│  - api_call (HTTP API 调用)               │
│                                          │
│  LLM: Qwen via DashScope                 │
└──────────────────────────────────────────┘
         │
         ▼
    ChromaDB (Vector Store)
```

## 项目结构

```
rag_project/
├── src/rag_agent/
│   ├── __init__.py
│   ├── config.py              # 配置管理
│   └── preparation/           # 数据准备模块
│       ├── loader.py          # 文档加载
│       ├── chunker.py         # 两级分块（学院级+专业级）
│       └── metadata.py        # 元数据提取
├── data/
│   ├── raw/                   # 原始文档
│   └── chroma_db/             # ChromaDB 持久化
├── docs/                      # 设计文档
├── tests/                     # 测试
├── main.py                    # CLI 入口
├── pyproject.toml
└── .env.example
```

## 环境配置

1. 复制 `.env.example` 为 `.env`
2. 填入 `QWEN_API_KEY`（阿里云 DashScope API Key）

## 安装

```bash
uv sync
```

## 使用

```bash
# 摄入文档
uv run python main.py ingest ./data/raw

# 查询
uv run python main.py query "你的问题"
```
