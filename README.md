# RAG Agent 校园知识库

校园知识问答 Agent，基于 LangChain ReAct 架构，使用阿里云 Qwen + ChromaDB + 混合检索（BM25 + 向量）。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Qwen（阿里云 DashScope） |
| 向量数据库 | ChromaDB |
| 检索方式 | 混合检索（BM25 + 向量） |
| API 框架 | FastAPI + Uvicorn |
| 前端 | Vue 3 + Vite |
| 包管理 | uv |

## 项目结构

```
src/
├── agent/                    # 外层 Agent 与工具边界
│   ├── main_agent.py         # Agent 入口
│   ├── cli.py                # CLI 命令行实现
│   ├── middleware/           # 中间件（导航人工审核、记忆注入）
│   ├── tools/                # 工具（知识工作流、校园导航）
│   ├── workflows/            # 工作流编排（分解、路由、改写、检索）
│   └── result_parser.py      # 结果解析
├── api_view/                 # FastAPI 路由与聊天服务
│   ├── web_main.py           # FastAPI 入口
│   ├── routers/              # `/campus/...` 路由
│   ├── schemas/              # API 请求响应 schema
│   └── services/             # 业务服务（ChatService）
├── domain/                   # 领域模块（无 agent 依赖）
│   └── knowledge/
│       ├── ingestion/        # 文档加载、分块、元数据提取
│       ├── indexing/         # ChromaDB 索引与嵌入向量
│       └── retrieval/        # 混合搜索、BM25、学院地图
├── memory/                   # 持久化会话记忆、线程记忆压缩
├── llm/                      # 模型配置、健康检查、提示词模板
├── utils/                    # 路径、错误、文本、时间工具
└── shared/                   # 可共享组件（可观测性）

frontend/                    # Vue 3 前端
├── src/
│   ├── App.vue               # 根组件
│   ├── components/           # UI 组件（ChatHeader、MessageList、ChatComposer 等）
│   ├── services/api.ts       # 后端 API 调用
│   ├── composables/          # 组合式函数（useChatSession）
│   └── types/                # TypeScript 类型定义

data/                        # 数据目录
├── chroma_db/               # ChromaDB 持久化存储
└── memory/sessions/         # 会话记忆存储
```

## 架构流程

```
用户 → 外层 Agent → knowledge_workflow_tool → KnowledgeWorkflowService
                                      ├─ decomposition（查询分解）
                                      ├─ routing（路由判断）
                                      ├─ rewrite（如需改写）
                                      └─ hybrid retrieval（BM25 + 向量）

返回 retrieval_context + evidence_bundle → Agent 生成最终答案
```

## 功能特性

### 知识问答
- 基于校园文档的 RAG 检索问答（政策、通知、办事指南等）
- 混合检索：BM25 关键词匹配 + 向量相似度融合
- 查询分解：将复杂问题拆分为多个子查询分别检索
- 路由判断：自动识别知识库查询 vs 其他问题

### 校园导航
- 校园地点查询与路线规划
- 导航请求需人工确认（Middleware 拦截 + 确认提示）
- 支持起点/终点修改

### 会话记忆
- 持久化线程（thread）存储在本地文件系统
- 线程间引用（attach reference thread）
- 自动压缩 older context 到摘要（减少 token 消耗）

### HTTP API
- `POST /campus/messages` — 发送非流式消息
- `POST /campus/messages/stream` — 发送流式消息
- `GET /campus/threads` — 获取线程摘要列表
- `GET /campus/threads/{thread_id}` — 获取线程详情
- `DELETE /campus/threads/{thread_id}` — 删除线程
- `DELETE /campus/threads/{thread_id}/turns/{turn_id}` — 删除轮次
- `GET /campus/health` — 健康检查，可追加 `?check_model=true` 探测模型连通性

## 启动方式

### 前置要求

1. 确保 `.env` 文件存在（从 `.env.example` 复制并填入真实值）：
   ```
   QWEN_API_KEY=你的阿里云 API Key
   QWEN_MODEL=qwen3.5-plus-2026-04-20
   QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

2. 安装依赖：
   ```bash
   uv sync
   ```

### 启动后端 API

```bash
uv run uvicorn api_view.web_main:app --reload
```

API 文档地址：`http://localhost:8000/docs`

### 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端地址：`http://localhost:5173`

### 运行 CLI Agent

交互模式：
```bash
uv run python main.py
```

单次查询：
```bash
uv run python main.py "图书馆开放时间"
```

CLI 常用命令：
| 命令 | 说明 |
|------|------|
| `/thread` | 显示当前线程状态 |
| `/threads` | 列出所有持久化线程 |
| `/new` | 创建新线程 |
| `/switch <id>` | 切换到指定线程 |
| `/attach <id>` | 附加引用线程 |
| `/history` | 显示最近对话 |
| `/artifacts` | 显示最新 artifacts |
| `/help` | 帮助 |

### 重建索引

```bash
uv run python -m domain.knowledge.indexing.index_builder
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `QWEN_API_KEY` | 阿里云 DashScope API Key | 必填 |
| `QWEN_MODEL` | 模型名称 | qwen3.5-plus |
| `QWEN_BASE_URL` | API 地址 | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化目录 | ./data/chroma_db |
| `EMBEDDING_MODEL` | 嵌入模型 | text-embedding-3-small |
| `MAX_ITERATIONS` | Agent 最大迭代步数 | 10 |
| `RELEVANCE_THRESHOLD` | 相关性阈值 | 0.7 |
| `TOP_K` | 检索返回数量 | 5 |

## 测试

```bash
uv run pytest tests/
```

代码检查：
```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```
