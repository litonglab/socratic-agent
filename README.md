# Networking Labs AI TA（RAG-Agent）

这是一个面向计算机网络实验课的教学助手系统，核心做法是把课程文档检索、拓扑结构化理解、分层提示策略放在同一条 Agent 流程里，让回答尽量做到有依据、可追溯，也能在教学场景中逐步引导学生思考，而不是直接给最终答案。

## 项目能力概览

- 文本 RAG：从 `data/` 下的实验文档检索证据，并在回答中附引用。
- 拓扑增强：将实验图转成结构化拓扑数据，运行时只读取审核通过的 `approved_json`。
- 分层教学引导：按问题类型和对话状态动态调整 `hint_level` 与提示策略。
- 多轮会话管理：支持注册登录、会话持久化、点赞点踩、用户水平记录。
- 可选联网搜索：可在对话中启用网络搜索作为补充信息来源。

## 核心流程（简版）

1. 用户提问进入 `agentic_rag/agent.py`。
2. 系统做相关性判断、问题分类和提示层级决策。
3. Agent 在循环中按需调用工具（检索 / 拓扑 / 搜索）。
4. 汇总结果并补充引用，返回给前端与 API，同时异步持久化会话状态。

## 快速开始

### 1) 环境准备

- Python 3.10 及以上（建议 3.10/3.11）。
- 建议在虚拟环境中运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

前端运行需要 Node.js（建议 18 LTS 及以上）与 npm。


### 2) 配置环境变量

复制模板并填写密钥：

```bash
cp .env.example .env
```

最少需要配置：

- `DEEPSEEK_API_KEY`：聊天主模型必填。
- `OPENAI_API_KEY`：仅在构建拓扑结构化数据或运行评测脚本时需要。

说明：`.env.example` 里的 `BACKEND_BASE_URL` 当前版本未被核心代码使用，可保留不动。

### 3) 准备课程文档与索引

- 把实验指导书（`.docx`）放到 `data/`。
- 默认会优先读取已有索引，不强制重建。
- 如果希望从文档重新切分并生成索引，可临时开启：

```bash
export RAG_REBUILD_INDEX=1
```

### 4) 启动后端

后端是 FastAPI（`server.py`），下面三种方式按需选用，**都只启后端**，前端在第 5) 节单独启动。

开发模式（推荐）：

```bash
bash scripts/start_dev.sh
```

- 启动 `uvicorn --reload`，监听 `http://127.0.0.1:8000`。
- 默认开启 `RAG_DEV_FAST_START=1`，后端先可访问，再后台预热 RAG（第一次请求会冷启动，第二次起明显变快）。

稳定模式（演示或部署前自测）：

```bash
bash scripts/start_all.sh
```

- uvicorn lifespan 阻塞直到 RAG 就绪后才开放端口，因此 `/health` 通过即代表服务可用。
- 默认后端端口 `8000`。

适合 WSL 的启动方法：

```bash
bash scripts/start_wsl.sh
```

- 自动加载 `.venv` 与 `.env`，优先使用本地 embedding 目录（`models/bge-m3`），默认禁用 reranker。
- 监听 `0.0.0.0:8000`，方便从 Windows 主机访问。

如果你只想直接调原生命令：

```bash
uvicorn server:app --reload --port 8000
```

### 5) 启动前端（React + Vite）

前端代码在 `frontend/` 目录，使用 React 19 + Vite + TailwindCSS。**首次运行**：

```bash
cd frontend
npm install
npm run dev
```

- 默认监听 `http://localhost:5173`。
- Vite 已配好开发代理：`/api` 与 `/health` 自动转发到 `http://localhost:8000`，无需改任何 URL。
- 因此通常的开发组合是两个终端：终端 A 跑 `bash scripts/start_dev.sh`（后端），终端 B 跑 `npm run dev`（前端）。

WSL / 远程主机想从其他设备访问，加 `--host`：

```bash
npm run dev -- --host
```

生产构建：

```bash
cd frontend
npm run build      # 产物在 dist/，由 Nginx 或 FastAPI 反代静态文件即可
npm run preview    # 本地预览生产构建（仅自测用）
```


## MCP 支持

项目新增了一个只读的 MCP Server，便于在 Cursor、Claude Desktop 或其他支持 MCP 的客户端中复用仓库内的稳定能力，而不影响现有 `FastAPI + React` 主链路。

当前开放的工具包括：

- `retrieve_course_docs`：检索课程文档，返回结构化引用与拼接后的上下文。
- `get_topology_context`：读取实验拓扑上下文，只使用审核通过的 `approved_json`。
- `list_available_experiments`：列出当前具备可用拓扑数据的实验编号。
- `get_experiment_manifest`：读取指定实验的 `manifest.json`；若缺失，则退化为基于 `approved_json` 的摘要。

启动方式：

```bash
python -m mcp_server.server
```

或：

```bash
bash scripts/start_mcp_server.sh
```

当前版本使用 `stdio` 传输，适合本地 IDE / Agent 客户端接入。客户端配置示例：

```json
{
  "mcpServers": {
    "networking-lab-agent": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/RAG-Agent"
    }
  }
}
```

说明：

- `retrieve_course_docs` 会复用项目现有的 RAG 检索栈，因此首次调用可能触发模型与索引冷启动。
- `get_topology_context` 与 `get_experiment_manifest` 默认读取 `TOPO_STORE_ROOT` 下的数据。
- MCP 层位于 `mcp_server/`，属于旁路接入，不会改变现有 `/api/chat` 与前端调用方式。
