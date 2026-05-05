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

如果你需要支持"在前端上传图片并做 OCR"，建议额外安装：

```bash
pip install rapidocr-onnxruntime
```

不安装也不会影响纯文本问答；只有当前端上传图片、后端 `vision` 模块尝试做 OCR 时才会用到。

### 2) 配置环境变量

复制模板并填写密钥：

```bash
cp .env.example .env
```

最少需要配置：

- `DEEPSEEK_API_KEY`：聊天主模型必填。
- `OPENAI_API_KEY`：仅在构建拓扑结构化数据或运行评测脚本时需要。

常用变量如下：

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无 | 聊天模型鉴权，必填 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek 接口地址 |
| `DEEPSEEK_CHAT_MODEL` | `deepseek-chat` | 聊天模型名 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | 文本向量模型 |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | 重排模型 |
| `DISABLE_RERANKER` | `0` | 为 `1` 时关闭重排，走纯混合召回 |
| `RAG_INDEX_DIR` | 自动优先 `faiss_index/enriched` | 指定向量索引目录 |
| `RAG_REBUILD_INDEX` | `0` | 为 `1` 时按 `data/*.docx` 重建索引 |
| `TOPO_STORE_ROOT` | `topo_store` | 拓扑结构化数据根目录 |
| `TOPO_DEFAULT_EXPERIMENT_ID` | 空 | 问题未带实验号时的默认实验 |
| `MAX_CHAT_CONCURRENCY` | `50` | 聊天并发上限 |
| `CORS_ALLOWED_ORIGINS` | `*` | FastAPI 跨域白名单 |
| `HF_ENDPOINT` | 空 | HuggingFace 镜像地址（网络不稳定时可配） |

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
cd ~/socratic-agent
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

## API 速查

`server.py` 提供以下接口：

- 无鉴权：`GET /health`、`GET /health/ready`、`POST /api/register`、`POST /api/login`
- 需鉴权（Bearer Token）：`/api/me`、`/api/chat`、`/api/chat/stream`、`/api/feedback`、`/api/sessions*`

### 鉴权流程

1. `POST /api/register`（或 `POST /api/login`）获取 `token`。
2. 调用受保护接口时带上请求头：
   `Authorization: Bearer <token>`

### ChatRequest / ChatResponse

请求体（`POST /api/chat`）：

```json
{
  "message": "用户问题",
  "session_id": "可选，留空会自动创建",
  "history": [{"role": "user|assistant|system", "content": "..."}],
  "debug": false,
  "max_turns": 5,
  "enable_websearch": true
}
```

响应体：

```json
{
  "session_id": "s_xxx",
  "message_id": "m_xxx",
  "reply": "可见回答",
  "thinking": "隐藏思考内容（可为空）",
  "history": [{"role": "user|assistant|system", "content": "..."}],
  "tool_traces": [{"tool": "检索|拓扑|搜索", "input": "...", "output": "..."}]
}
```

### 流式接口

`POST /api/chat/stream` 使用 SSE，事件类型包括：

- `meta`：会话与消息元信息
- `delta`：增量 token
- `done`：完整结束包
- `error`：异常信息

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

## 拓扑结构化数据构建

当你需要把实验文档中的拓扑图转成结构化 JSON，可执行：

```bash
python scripts/build_topology_store.py \
  --docx "data/实验13：子网划分（详细版）.docx" \
  --overwrite
```

处理流程是：抽图 -> 前置分类（topology/non_topology/unclear）-> 抽取 -> 审核 -> 发布。  
运行时仅读取：

```text
topo_store/<experiment_id>/approved_json/
```

建议目录形态：

```text
topo_store/
└── lab13/
    ├── images/
    ├── classifications/
    ├── raw_json/
    ├── reviews/
    ├── approved_json/
    └── manifest.json
```

## 构建拓扑增强评测集

如果你希望在评测里更充分体现 Topology RAG 的能力，可以用题库脚本生成“拓扑题占比更高”的数据集：

```bash
python eval/build_balanced_qa_dataset.py \
  --topo-ratio 0.45 \
  --target-size 93 \
  --min-topo-per-experiment 3 \
  --output eval/qa_dataset_topo_balanced.json
```

脚本会合并 `eval/topology_question_bank.json` 并自动打上 `requires_topology` 字段，随后你可以在评测脚本里显式指定数据集路径，例如：

```bash
python eval/ablation_study.py --dataset eval/qa_dataset_topo_balanced.json
python eval/performance_benchmark.py --dataset eval/qa_dataset_topo_balanced.json
python eval/judge_consistency.py --dataset eval/qa_dataset_topo_balanced.json
python eval/topology_evaluation.py --questions-file eval/topology_question_bank.json
```

如果你已经补齐了多个实验的 `topo_store/*/approved_json`，建议先扩充拓扑题库，再重建评测集：

```bash
python eval/expand_topology_question_bank.py \
  --bank eval/topology_question_bank.json \
  --output eval/topology_question_bank.json \
  --max-new-per-topology 4
```

该脚本会跨实验自动补题（去重并延续 `TQ` 编号），随后可继续执行上面的平衡构建与评测命令。
其中 `--min-topo-per-experiment` 用于约束每个实验在平衡集里的最低拓扑题数量，能有效避免题目过度集中在单个实验。

如果你想在不重写全量题库的前提下提升评测集质量，可以再执行一次清洗补齐：

```bash
python eval/curate_qa_dataset_v2.py \
  --seed-dataset eval/qa_dataset_topo_balanced.json \
  --output eval/qa_dataset_topo_balanced_v2.json \
  --report eval/qa_dataset_topo_balanced_v2_report.json \
  --min-reference-len 20
```

说明：`--min-reference-len` 越高，题目可判分性越好，但可用拓扑题上限会下降；可结合报告里的 `high_quality_candidate_topology_n` 调整阈值。

## 目录结构（核心部分）

```text
RAG-Agent/
├── agentic_rag/
│   ├── agent.py
│   ├── rag.py
│   ├── topo_rag.py
│   ├── topo_models.py
│   ├── llm_config.py
│   ├── vision.py
│   ├── chat_format.py
│   └── web_search.py
├── frontend/                  # React 19 + Vite + TailwindCSS 前端
│   ├── src/
│   │   ├── pages/             # ChatPage / LoginPage
│   │   ├── components/        # 消息流、输入框、侧栏、徽标、Toast 等
│   │   ├── lib/               # 与后端的 fetch + SSE 客户端
│   │   └── hooks/             # 鉴权 hook
│   └── package.json
├── mcp_server/                 # MCP 服务层，暴露只读工具
├── storage/                   # 用户、会话、反馈、画像等存储逻辑
├── scripts/                   # 启动与部署脚本（含 MCP 启动脚本）
├── eval/                      # 评测与对比实验脚本
├── data/                      # 课程 docx 原始数据
├── faiss_index/               # 向量索引
├── topo_store/                # 拓扑结构化产物
├── data_store/                # SQLite 数据文件
└── server.py                  # FastAPI API
```

## 论文对齐与实验结论（V4）

本仓库与论文稿的系统设计基本对齐，核心对应关系可以概括为：检索层做细粒度优化，教学层做分层苏格拉底引导，系统层以 FastAPI + React 形成可运行原型。

### 研究问题

- 问题一：在计算机网络实验教学场景中，如何缓解通用 LLM 在课程专有知识上的证据偏差与召回失配，使回答具备可追溯的证据链。
- 问题二：在教学交互中，如何避免“直接给答案”的单轮问答模式，转向可分层推进的苏格拉底式引导，并根据学生状态动态调节提示强度。
- 问题三：如何将检索优化与教学策略整合为可运行、可评估的原型系统，并通过实验验证其在教学场景中的有效性与适用性。

### 研究方案

- 方案一（检索层）：基于“问题类型 + 提示层级”做细粒度检索优化，组合 `BM25 + Dense + RRF + CrossEncoderReranker`，并引入拓扑结构化上下文增强实验问答。
- 方案二（教学层）：构建多场景、多层级苏格拉底提示策略，将实验问题分类后执行分层引导，`hint_level` 按 `MAINTAIN / INCREASE / JUMP_TO_MAX` 动态更新。
- 方案三（系统层）：实现个性化教学 Agent 原型，集成 `FastAPI + React + SSE`、会话状态、反馈与水平估计模块，并通过对比、消融和案例分析进行验证。

### 设计对齐摘要

- 检索层：按问题类别与 `hint_level` 调整检索参数，组合 `BM25 + Dense + RRF + CrossEncoderReranker`。
- 教学层：采用多场景分层提示策略，`hint_level` 按 `MAINTAIN / INCREASE / JUMP_TO_MAX` 动态更新。
- 系统层：FastAPI 后端 + React 前端，支持 `SSE` 流式输出、会话状态管理、反馈闭环与水平估计。

### 论文中的实验配置（摘录）

- 文本抽取结果显示，实验样本规模为 93 条，分组为 30/30/29。
- 检索综合评分采用：
  `Sret = 0.2R + 0.3F + 0.3C + 0.2T`
- 消融实验评分采用：
  `Sabl = 0.15R + 0.20F + 0.15C + 0.15T + 0.20G + 0.15P`
- 统计检验包含 `Wilcoxon` 与 `Cliff's delta`。


## 常见问题

### 1) 启动后首轮响应较慢

这通常是模型和索引在冷启动，属于正常现象。开发模式下会后台预热，第一轮后会明显变快。

### 2) `/api/chat` 返回 401

`/api/chat` 是鉴权接口，先调用登录/注册拿到 token，再在请求头带 `Authorization: Bearer <token>`。

### 3) 前端上传图片后报 OCR 相关错误

后端 `agentic_rag/vision.py` 在收到图片时会尝试做 OCR，依赖 `rapidocr-onnxruntime`。如果只跑文字问答可以忽略；需要 OCR 时安装即可：

```bash
pip install rapidocr-onnxruntime
```

### 4) 拓扑问答提示“未识别到实验编号”

在问题里明确写“实验13”或“lab13”，或者配置 `TOPO_DEFAULT_EXPERIMENT_ID`。

### 5) 需要只看 API，不跑前端

任意一种后端启动方式（`bash scripts/start_dev.sh` / `bash scripts/start_all.sh` / `uvicorn server:app --port 8000`）都不会自动拉起前端。Vite 前端只有在你显式 `cd frontend && npm run dev` 时才会启动。

后端独立跑起来后，用 Postman 或 `curl` 调 `/api/*` 即可。
