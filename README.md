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

如果你需要在 Streamlit 聊天中上传图片并做 OCR，建议额外安装：

```bash
pip install rapidocr-onnxruntime
```

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
| `PERSIST_PATH` | `sessions.json` | Streamlit 本地会话落盘路径 |
| `HF_ENDPOINT` | 空 | HuggingFace 镜像地址（网络不稳定时可配） |

说明：`.env.example` 里的 `BACKEND_BASE_URL` 当前版本未被核心代码使用，可保留不动。

### 3) 准备课程文档与索引

- 把实验指导书（`.docx`）放到 `data/`。
- 默认会优先读取已有索引，不强制重建。
- 如果希望从文档重新切分并生成索引，可临时开启：

```bash
export RAG_REBUILD_INDEX=1
```

### 4) 启动服务

开发模式（推荐）：

```bash
bash scripts/start_dev.sh
```

- 启动 `uvicorn --reload` 与 Streamlit。
- 默认开启 `RAG_DEV_FAST_START=1`，后端会先可访问，再后台预热 RAG。

稳定模式（演示或部署前自测）：

```bash
bash scripts/start_all.sh
```

- 启动后端并等待健康检查通过，再拉起 Streamlit。
- 默认后端端口 `8000`，Streamlit 端口 `8501`。

也可分开启动：

```bash
uvicorn server:app --reload --port 8000
streamlit run app_streamlit.py
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
  --topo-ratio 0.4 \
  --target-size 93 \
  --output eval/qa_dataset_topo_balanced.json
```

脚本会合并 `eval/topology_question_bank.json` 并自动打上 `requires_topology` 字段，随后你可以在评测脚本里显式指定数据集路径，例如：

```bash
python eval/ablation_study.py --dataset eval/qa_dataset_topo_balanced.json
python eval/performance_benchmark.py --dataset eval/qa_dataset_topo_balanced.json
python eval/judge_consistency.py --dataset eval/qa_dataset_topo_balanced.json
python eval/topology_evaluation.py --questions-file eval/topology_question_bank.json
```

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
│   └── web_search.py
├── components/                # Streamlit 自定义输入组件
├── storage/                   # 用户、会话、反馈、画像等存储逻辑
├── scripts/                   # 启动与部署脚本
├── eval/                      # 评测与对比实验脚本
├── data/                      # 课程 docx 原始数据
├── faiss_index/               # 向量索引
├── topo_store/                # 拓扑结构化产物
├── data_store/                # SQLite 数据文件
├── server.py                  # FastAPI API
└── app_streamlit.py           # Streamlit UI
```

## 论文对齐与实验结论（V4）

本仓库与论文稿的系统设计基本对齐，核心对应关系可以概括为：检索层做细粒度优化，教学层做分层苏格拉底引导，系统层以 FastAPI + Streamlit 形成可运行原型。

### 研究问题

- 问题一：在计算机网络实验教学场景中，如何缓解通用 LLM 在课程专有知识上的证据偏差与召回失配，使回答具备可追溯的证据链。
- 问题二：在教学交互中，如何避免“直接给答案”的单轮问答模式，转向可分层推进的苏格拉底式引导，并根据学生状态动态调节提示强度。
- 问题三：如何将检索优化与教学策略整合为可运行、可评估的原型系统，并通过实验验证其在教学场景中的有效性与适用性。

### 研究方案

- 方案一（检索层）：基于“问题类型 + 提示层级”做细粒度检索优化，组合 `BM25 + Dense + RRF + CrossEncoderReranker`，并引入拓扑结构化上下文增强实验问答。
- 方案二（教学层）：构建多场景、多层级苏格拉底提示策略，将实验问题分类后执行分层引导，`hint_level` 按 `MAINTAIN / INCREASE / JUMP_TO_MAX` 动态更新。
- 方案三（系统层）：实现个性化教学 Agent 原型，集成 `FastAPI + Streamlit + SSE`、会话状态、反馈与水平估计模块，并通过对比、消融和案例分析进行验证。

### 设计对齐摘要

- 检索层：按问题类别与 `hint_level` 调整检索参数，组合 `BM25 + Dense + RRF + CrossEncoderReranker`。
- 教学层：采用多场景分层提示策略，`hint_level` 按 `MAINTAIN / INCREASE / JUMP_TO_MAX` 动态更新。
- 系统层：支持 `SSE` 流式输出、会话状态管理、反馈闭环与水平估计。

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

### 3) 提示缺少 OCR 依赖

安装 `rapidocr-onnxruntime`，该依赖在上传图片并执行 OCR 时会被调用。

### 4) 拓扑问答提示“未识别到实验编号”

在问题里明确写“实验13”或“lab13”，或者配置 `TOPO_DEFAULT_EXPERIMENT_ID`。

### 5) 需要只看 API，不跑前端

直接运行：

```bash
uvicorn server:app --port 8000
```

随后用 Postman 或 `curl` 调 `/api/*` 即可。
