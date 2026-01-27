<!-- 文件：README.md -->

# 拓扑图驱动多模态RAG + 苏格拉底教学Agent + 多模型评审集成（Networking Labs AI TA）

## 1. 项目背景（Why）
本课题面向 AI 在计算机网络实验课中的三类突出问题：

1) **证据找不准**：实验指导书/拓扑图/CLI 输出信息分散，传统 RAG 常出现召回不相关、跨模态信息对不齐的问题。  
2) **直接给答案**：通用大模型倾向于“给最终配置/最终命令”，抑制学生思考，且在实验教学场景存在“越俎代庖”的教学风险。  
3) **多模型答案难取舍**：不同大模型输出差异大，教师与学生难以判断可靠性与可追溯性。

本项目拟设计一套垂直 AI 助教系统：**拓扑图驱动多模态 RAG + 苏格拉底教学 Agent + 多模型答案评审集成**。前期问卷显示学生在实验原理理解、错误诊断与 AI 回答可靠性方面痛点明显，但对课程官方 AI 助手具有高使用意愿，为系统落地提供真实需求基础。

系统目标：在统一上下文与教学意图下，只输出 **唯一、可追溯、有证据链** 的高质量答复，并支持多轮引导式排错与原理追问；最终与“直接调用通用大模型”基线对比，系统评估检索质量、任务完成度与教学有效性等指标。

---

## 2. 核心贡献（What）
- **Topo-KG（拓扑知识图谱）**：统一表达实验指导书文本、网络拓扑图、CLI 命令与输出的结构化知识基座。
- **Agentic RAG**：在 Topo-KG 基础上，构建多工具、多索引的 RAG Agent，自动生成面向故障排查的“错误诊断链（Diagnosis Chain）”。
- **Socratic Tutor（苏格拉底教学智能体）**：将诊断链结构化为分层提问树，基于 LangGraph + 硬约束策略实现多轮引导式教学对话。
- **Evaluator（多模型评审与集成）**：并行调用多种 LLM，在统一上下文与教学意图下进行 LLM-as-Judge 评审与集成，仅输出唯一且可追溯的答案。

---

## 2.1 项目目录结构与模块职责

```
RAG-Agent/
├── agentic_rag/              # 核心 Agent/RAG/Topo 模块
│   ├── agent.py              # 入口：问题分类 + 提示策略 + 工具调用 + 对话状态
│   ├── rag.py                # 文本 RAG：docx→分块→FAISS→MMR 检索 + 引用
│   ├── topo_rag.py           # 拓扑图解析/索引/检索（含拓扑 JSON/Graph）
│   ├── socratic/
│   │   └── ping_controller.py # Ping 场景苏格拉底排错控制器
│   ├── llm_config.py         # DeepSeek/OpenAI Chat LLM 构建
│   ├── embedding.py          # 简易向量化脚本（FAISS）
│   ├── utils.py              # RAG 输出清洗与证据摘要
│   └── __init__.py
├── server.py                 # FastAPI 后端 API（/api/chat）
├── app_streamlit.py          # Streamlit 前端
├── data/                     # 原始实验指导书（.docx）
├── faiss_index/              # 文本向量索引（FAISS）
├── topo_store_lab1/          # 拓扑索引与解析结果（示例）
├── topo_store_lab13/         # 默认拓扑索引与解析结果
├── scripts/
│   └── download_bge_m3.py     # 本地嵌入模型下载脚本
├── generate_qa_dataset.py    # QA 数据集生成脚本（路径参数当前写死）
├── sessions.json             # Streamlit 会话持久化（可选）
├── evaluator/                # 预留评审模块（当前为空）
└── socratic_tutor/            # 预留 Tutor 模块（当前为空）
```

### `agentic_rag/agent.py` 职责概述
`agentic_rag/agent.py` 是当前系统的核心入口与调度器，负责：
1. **相关性守卫**：判断问题是否与网络课程相关，不相关则拒答。
2. **问题分类与提示策略**：将问题分为 LAB/THEORY/REVIEW/CALC，动态选择系统提示词与 Hint Level。
3. **工具调用**：按“工具：检索/拓扑：query”规范调用 `RAGAgent` 与 `TopoRetriever`。
4. **苏格拉底引导**：对 Ping 场景切换到 `socratic/ping_controller.py` 的槽位式排错流程。
5. **会话与状态**：维护 `history`、`tool_traces` 与 `state`（hint_level、question_category 等）。

---

## 3. 总体架构（How）
### 3.1 模块分层
1) **数据与知识层**
- Word 实验指导书（含拓扑图/示意图）
- 图像理解产物（caption、结构化拓扑 Topo-JSON）
- Topo-KG（图数据库或轻量图结构）

2) **检索与推理层**
- 多模态索引：文本向量检索 / 图示检索（caption 或图向量）/ 拓扑图查询（Topo-KG）
- Agentic RAG：工具化检索引擎 + 诊断链生成

3) **教学与评审层**
- Socratic Tutor：对话状态机 + 提示阶梯 + 学生模型
- Evaluator：多模型候选答案 + 评审打分 + 答案集成

### 3.2 推荐技术路线（默认）
- **LLM 接入**：`langchain_deepseek.ChatDeepSeek` 统一封装，优先使用 DeepSeek（若 `DEEPSEEK_API_KEY` 存在），否则回退 OpenAI（`OPENAI_API_KEY`）。
- **文本 RAG**：`Docx2txtLoader` 读取实验指导书 → `RecursiveCharacterTextSplitter` 分块 → `HuggingFaceEmbeddings`（默认 `BAAI/bge-m3`）→ `FAISS` 向量库 → MMR 检索。
- **拓扑理解**：`python-docx` 提取 docx 图像 → OpenAI Responses API 结构化解析为 `TopologyExtraction` → `networkx` 构图 + FAISS 检索 → 返回拓扑上下文文本。
- **服务与前端**：FastAPI 提供 `/api/chat`，Streamlit 提供对话 UI 与会话持久化。

---

## 4. 关键数据契约（AI/开发必须遵守）
> 任何模块改动都不得破坏这些契约；如需变更，必须同步更新下游与测试样例。

### 4.1 对话请求/响应与消息格式
`/api/chat` 请求与响应结构（与 `server.py` 一致）：

```json
// ChatRequest
{
  "message": "用户问题",
  "session_id": "可选",
  "history": [{"role": "user|assistant|system", "content": "..."}],
  "debug": false,
  "max_turns": 5
}
```

```json
// ChatResponse
{
  "session_id": "s_xxx",
  "reply": "模型回复",
  "history": [{"role": "user|assistant|system", "content": "..."}],
  "tool_traces": [{"tool": "检索|拓扑", "input": "...", "output": "..."}]
}
```

### 4.2 工具调用与证据结构
- 工具调用格式：`工具：检索：{query}` 或 `工具：拓扑：{query}`。
- RAGAgent 返回：`{"answer": "...", "citations": [{"id": 1, "source": "xx.docx", "snippet": "..."}]}`。
- 证据记录结构（用于教学链/调试）：

```json
{
  "id": "E1",
  "query": "检索问题",
  "excerpt": "证据摘要",
  "raw_text": "工具原始输出"
}
```

### 4.3 拓扑结构化数据（TopologyExtraction）
拓扑图结构化输出必须符合以下字段（与 `topo_rag.py` 一致）：

```json
{
  "schema_version": 2,
  "devices": [{"name": "R1", "type": "router|switch|host|firewall|server|unknown", "mgmt_ip": "可选"}],
  "interfaces": [{
    "device": "R1",
    "name": "GE0/0/1",
    "kind": "physical|svi|host_nic|unknown",
    "mode": "access|trunk|unknown",
    "allowed_vlans": "5-6",
    "access_vlan": "5",
    "ip": "可选",
    "mask": "可选",
    "vlan": "可选",
    "ip_raw": "原始值（含不确定字符）",
    "mask_raw": "原始值（含不确定字符）",
    "vlan_raw": "原始值（含不确定字符）"
  }],
  "links": [{"a": {"device": "R1", "interface": "GE0/0/1"}, "b": {"device": "SW1", "interface": "GE0/0/2"}, "medium": "unknown"}],
  "subnets": [{"cidr": "192.168.1.0/24", "members": [{"device": "R1", "interface": "GE0/0/1"}]}],
  "warnings": ["解析不确定性说明"]
}
```

## 5.启动方法
启动后端:
uvicorn server:app --reload --port 8000
启动前端:
streamlit run app_streamlit.py

---

## 6. 本地嵌入模型（协作建议）
为了避免将 2GB+ 模型推到 GitHub，推荐使用脚本下载到本地并通过环境变量指向。

### 6.1 启动前设置路径
```bash
export EMBEDDING_MODEL_NAME=./models/bge-m3
```

### 6.2 网络不稳定时设置镜像，则不需开VPN（可选，推荐设置）
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 6.3 一键下载（推荐）
```bash
python scripts/download_bge_m3.py
```
默认下载到 `models/bge-m3/`（已在 `.gitignore` 中忽略）。