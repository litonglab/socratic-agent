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
├── config.py                 # 全局配置，如模型名称、API key、路径等
├── main.py                   # 系统主入口，提供 ingest, build_index, query 等接口
├── requirements.txt          # 项目依赖
├── README.md                 # 项目介绍、安装、使用说明
├── data/                     # 原始 Word 文档存放目录
│   └── *.docx
├── processed_data/           # 解析后的中间数据和图片存放目录
│   ├── images/
│   │   └── *.png
│   └── chunks.json           # 所有 chunk 的 JSON 列表
├── index_store/              # 索引文件存放目录
│   ├── chroma_db/            # Chroma 向量数据库持久化目录
│   └── bm25_index.pkl        # BM25 索引文件
├── src/
│   ├── data_loader/
│   │   ├── document_parser.py    # Word 文档解析，提取文本、图片、表格
│   │   ├── file_utils.py         # 文件操作工具
│   │   └── README.md             # data_loader 模块的说明文档
│   ├── chunking/
│   │   ├── text_splitter.py      # 文档分块策略实现
│   │   └── README.md             # chunking 模块的说明文档
│   ├── indexing/
│   │   ├── indexer.py            # 索引构建器 (BM25 + Vector)
│   │   ├── vector_store.py       # 向量数据库操作封装
│   │   ├── bm25_store.py         # BM25 索引操作封装
│   │   └── README.md             # indexing 模块的说明文档
│   ├── retrieval/
│   │   ├── retriever.py          # 混合检索逻辑，RRF 重排
│   │   └── README.md             # retrieval 模块的说明文档
│   ├── rag/
│   │   ├── generator.py          # RAG 生成器，LLM 调用及提示词管理
│   │   └── README.md             # rag 模块的说明文档
│   ├── models/                   # 存放嵌入模型和 LLM 客户端的初始化
│   │   ├── embedding_model.py
│   │   ├── llm_client.py
│   │   └── README.md             # models 模块的说明文档
│   └── __init__.py
├── evaluation/
│   ├── qa_dataset.json       # 评估数据集
│   ├── evaluator.py
│   └── README.md                 # evaluation 模块的说明文档
└── tests/                    # 单元测试和集成测试
    └── test_*.py
```

### `main.py` 职责概述
`main.py` 是整个 RAG 系统的入口文件。它负责协调各个模块，提供以下主要功能：
1.  **文档摄取 (Ingest)**: 调用 `data_loader` 和 `chunking` 模块来解析原始文档并生成 chunk。
2.  **构建索引 (Build Index)**: 调用 `indexing` 模块来构建和持久化 BM25 和向量索引。
3.  **查询 (Query)**: 接收用户问题，调用 `retrieval` 模块获取相关 chunk，然后调用 `rag` 模块生成最终回答。
4.  **评估 (Evaluate)**: （可选）集成 `evaluation` 模块，对系统性能进行评估。
5.  **命令行接口 (CLI)**: 可以设计为通过命令行参数来触发上述功能。

`main.py` 不包含具体的业务逻辑，而是作为粘合剂，将各个功能模块串联起来，形成一个完整的 RAG 问答系统。

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
- **LangGraph**：负责“可控流程编排（状态机、并发、分支、重试、持久化）”，适合 Tutor 与 Evaluator 的确定性链路。
- **检索/索引实现**：可用 LlamaIndex（更偏“RAG 引擎与工具封装”）或 LangChain 的 Retriever 体系。  
  - 推荐组合：**LangGraph（编排） + LlamaIndex（索引/QueryEngine 工具化）**，减少重复造轮子。

> 若你希望“全栈统一”，也可以选择：LangChain/LangGraph 全套（Retriever + Tools + Graph）；或 LlamaIndex（Index + Agent）为主再少量流程化。当前文档以“LangGraph 编排”为默认。

---

## 4. 关键数据契约（AI/开发必须遵守）
> 任何模块改动都不得破坏这些契约；如需变更，必须同步更新下游与测试样例。

### 4.1 统一知识节点 Node（文本/图/拓扑）
最小字段建议如下（JSON）：

```json
{
  "node_id": "string",
  "doc_id": "word_filename_or_lab_id",
  "section_path": ["H1", "H2", "H3"],
  "modality": "text | figure | topo | table",
  "content_text": "clean text or caption/interpretation",
  "assets": {
    "image_path": "optional",
    "figure_context": "optional text around the figure"
  },
  "topo_json": { "optional": "only for topo nodes" },
  "metadata": {
    "lab": "e.g., OSPF",
    "topic": "e.g., neighbor",
    "difficulty": "intro|mid|adv",
    "tags": ["troubleshooting", "routing"]
  }
}

## 5.启动方法
启动后端：
uvicorn server:app --reload --port 8000


启动前端：
streamlit run app_streamlit.py