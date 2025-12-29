你是一名资深 RAG 架构师 + Python 工程师。请帮我从 0 到 1 构建一个“基于文本检索”的 RAG 系统，用于问答。

【数据】
- 数据是一批 Word 文档（.docx），每个文档包含：正文段落、标题层级、表格、以及图片（图片可能包含文字/图表/截图）。
- 文档：文件夹路径/Users/baoliliu/Downloads/networking-agent/RAG-Agent/data；单文档规模：不确定，10页左右；语言：中英文混合；领域：计算机网络。
- 图片处理策略（请选择并按此实现）：暂不考虑。
- 是否涉密/离线要求：无。

【目标】
1) 建立离线索引（支持增量更新），可根据用户问题检索相关内容并生成回答（RAG）。
2) 检索以“文本”为核心：最终被索引的是文本块（chunk），图片如果做 OCR/caption，也要转为文本纳入索引；同时保留图片文件与其在文档中的位置映射。
3) 回答必须提供可追溯引用：至少包含 docx 文件名 + 章节路径/标题 + 段落/表格位置（可选页码/图片编号）。
4) 需要工程化交付：给出清晰目录结构、依赖、可运行脚本、配置文件、README、以及最小可行 demo。
5) 给出评估方案：如何构造测试集、衡量检索召回与幻觉率，并给出一个 evaluation 脚本雏形。

【技术偏好/约束】
- 语言：Python
- 检索方式偏好：混合检索
- 向量库/搜索引擎偏好：
  - 本地轻量：FAISS 或 Chroma
- 项目需支持：Docx 解析、chunking、metadata 设计、索引构建、查询检索、RAG 生成、引用输出、日志与错误处理。

【输出要求】
请按以下顺序输出：
A. 需求/假设清单：列出你做的默认假设 + 我需要确认的信息（不要一直问，先给默认方案）
B. 架构设计：数据流、模块划分、关键数据结构（chunk schema）
C. 解析策略：如何从 docx 抽取（标题层级/段落/表格/图片）；图片如何命名、如何与文本位置对应
D. chunking 策略：按标题切分、长度阈值、重叠、表格处理规则、去噪规则
E. 索引与检索：BM25/向量/混合检索细节；topK、rerank（如需要）；如何返回引用
F. 生成策略：提示词（system prompt + user prompt 模板），如何强制基于引用回答与拒答
G.评估：构造 QA、指标（Recall@K、MRR、faithfulness 等）、evaluation 脚本雏形，依据测试代码来进一步实现工程代码
H. 工程交付：项目目录结构 + 每个文件职责；给出可运行的核心代码（至少 ingest/build_index/query 三部分）
I. 风险与优化路线：性能、成本、中文分词、长文、表格、图片 OCR 误差等

---

## 项目目录结构与模块职责

```
test-rag/                 # RAG 系统所有相关代码都在这里
├── config.py                 # 全局配置，如模型名称、API key、路径等
├── main.py                   # 系统主入口，提供 ingest, build_index, query 等接口
├── requirements.txt          # 项目依赖
├── README.md             # test-rag 项目的根 README，将更新以反映新结构
├── processed_data/       # 解析后的中间数据和图片存放目录
│   ├── images/
│   │   └── *.png
│   └── chunks.json           # 所有 chunk 的 JSON 列表
├── index_store/              # 索引文件存放目录
│   ├── chroma_db/            # Chroma 向量数据库持久化目录
│   └── bm25_index.pkl        # BM25 索引文件
├── src/                  # RAG 系统的核心代码
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
├── evaluation/           # 评估模块
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
