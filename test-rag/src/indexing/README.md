# `indexing` 模块说明文档

## 1. 模块概述
`indexing` 模块负责管理和构建 RAG 系统所需的两种主要索引：BM25 稀疏索引和向量密集索引。它接收来自 `chunking` 模块生成的 chunk 列表，并将其内容和元数据存储到相应的索引结构中，以便 `retrieval` 模块进行高效检索。

## 2. 关键设计
*   **`Indexer` 类**: 作为索引构建的协调器，统一管理 BM25 和向量索引的构建过程。
*   **`VectorStore` 类**: 封装与向量数据库（Chroma）的交互逻辑，包括文档添加、查询等。
*   **`BM25Store` 类**: 封装 BM25 索引的构建和查询逻辑，针对中文分词进行优化。
*   **持久化**: 支持将构建好的索引持久化到磁盘，以便系统重启后快速加载。

## 3. 输入/输出

### 输入
*   `chunks` (List[Dict]): `chunking` 模块的输出，即一个 chunk 字典的列表。每个 chunk 字典包含 `chunk_id`, `doc_id`, `content_text`, `metadata` 等。

### 输出
*   无直接输出（构建过程是副作用），但会创建或更新以下文件：
    *   `index_store/chroma_db/`: Chroma 向量数据库的持久化目录。
    *   `index_store/bm25_index.pkl`: BM25 索引的 Python pickle 文件。

## 4. 主要类/函数

### `Indexer` 类
*   **`__init__(self, chroma_db_dir: str, bm25_index_file: str, embedding_model)`**: 构造函数，初始化 `VectorStore` 和 `BM25Store` 实例，并传入嵌入模型。
*   **`build_indexes(self, chunks: List[Dict])`**: 核心方法，接收 chunk 列表，并分别调用 `VectorStore` 和 `BM25Store` 的方法来构建索引。

### `VectorStore` 类
*   **`__init__(self, persist_directory: str, embedding_model)`**: 构造函数，初始化 Chroma 客户端和集合，并指定使用的嵌入模型（`text-embedding-3-small`）。
*   **`add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str])`**: 向 Chroma 集合添加文档。
*   **`query(self, query_text: str, top_k: int = 5) -> List[Dict]`**: 查询 Chroma 集合，返回最相似的 `top_k` 个 chunk。

### `BM25Store` 类
*   **`__init__(self, index_file: str)`**: 构造函数，初始化 BM25 索引，并尝试从文件加载已有索引。
*   **`build_index(self, texts: List[str], metadatas: List[Dict], ids: List[str])`**: 构建 BM25 索引。
*   **`query(self, query_text: str, top_k: int = 5) -> List[Dict]`**: 查询 BM25 索引，返回最相关的 `top_k` 个 chunk。
    *   **内部辅助方法 (私有)**：
        *   `_tokenize(text: str) -> List[str]`: 使用 `jieba` 对中文文本进行分词。

## 5. 依赖
*   `chromadb`: 向量数据库。
*   `rank_bm25`: BM25 索引实现。
*   `jieba`: 中文分词库。
*   `pickle`: 用于 BM25 索引的持久化。
*   `src.models.embedding_model`: 用于获取嵌入模型实例。

## 6. 实现细节

### `Indexer` 实现流程
1.  **初始化**: 在 `Indexer` 的 `__init__` 中，传入 `chroma_db_dir`, `bm25_index_file` 和 `embedding_model` 实例。创建 `VectorStore` 和 `BM25Store` 实例。
2.  **`build_indexes` 方法:**
    *   从输入的 `chunks` 列表中提取 `content_text` (作为 `texts`)、完整的 chunk 字典 (作为 `metadatas`) 和 `chunk_id` (作为 `ids`)。
    *   调用 `self.vector_store.add_documents(texts, metadatas, ids)` 来构建向量索引。
    *   调用 `self.bm25_store.build_index(texts, metadatas, ids)` 来构建 BM25 索引。

### `VectorStore` 实现细节
1.  **初始化**: 使用 `chromadb.PersistentClient(path=persist_directory)` 创建持久化客户端。
2.  **嵌入函数**: 在 `__init__` 中，通过 `embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model.model_name)` (或直接使用 OpenAI 的嵌入 API，需要自定义 `EmbeddingFunction` 适配 `chromadb`) 来设置嵌入函数。这里我们将直接与 `embedding_model` 模块交互，而不是通过 `SentenceTransformer`。
    *   **注意**: ChromaDB 需要一个实现 `embed_documents` 和 `embed_query` 方法的 `EmbeddingFunction`。如果直接使用 OpenAI 的 `text-embedding-3-small` API，需要包装成符合 ChromaDB 要求的 `EmbeddingFunction`。
3.  **`add_documents`**: 直接调用 Chroma 集合的 `add` 方法。
4.  **`query`**: 调用 Chroma 集合的 `query` 方法，将 `query_text` 传递给嵌入函数进行向量化，然后进行相似性搜索。

### `BM25Store` 实现细节
1.  **初始化**: 在 `__init__` 中，尝试从 `index_file` 加载之前保存的 BM25 索引。如果文件不存在，则 `bm25` 实例为 `None`。
2.  **`_tokenize` 方法**: 这是 BM25 的关键。对于中文文本，必须使用 `jieba.cut(text)` 进行分词，然后返回词语列表。
3.  **`build_index` 方法:**
    *   遍历输入的 `texts`，对每个文本使用 `_tokenize` 方法进行分词，构建 `tokenized_corpus`。
    *   使用 `BM25Okapi(tokenized_corpus)` 初始化 BM25 索引。
    *   保存 `bm25` 实例、原始 `corpus`、`metadatas` 和 `ids` 到 `index_file`。
4.  **`query` 方法:**
    *   对 `query_text` 使用 `_tokenize` 进行分词。
    *   调用 `self.bm25.get_scores(tokenized_query)` 获取每个文档的得分。
    *   根据得分排序，返回 `top_k` 个文档对应的 `metadata`。注意，这里需要将 `content_text` 从 `corpus` 中重新填充到返回的 `metadata` 字典中，因为 `metadatas` 存储的是完整的 chunk 字典，而 `corpus` 存储的是纯文本内容。

