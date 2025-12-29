# `retrieval` 模块说明文档

## 1. 模块概述
`retrieval` 模块是 RAG 系统的核心之一，它负责接收用户查询，并利用 `indexing` 模块构建的 BM25 索引和向量索引来检索最相关的文档片段（chunks）。本模块的重点在于实现混合检索（Hybrid Retrieval）策略，以结合稀疏检索（关键词匹配）和密集检索（语义匹配）的优势，提高检索的召回率和精确率。最后，通过重排（Reranking）技术进一步优化结果。

## 2. 关键设计
*   **`Retriever` 类**: 封装混合检索和重排逻辑。
*   **并行检索**: 同时执行 BM25 检索和向量检索，以获取多角度的相关文档。
*   **结果融合**: 采用 Reciprocal Rank Fusion (RRF) 等算法将不同检索源的结果进行融合，兼顾各个检索器的优势。
*   **Top-K 控制**: 灵活控制每个检索器以及最终融合结果的返回数量（Top-K）。
*   **可扩展性**: 设计时考虑未来引入更多检索器或更复杂的重排模型。

## 3. 输入/输出

### 输入
*   `query_text` (str): 用户输入的查询文本。

### 输出
*   `List[Dict]`: 一个排序后的相关 chunk 字典列表。每个 chunk 字典的结构与 `chunking` 模块的输出保持一致，并可能包含额外的检索得分信息（如果需要）。

## 4. 主要类/函数

### `Retriever` 类
*   **`__init__(self, indexer: Indexer, bm25_top_k: int = 5, vector_top_k: int = 5, hybrid_top_k: int = 10)`**: 构造函数，接收 `indexing.Indexer` 实例以及 BM25、向量和混合检索的 Top-K 参数。
*   **`retrieve(self, query_text: str) -> List[Dict]`**: 核心方法，执行混合检索流程。
    *   **内部辅助方法 (私有)**：
        *   `_reciprocal_rank_fusion(bm25_results: List[Dict], vector_results: List[Dict], k: int = 60) -> List[Dict]`: 实现 RRF 融合算法。

## 5. 依赖
*   `src.indexing.indexer`: 用于调用 BM25 和向量索引的查询方法。

## 6. 实现细节

### 检索流程
1.  **初始化 `Retriever`**: 在 `__init__` 中接收 `Indexer` 实例以及配置的 `top_k` 参数。
2.  **`retrieve` 方法:**
    *   **BM25 检索**: 调用 `self.indexer.bm25_store.query(query_text, self.bm25_top_k)` 获取 BM25 检索结果。
    *   **向量检索**: 调用 `self.indexer.vector_store.query(query_text, self.vector_top_k)` 获取向量检索结果。
    *   **结果融合**: 调用 `_reciprocal_rank_fusion` 方法将 BM25 结果和向量结果进行融合。
    *   **返回最终结果**: 返回融合后的前 `hybrid_top_k` 个 chunk 列表。

### `_reciprocal_rank_fusion` 实现细节
1.  **分数计算**: 对于每个检索结果（来自 BM25 和向量检索），根据其排名计算一个 RRF 分数：`1.0 / (k + rank + 1)`，其中 `k` 是一个平滑因子（通常设为 `60`），`rank` 是从 0 开始的排名。
2.  **融合**: 将两个检索结果集中所有唯一 chunk 的 RRF 分数进行累加。
3.  **排序**: 根据累加后的 RRF 分数对所有唯一 chunk 进行降序排序。
4.  **返回**: 返回排序后的 chunk 列表。

### 重排 (Reranking) 考量 (当前阶段为 RRF 融合，未来优化方向)
*   虽然 RRF 是一种简单的重排方法，但如果需要更高级的重排，可以考虑引入一个额外的交叉编码器（Cross-Encoder）模型。这个模型会接收查询和候选文档对，并输出一个相关性分数。在 RRF 融合之后，可以使用交叉编码器对融合后的 Top N 个文档进行二次打分和排序。这通常会放在 `retrieval` 模块中作为可选的步骤。

