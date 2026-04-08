"""
检索实验配置文件
定义索引方案、检索方案、以及要运行的实验组合。

修改此文件来添加新的实验方案，无需改动评测代码。
"""

# ══ 索引方案：不同的文档分块策略 ══════════════════════════════
# 每种方案生成独立的 FAISS 索引，存储在不同目录
INDEX_VARIANTS = {
    # 兼容旧实验：复用已有的生产索引
    "baseline": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index",  # 复用现有生产索引
    },
    # 新实验基线：独立构建 500 字符 chunk，避免覆盖生产索引
    "baseline_500": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/baseline_500",
    },
    # 加大 chunk，减少截断
    "large_chunk": {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/large_chunk",
    },
    # 加大 chunk + 上下文前缀（chunk 文本开头加 "【实验N-名称】"）
    "enriched": {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "context_prefix": True,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/enriched",
    },

    # ══ Chunk Size Sweep（固定 hybrid_rerank，只改分块大小）══════
    "chunk_300": {
        "chunk_size": 300,
        "chunk_overlap": 60,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/chunk_300",
    },
    "chunk_1000": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/chunk_1000",
    },
    "chunk_1200": {
        "chunk_size": 1200,
        "chunk_overlap": 240,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/chunk_1200",
    },
    "chunk_1500": {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "context_prefix": False,
        "separators": ["\n\n", "\n", "。", "，", " ", ""],
        "index_dir": "faiss_index/chunk_1500",
    },
}

# ══ 检索方案：不同的检索算法 ══════════════════════════════════
# 同一套索引上可以跑不同的检索方案
RETRIEVER_VARIANTS = {
    "similarity": {
        "type": "similarity",
        "k": 6,
    },
    "mmr": {
        "type": "mmr",
        "k": 6,
        "fetch_k": 30,
        "lambda_mult": 0.5,
    },
    "mmr_rerank": {
        "type": "mmr_rerank",
        "k": 6,          # 最终返回文档数
        "fetch_k": 50,    # FAISS 候选数
        "base_k": 20,     # MMR 粗筛数
    },
    "hybrid": {
        "type": "hybrid",
        "k": 6,
        "bm25_k": 10,
        "dense_k": 10,
        "dense_search_type": "similarity",
        "bm25_weight": 0.45,
        "dense_weight": 0.55,
    },
    "hybrid_rerank": {
        "type": "hybrid_rerank",
        "k": 6,
        "bm25_k": 12,
        "dense_k": 12,
        "fusion_k": 18,
        "dense_search_type": "similarity",
        "bm25_weight": 0.45,
        "dense_weight": 0.55,
    },
}

# ══ 实验矩阵 ════════════════════════════════════════════════
# 选择要运行的 (索引, 检索) 组合，不需要跑全部
EXPERIMENTS = [
    # ── 基线索引上的对照实验 ──
    {"name": "baseline_similarity",     "index": "baseline_500", "retriever": "similarity"},
    {"name": "baseline_mmr",            "index": "baseline_500", "retriever": "mmr"},
    {"name": "baseline_mmr_rerank",     "index": "baseline_500", "retriever": "mmr_rerank"},
    {"name": "baseline_hybrid",         "index": "baseline_500", "retriever": "hybrid"},
    {"name": "baseline_hybrid_rerank",  "index": "baseline_500", "retriever": "hybrid_rerank"},

    # ── 上下文增强索引 ──
    {"name": "enriched_hybrid_rerank",  "index": "enriched", "retriever": "hybrid_rerank"},

    # ══ Chunk Size Sweep：固定 hybrid_rerank，只改分块大小 ══════
    {"name": "chunk800_hybrid_rerank",  "index": "large_chunk", "retriever": "hybrid_rerank"},
    {"name": "chunk1000_hybrid_rerank", "index": "chunk_1000",  "retriever": "hybrid_rerank"},
    {"name": "chunk1200_hybrid_rerank", "index": "chunk_1200",  "retriever": "hybrid_rerank"},
    {"name": "chunk1500_hybrid_rerank", "index": "chunk_1500",  "retriever": "hybrid_rerank"},
]

# ══ 评测通用配置 ════════════════════════════════════════════
EVAL_CONFIG = {
    "judge_model": "gpt-4o",
    "snippet_len": 300,        # 送给评委的检索片段截断长度
    "answer_snippet": 1200,    # 送给评委的回答截断长度
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
}
