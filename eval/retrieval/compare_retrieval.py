"""
检索方案对比测试脚本
用法：python eval/compare_retrieval.py

对比：
  - 旧方案：纯 MMR 向量检索
  - 新方案：MMR + BGE Reranker 精排

输出每个测试查询下两种方案各自检索到的文档片段，便于人工判断精度差异。
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 设置环境变量（不重建索引，加速测试）
os.environ.setdefault("RAG_REBUILD_INDEX", "0")

TEST_QUERIES = [
    "show ip route 命令的输出如何解读",
    "OSPF 路由协议的工作原理",
    "ping 不通应该如何排查故障",
    "如何配置交换机 VLAN",
    "子网划分的计算方法",
    "access-list 访问控制列表配置",
]

SNIPPET_LEN = 200  # 显示的片段长度


def load_base_retriever():
    """加载纯 MMR 检索器（禁用 Reranker）"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    INDEX_DIR = ROOT / "faiss_index"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.5},
    )


def load_reranker_retriever():
    """粗检索 20 条 → BGE Reranker 精排取 6 条"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

    INDEX_DIR = ROOT / "faiss_index"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # 过检索：给 reranker 充足的候选池
    base_retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5},
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=6)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


def format_docs(docs, snippet_len=SNIPPET_LEN):
    lines = []
    for i, doc in enumerate(docs, 1):
        src = (doc.metadata or {}).get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ").strip()[:snippet_len]
        lines.append(f"  [{i}] {src}\n      {snippet}")
    return "\n".join(lines) if lines else "  (无结果)"


def run_comparison():
    print("=" * 70)
    print("加载向量库和检索器（首次运行需下载 Reranker 模型，请稍候）...")
    print("=" * 70)

    base_retriever = load_base_retriever()
    reranker_retriever = load_reranker_retriever()

    print("\n模型加载完毕，开始对比测试\n")

    for query in TEST_QUERIES:
        print("=" * 70)
        print(f"查询：{query}")
        print("-" * 70)

        # 旧方案
        old_docs = base_retriever.invoke(query)
        print(f"[旧] 纯 MMR（{len(old_docs)} 条）：")
        print(format_docs(old_docs))

        print()

        # 新方案
        new_docs = reranker_retriever.invoke(query)
        print(f"[新] MMR + Reranker（{len(new_docs)} 条）：")
        print(format_docs(new_docs))

        print()


if __name__ == "__main__":
    run_comparison()
