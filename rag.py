import os
import pickle
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# -------------------------
# 路径与环境配置（廉价，保持模块级）
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
_INDEX_ENV = os.getenv("RAG_INDEX_DIR")
if _INDEX_ENV:
    INDEX_DIR = BASE_DIR / _INDEX_ENV
else:
    _preferred_index = BASE_DIR / "faiss_index" / "enriched"
    INDEX_DIR = _preferred_index if _preferred_index.exists() else BASE_DIR / "faiss_index"
REBUILD_INDEX = os.getenv("RAG_REBUILD_INDEX", "0").lower() in {"1", "true", "yes"}
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
DISABLE_RERANKER = os.getenv("DISABLE_RERANKER", "0").lower() in {"1", "true", "yes"}
ENABLE_QUERY_FUSION = os.getenv("RAG_QUERY_FUSION", "1").lower() not in {"0", "false", "no"}

# -------------------------
# 延迟初始化状态（首次查询时才真正加载）
# -------------------------
_loaded_vectorstore = None
_loaded_chunks = None
_cross_encoder = None
_retriever_cache: dict = {}
_initialized = False
_init_lock = threading.Lock()

# -------------------------
# 文本分割器（无 IO，可保持模块级）
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " ", ""],
    length_function=len,
    add_start_index=True,
)

@dataclass(frozen=True)
class RetrievalProfile:
    """不同教学场景的召回画像。"""

    fetch_k: int
    top_n: int
    bm25_k: int
    dense_k: int
    fusion_k: int
    bm25_weight: float
    dense_weight: float
    dense_search_type: str = "similarity"
    dense_lambda_mult: float = 0.5


# 7. 教学意图感知检索策略参数表
# 设计原则：
# - 理论题更依赖语义召回，使用 MMR 降低相似片段堆叠。
# - 排错/配置题包含大量命令、接口名和错误现象，BM25 权重更高。
# - 计算题候选池更窄，避免无关实验步骤污染公式与数值推理。
_RETRIEVAL_PROFILES: dict = {
    "THEORY_CONCEPT": {
        "low": RetrievalProfile(40, 4, 12, 22, 28, 0.35, 0.65, "mmr", 0.35),
        "high": RetrievalProfile(46, 6, 14, 28, 36, 0.35, 0.65, "mmr", 0.35),
    },
    "LAB_TROUBLESHOOTING": {
        "low": RetrievalProfile(28, 3, 14, 16, 24, 0.55, 0.45, "mmr", 0.30),
        "high": RetrievalProfile(36, 4, 18, 22, 32, 0.55, 0.45, "mmr", 0.30),
    },
    "CONFIG_REVIEW": {
        "low": RetrievalProfile(30, 3, 16, 16, 26, 0.60, 0.40, "similarity", 0.5),
        "high": RetrievalProfile(38, 5, 20, 22, 34, 0.60, 0.40, "similarity", 0.5),
    },
    "CALCULATION": {
        "low": RetrievalProfile(22, 3, 10, 14, 20, 0.45, 0.55, "similarity", 0.5),
        "high": RetrievalProfile(26, 4, 12, 16, 24, 0.45, 0.55, "similarity", 0.5),
    },
}
_DEFAULT_PROFILE = RetrievalProfile(30, 4, 12, 18, 24, 0.45, 0.55, "mmr", 0.4)

_CATEGORY_QUERY_HINTS = {
    "THEORY_CONCEPT": ["原理 机制 概念 作用 特点"],
    "LAB_TROUBLESHOOTING": ["故障 排查 原因 现象 解决 连通性"],
    "CONFIG_REVIEW": ["配置 命令 示例 错误 修正"],
    "CALCULATION": ["公式 计算 步骤 掩码 地址范围"],
}

_DOMAIN_QUERY_HINTS = [
    (re.compile(r"\bping\b|不通|连通", re.IGNORECASE), "ICMP 超时 目的主机不可达 连通性检查"),
    (re.compile(r"\bvlan\b|虚拟局域网", re.IGNORECASE), "VLAN access trunk 端口划分 交换机"),
    (re.compile(r"\bospf\b", re.IGNORECASE), "OSPF 邻居 路由协议 链路状态"),
    (re.compile(r"\bacl\b|access-list|访问控制", re.IGNORECASE), "ACL access-list permit deny 访问控制列表"),
    (re.compile(r"\bnat\b|地址转换", re.IGNORECASE), "NAT inside outside 地址转换"),
    (re.compile(r"子网|掩码|网段|广播地址", re.IGNORECASE), "子网掩码 网络地址 广播地址 可用主机数"),
    (re.compile(r"路由|route|下一跳", re.IGNORECASE), "路由表 show ip route 下一跳 静态路由"),
    (re.compile(r"网线|双绞线|线序", re.IGNORECASE), "双绞线 线序 直通线 交叉线 RJ45"),
]


# -------------------------
# 纯函数辅助（无 IO）
# -------------------------

def _extract_section(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    patterns = [
        r"^第.+(章|节).*$",
        r"^实验\s*\d+.*$",
        r"^\d+(?:\.\d+){0,3}\s+.+$",
    ]
    for ln in lines[:10]:
        for p in patterns:
            if re.match(p, ln):
                return ln
    return None


def _extract_experiment_prefix(source_name: str) -> str:
    match = re.match(r"(实验\d+.*?)（", source_name)
    return match.group(1) if match else ""


def _with_context_prefix(doc):
    meta = dict(doc.metadata or {})
    source_name = meta.get("source", "")
    exp_info = _extract_experiment_prefix(source_name)
    section = meta.get("section", "")

    prefix_parts = []
    if exp_info:
        prefix_parts.append(f"【{exp_info}】")
    if section and section != exp_info:
        prefix_parts.append(section)

    if prefix_parts:
        prefix = "\n".join(prefix_parts)
        content = (doc.page_content or "").strip()
        if not content.startswith(prefix):
            doc.page_content = f"{prefix}\n{content}"
    return doc


def _enrich_metadata(chunks, source_path: str) -> List:
    source_name = Path(source_path).name
    for i, doc in enumerate(chunks):
        meta = dict(doc.metadata or {})
        meta["source"] = source_name
        meta["chunk_id"] = f"{Path(source_path).stem}-{i:05d}"
        meta["start_index"] = meta.get("start_index")
        section = _extract_section(doc.page_content)
        if section:
            meta["section"] = section
        doc.metadata = meta
    return chunks


def normalize_source_to_filename(docs):
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path")
        if src:
            d.metadata["source"] = Path(src).name
        else:
            d.metadata["source"] = "unknown"
    return docs


def _prepare_runtime_chunks(docs: List, add_context_prefix: bool = False) -> List:
    docs = normalize_source_to_filename(docs)
    if add_context_prefix:
        docs = [_with_context_prefix(doc) for doc in docs]
    return docs


def _load_single_docx(docx_path: Path) -> List:
    """加载并分块单个 docx 文件，供并行调用。"""
    loader = Docx2txtLoader(str(docx_path))
    try:
        docs = loader.load()
    except zipfile.BadZipFile:
        print(f"[Skip] 非法 docx 文件（不是 zip）：{docx_path}")
        return []
    except Exception as e:
        print(f"[Skip] 读取 docx 失败：{docx_path}，原因：{e}")
        return []
    chunks = text_splitter.split_documents(docs)
    chunks = _enrich_metadata(chunks, str(docx_path))
    return chunks


def _load_chunks_from_index(index_dir: Path) -> Optional[List]:
    chunks_path = index_dir / "chunks.pkl"
    if not chunks_path.exists():
        return None
    with open(chunks_path, "rb") as f:
        docs = pickle.load(f)
    return _prepare_runtime_chunks(docs, add_context_prefix=False)


def _doc_key(doc) -> str:
    meta = dict(doc.metadata or {})
    chunk_id = meta.get("chunk_id")
    start_index = meta.get("start_index")
    source = meta.get("source", "unknown")
    if chunk_id:
        return f"{source}:{chunk_id}:{start_index}"
    return f"{source}:{(doc.page_content or '')[:200]}"


class _RRFRetriever:
    """将 BM25 与 Dense 结果做简单 RRF 融合。"""

    def __init__(self, retrievers, weights, k=6, rrf_k: int = 60):
        self.retrievers = retrievers
        self.weights = weights
        self.k = k
        self.rrf_k = rrf_k

    def invoke(self, query: str):
        doc_scores = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                if key not in doc_scores:
                    doc_scores[key] = {"doc": doc, "score": 0.0}
                doc_scores[key]["score"] += weight / (rank + self.rrf_k)
        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[: self.k]]

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


class _QueryFusionRetriever:
    """RAG-Fusion 风格的轻量多查询召回，避免一次查询表述过窄。"""

    def __init__(self, base_retriever, category: Optional[str], k=6, enabled: bool = True):
        self.base_retriever = base_retriever
        self.category = category
        self.k = k
        self.enabled = enabled

    def invoke(self, query: str):
        variants = _build_query_variants(query, self.category) if self.enabled else [query]
        doc_scores = {}
        for query_rank, variant in enumerate(variants):
            query_weight = 1.0 if query_rank == 0 else 0.72
            docs = self.base_retriever.invoke(variant)
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                if key not in doc_scores:
                    doc_scores[key] = {"doc": doc, "score": 0.0}
                doc_scores[key]["score"] += query_weight / (rank + 60)
        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[: self.k]]

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


class _HybridRerankRetriever:
    """先做混合召回，再做 cross-encoder 精排。"""

    def __init__(self, base_retriever, reranker, k=6):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.k = k

    def invoke(self, query: str):
        candidates = self.base_retriever.invoke(query)
        reranked = self.reranker.compress_documents(candidates, query)
        return list(reranked)[: self.k]

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


def _build_query_variants(query: str, category: Optional[str] = None) -> List[str]:
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized:
        return []

    variants = [normalized]
    seen = {normalized}

    hints: List[str] = []
    for pattern, hint in _DOMAIN_QUERY_HINTS:
        if pattern.search(normalized):
            hints.append(hint)
    hints.extend(_CATEGORY_QUERY_HINTS.get(category or "", []))

    for hint in hints:
        variant = f"{normalized} {hint}".strip()
        if variant not in seen:
            variants.append(variant)
            seen.add(variant)
        if len(variants) >= 3:
            break

    return variants


def _get_retrieval_profile(category: Optional[str] = None, hint_level: int = 0) -> RetrievalProfile:
    bucket = "high" if hint_level >= 2 else "low"
    return _RETRIEVAL_PROFILES.get(category or "", {}).get(bucket, _DEFAULT_PROFILE)


def _build_hybrid_retriever(profile: RetrievalProfile):
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as exc:
        raise ImportError("Hybrid 检索需要 rank-bm25，请先安装依赖。") from exc

    if not _loaded_chunks:
        raise RuntimeError("Hybrid 检索缺少运行时 chunks 数据。")

    try:
        import jieba
        preprocess = lambda text: list(jieba.cut(text))
    except ImportError:
        preprocess = lambda text: text.split()

    bm25 = BM25Retriever.from_documents(
        _loaded_chunks,
        preprocess_func=preprocess,
        k=profile.bm25_k,
    )
    dense_kwargs = {"k": profile.dense_k}
    if profile.dense_search_type == "mmr":
        dense_kwargs.update({
            "fetch_k": max(profile.fetch_k, profile.dense_k),
            "lambda_mult": profile.dense_lambda_mult,
        })
    dense = _loaded_vectorstore.as_retriever(
        search_type=profile.dense_search_type,
        search_kwargs=dense_kwargs,
    )
    return _RRFRetriever(
        retrievers=[bm25, dense],
        weights=[profile.bm25_weight, profile.dense_weight],
        k=profile.fusion_k,
    )


# -------------------------
# 延迟初始化入口
# -------------------------

def _ensure_rag_initialized():
    """首次调用时执行完整的 RAG 初始化流程（线程安全）。"""
    global _loaded_vectorstore, _loaded_chunks, _cross_encoder, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        print("[RAG] 开始初始化...")

        # 1. 嵌入模型
        print(f"[RAG] 加载 Embedding 模型：{EMBEDDING_MODEL_NAME} ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2. 优先复用 index 对应的 chunks，确保 BM25 与 Dense 使用同一套文档块
        chunks = None
        if not REBUILD_INDEX and Path(INDEX_DIR).exists():
            chunks = _load_chunks_from_index(Path(INDEX_DIR))
            if chunks:
                print(f"[RAG] 从 {INDEX_DIR / 'chunks.pkl'} 加载运行时 chunks，共 {len(chunks)} 个")

        # 3. 如果需要，再从 docx 重新切分并构建索引
        if chunks is None:
            docx_files = sorted(DATA_DIR.glob("*.docx"))
            if not docx_files:
                raise RuntimeError(f"未在 {DATA_DIR} 下找到 .docx 文件")

            print(f"[RAG] 并行加载 {len(docx_files)} 个 docx 文件...")
            all_chunks: List = []
            with ThreadPoolExecutor(max_workers=min(4, len(docx_files))) as executor:
                futures = {executor.submit(_load_single_docx, p): p for p in docx_files}
                for future in as_completed(futures):
                    all_chunks.extend(future.result())

            chunks = _prepare_runtime_chunks(
                all_chunks,
                add_context_prefix=(INDEX_DIR.name == "enriched"),
            )
            print(f"[RAG] 文档加载完成，共 {len(chunks)} 个 chunk")

        _loaded_chunks = chunks

        # 4. 构建或加载 FAISS
        if REBUILD_INDEX or not Path(INDEX_DIR).exists():
            print("[RAG] 正在构建 FAISS 索引...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(INDEX_DIR)
            with open(Path(INDEX_DIR) / "chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)
            print(f"[RAG] 向量索引已保存到 {INDEX_DIR}/")

        _loaded_vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[RAG] FAISS 索引加载完成")

        # 5. Reranker
        if not DISABLE_RERANKER:
            print(f"[RAG] 加载 Reranker 模型：{RERANKER_MODEL_NAME} ...")
            _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
            print("[RAG] Reranker 加载完成")
        else:
            _cross_encoder = None
            print("[RAG] Reranker 已禁用，使用 Hybrid 检索")

        _initialized = True
        print("[RAG] 初始化完成")


# -------------------------
# 检索
# -------------------------

def _get_adaptive_retriever(category: Optional[str] = None, hint_level: int = 0):
    _ensure_rag_initialized()

    profile = _get_retrieval_profile(category, hint_level)
    cache_key = (category, hint_level >= 2, profile, ENABLE_QUERY_FUSION, DISABLE_RERANKER)
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    hybrid = _build_hybrid_retriever(profile)
    candidate_retriever = _QueryFusionRetriever(
        base_retriever=hybrid,
        category=category,
        k=profile.fusion_k,
        enabled=ENABLE_QUERY_FUSION,
    )

    if DISABLE_RERANKER or _cross_encoder is None:
        result = _QueryFusionRetriever(
            base_retriever=hybrid,
            category=category,
            k=profile.top_n,
            enabled=ENABLE_QUERY_FUSION,
        )
    else:
        reranker = CrossEncoderReranker(model=_cross_encoder, top_n=profile.top_n)
        result = _HybridRerankRetriever(
            base_retriever=candidate_retriever,
            reranker=reranker,
            k=profile.top_n,
        )

    _retriever_cache[cache_key] = result
    return result


def build_citations(docs, max_sources: int = 6, snippet_len: int = 180):
    citations = []
    for i, d in enumerate(docs[:max_sources], start=1):
        src = (d.metadata or {}).get("source", "unknown")
        snippet = (d.page_content or "").replace("\n", " ").strip()[:snippet_len]
        citations.append({
            "id": i,
            "source": src,
            "snippet": snippet,
        })
    return citations


def build_numbered_context(docs, citations):
    parts = []
    for c, d in zip(citations, docs):
        header = f"[{c['id']}] {c['source']}"
        parts.append(f"{header}\n{(d.page_content or '').strip()}")
    return "\n\n".join(parts)


def retrieve_course_docs(
    query: str,
    category: Optional[str] = None,
    hint_level: int = 0,
    max_sources: int = 6,
) -> Dict[str, Any]:
    """纯检索接口，返回原始证据上下文与引用元数据，不做 LLM 整理。"""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return {"ok": False, "error": "query is required"}

    try:
        normalized_hint_level = max(0, int(hint_level))
    except (TypeError, ValueError):
        normalized_hint_level = 0

    try:
        normalized_max_sources = max(1, int(max_sources))
    except (TypeError, ValueError):
        normalized_max_sources = 6

    try:
        _ensure_rag_initialized()

        active_retriever = _get_adaptive_retriever(category, normalized_hint_level)
        try:
            docs = active_retriever.invoke(normalized_query)
        except Exception:
            docs = active_retriever.get_relevant_documents(normalized_query)

        citations = build_citations(docs, max_sources=normalized_max_sources)
        context_docs = docs[: len(citations)]
        context = build_numbered_context(context_docs, citations) if citations else ""

        return {
            "ok": True,
            "query": normalized_query,
            "category": category,
            "hint_level": normalized_hint_level,
            "source_count": len(citations),
            "citations": citations,
            "context": context,
        }
    except Exception as exc:
        return {
            "ok": False,
            "query": normalized_query,
            "category": category,
            "hint_level": normalized_hint_level,
            "error": f"retrieve_course_docs failed: {exc}",
        }


def RAGAgent(message, category: Optional[str] = None, hint_level: int = 0):
    """
    兼容旧名称：当前返回纯检索结果，不再做 LLM 整理。

    category: 问题分类（THEORY_CONCEPT / LAB_TROUBLESHOOTING / CONFIG_REVIEW / CALCULATION）
    hint_level: 0-3，影响检索广度和返回数量
    """
    return retrieve_course_docs(
        query=message,
        category=category,
        hint_level=hint_level,
    )


if __name__ == "__main__":
    context = "如何制作网线？"
    answer = RAGAgent(context)
    print(answer)
