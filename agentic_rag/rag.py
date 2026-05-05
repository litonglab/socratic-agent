import os
import pickle
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _resolve_index_dir() -> Path:
    index_env = os.getenv("RAG_INDEX_DIR")
    if index_env:
        candidate = Path(index_env)
        return candidate if candidate.is_absolute() else BASE_DIR / candidate
    preferred_index = BASE_DIR / "faiss_index" / "enriched"
    return preferred_index if preferred_index.exists() else BASE_DIR / "faiss_index"


def _resolve_runtime_config() -> Dict[str, Any]:
    return {
        "index_dir": _resolve_index_dir(),
        "rebuild_index": os.getenv("RAG_REBUILD_INDEX", "0").lower() in {"1", "true", "yes"},
        "embedding_model_name": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        "reranker_model_name": os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        "disable_reranker": os.getenv("DISABLE_RERANKER", "0").lower() in {"1", "true", "yes"},
    }

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

# 7. 教学意图感知检索策略参数表
_RETRIEVAL_PARAMS: dict = {
    "THEORY_CONCEPT":      {"low": (40, 4), "high": (40, 6)},
    "LAB_TROUBLESHOOTING": {"low": (30, 4), "high": (30, 4)},
    "CONFIG_REVIEW":       {"low": (25, 3), "high": (30, 5)},
    "CALCULATION":         {"low": (20, 3), "high": (20, 4)},
}
_DEFAULT_RETRIEVAL = (30, 4)


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

    def __init__(self, retrievers, weights, k=6):
        self.retrievers = retrievers
        self.weights = weights
        self.k = k

    def invoke(self, query: str):
        doc_scores = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                if key not in doc_scores:
                    doc_scores[key] = {"doc": doc, "score": 0.0}
                doc_scores[key]["score"] += weight / (rank + 60)
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


def _build_hybrid_retriever(bm25_k: int, dense_k: int, fusion_k: int):
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
        k=bm25_k,
    )
    dense = _loaded_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": dense_k},
    )
    return _RRFRetriever(
        retrievers=[bm25, dense],
        weights=[0.45, 0.55],
        k=fusion_k,
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

        config = _resolve_runtime_config()
        index_dir: Path = config["index_dir"]
        rebuild_index: bool = config["rebuild_index"]
        embedding_model_name: str = config["embedding_model_name"]
        reranker_model_name: str = config["reranker_model_name"]
        disable_reranker: bool = config["disable_reranker"]

        print("[RAG] 开始初始化...")

        # 1. 优先复用 index 对应的 chunks，确保 BM25 与 Dense 使用同一套文档块
        chunks = None
        if not rebuild_index and index_dir.exists():
            chunks = _load_chunks_from_index(index_dir)
            if chunks:
                print(f"[RAG] 从 {index_dir / 'chunks.pkl'} 加载运行时 chunks，共 {len(chunks)} 个")

        # 2. 如果需要，再从 docx 重新切分
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
                add_context_prefix=(index_dir.name == "enriched"),
            )
            print(f"[RAG] 文档加载完成，共 {len(chunks)} 个 chunk")

        _loaded_chunks = chunks

        # 3 & 4 并行：Embedding+FAISS 与 Reranker 互相独立，同时加载
        _emb_error: List[BaseException] = []
        _rnk_error: List[BaseException] = []
        _vectorstore_box: List = [None]
        _reranker_box: List = [None]

        def _load_embedding_and_faiss():
            try:
                print(f"[RAG] 加载 Embedding 模型：{embedding_model_name} ...")
                emb = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    encode_kwargs={"normalize_embeddings": True},
                )
                if rebuild_index or not index_dir.exists():
                    print("[RAG] 正在构建 FAISS 索引...")
                    vs = FAISS.from_documents(chunks, emb)
                    index_dir.mkdir(parents=True, exist_ok=True)
                    vs.save_local(index_dir)
                    with open(index_dir / "chunks.pkl", "wb") as f:
                        pickle.dump(chunks, f)
                    print(f"[RAG] 向量索引已保存到 {index_dir}/")
                _vectorstore_box[0] = FAISS.load_local(
                    index_dir, emb, allow_dangerous_deserialization=True
                )
                print("[RAG] FAISS 索引加载完成")
            except BaseException as exc:  # noqa: BLE001
                _emb_error.append(exc)

        def _load_reranker():
            try:
                print(f"[RAG] 加载 Reranker 模型：{reranker_model_name} ...")
                _reranker_box[0] = HuggingFaceCrossEncoder(model_name=reranker_model_name)
                print("[RAG] Reranker 加载完成")
            except BaseException as exc:  # noqa: BLE001
                _rnk_error.append(exc)

        t_emb = threading.Thread(target=_load_embedding_and_faiss, daemon=True, name="rag-emb")
        t_emb.start()

        if not disable_reranker:
            t_rnk = threading.Thread(target=_load_reranker, daemon=True, name="rag-reranker")
            t_rnk.start()
            t_rnk.join()
        else:
            print("[RAG] Reranker 已禁用，使用 Hybrid 检索")

        t_emb.join()

        if _emb_error:
            raise _emb_error[0]
        if _rnk_error:
            raise _rnk_error[0]

        _loaded_vectorstore = _vectorstore_box[0]
        _cross_encoder = _reranker_box[0]

        _initialized = True
        print("[RAG] 初始化完成")


# -------------------------
# 检索
# -------------------------

def _get_adaptive_retriever(category: Optional[str] = None, hint_level: int = 0):
    _ensure_rag_initialized()
    config = _resolve_runtime_config()
    disable_reranker: bool = config["disable_reranker"]

    bucket = "high" if hint_level >= 2 else "low"
    fetch_k, top_n = _RETRIEVAL_PARAMS.get(category or "", {}).get(bucket, _DEFAULT_RETRIEVAL)

    cache_key = (fetch_k, top_n, disable_reranker)
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    dense_k = max(top_n * 3, min(fetch_k, 18))
    bm25_k = max(top_n * 2, min(fetch_k // 2, 12))
    fusion_k = max(top_n * 3, min(fetch_k, 24))
    hybrid = _build_hybrid_retriever(
        bm25_k=bm25_k,
        dense_k=dense_k,
        fusion_k=fusion_k,
    )

    if disable_reranker or _cross_encoder is None:
        result = _RRFRetriever(
            retrievers=hybrid.retrievers,
            weights=hybrid.weights,
            k=top_n,
        )
    else:
        reranker = CrossEncoderReranker(model=_cross_encoder, top_n=top_n)
        result = _HybridRerankRetriever(
            base_retriever=hybrid,
            reranker=reranker,
            k=top_n,
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
