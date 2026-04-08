import os
import pickle
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from agentic_rag.llm_config import build_chat_llm

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

# -------------------------
# 延迟初始化状态（首次查询时才真正加载）
# -------------------------
_llm = None
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
    "LAB_TROUBLESHOOTING": {"low": (25, 3), "high": (30, 4)},
    "CONFIG_REVIEW":       {"low": (25, 3), "high": (30, 5)},
    "CALCULATION":         {"low": (20, 3), "high": (20, 4)},
}
_DEFAULT_RETRIEVAL = (30, 4)

# 创建带有 system 消息的模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是网络实验排错助教。你必须仅依据"已知信息"回答，不得编造。
回答中的每个关键结论句末尾必须用引用编号标注来源，格式为 [n]，n 来自已知信息块的编号。
如果已知信息不足以支持结论，回答"无法从已知信息确定"。

已知信息:
{context} """),
    ("user", "{question}")
])


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
    global _llm, _loaded_vectorstore, _loaded_chunks, _cross_encoder, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        print("[RAG] 开始初始化...")

        # 1. LLM
        _llm = build_chat_llm(temperature=0)

        # 2. 嵌入模型
        print(f"[RAG] 加载 Embedding 模型：{EMBEDDING_MODEL_NAME} ...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        # 3. 优先复用 index 对应的 chunks，确保 BM25 与 Dense 使用同一套文档块
        chunks = None
        if not REBUILD_INDEX and Path(INDEX_DIR).exists():
            chunks = _load_chunks_from_index(Path(INDEX_DIR))
            if chunks:
                print(f"[RAG] 从 {INDEX_DIR / 'chunks.pkl'} 加载运行时 chunks，共 {len(chunks)} 个")

        # 4. 如果需要，再从 docx 重新切分并构建索引
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

        # 5. 构建或加载 FAISS
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

        # 6. Reranker
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
# 检索与生成
# -------------------------

def _get_adaptive_retriever(category: Optional[str] = None, hint_level: int = 0):
    _ensure_rag_initialized()

    bucket = "high" if hint_level >= 2 else "low"
    fetch_k, top_n = _RETRIEVAL_PARAMS.get(category or "", {}).get(bucket, _DEFAULT_RETRIEVAL)

    cache_key = (fetch_k, top_n, DISABLE_RERANKER)
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

    if DISABLE_RERANKER or _cross_encoder is None:
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


def _ensure_inline_citations(answer: str, default_id: int = 1) -> str:
    if re.search(r"\[\d+\]", answer or ""):
        return answer
    if not answer:
        return answer
    answer = re.sub(r"([。！？!?])", r"\1" + f" [{default_id}]", answer)
    if f"[{default_id}]" not in answer:
        answer = answer.rstrip() + f" [{default_id}]"
    return answer


def RAGAgent(message, category: Optional[str] = None, hint_level: int = 0):
    """
    category: 问题分类（THEORY_CONCEPT / LAB_TROUBLESHOOTING / CONFIG_REVIEW / CALCULATION）
    hint_level: 0-3，影响检索广度和返回数量
    """
    _ensure_rag_initialized()

    active_retriever = _get_adaptive_retriever(category, hint_level)
    try:
        docs = active_retriever.invoke(message)
    except Exception:
        docs = active_retriever.get_relevant_documents(message)

    citations = build_citations(docs)
    context = build_numbered_context(docs, citations)

    messages = prompt_template.format_messages(context=context, question=message)
    resp = _llm.invoke(messages)
    answer = getattr(resp, "content", str(resp))
    if citations:
        answer = _ensure_inline_citations(answer, default_id=citations[0]["id"])

    if citations:
        footer = "\n".join([f"[{c['id']}] {c['source']}" for c in citations])
        answer = answer.rstrip() + "\n\n引用：\n" + footer

    return {"answer": answer, "citations": citations}


if __name__ == "__main__":
    context = "如何制作网线？"
    answer = RAGAgent(context)
    print(answer)
