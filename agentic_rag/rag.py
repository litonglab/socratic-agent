import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import BaseModel
from tqdm import tqdm

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.llm_config import build_chat_llm

# 加载环境变量，读取本地 .env 文件，里面定义了 OPENAI_API_KEY
#_ = load_dotenv(find_dotenv())

# llm
llm = build_chat_llm(temperature=0)

# 1. 加载docx文档（批量，使用相对路径）
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"
REBUILD_INDEX = os.getenv("RAG_REBUILD_INDEX", "1").lower() in {"1", "true", "yes"}
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")


def _extract_section(text: str) -> Optional[str]:
    """
    简单的章节抽取：优先找“第X章/节/实验X”等标题式行。
    """
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

# 使用

# 2. 创建文本分割器（必须分块 + 记录 start_index）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " ", ""],
    length_function=len,
    add_start_index=True,
)
print("finish loader")

# 2. 批量加载 docx
docx_files = sorted(DATA_DIR.glob("*.docx"))
if not docx_files:
    raise RuntimeError(f"未在 {DATA_DIR} 下找到 .docx 文件")

all_chunks: List = []
for docx_path in docx_files:
    loader = Docx2txtLoader(str(docx_path))
    try:
        docs = loader.load()
    except zipfile.BadZipFile:
        print(f"[Skip] 非法 docx 文件（不是 zip）：{docx_path}")
        continue
    except Exception as e:
        print(f"[Skip] 读取 docx 失败：{docx_path}，原因：{e}")
        continue
    chunks = text_splitter.split_documents(docs)
    chunks = _enrich_metadata(chunks, str(docx_path))
    all_chunks.extend(chunks)

# 3. 分割文本 + metadata（已批量处理）
chunks = normalize_source_to_filename(all_chunks)

# 4. 嵌入模型（本地优先：BAAI/bge-m3）
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)

# 5. 构建或加载 FAISS
if REBUILD_INDEX or not Path(INDEX_DIR).exists():
    print("正在构建FAISS索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    print(f"向量索引已保存到 {INDEX_DIR}/ 目录")

loaded_vectorstore = FAISS.load_local(
    INDEX_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)

# 6. 使用 MMR 检索（更分散）
retriever = loaded_vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)
print("finish embeding")
# 创建带有 system 消息的模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是网络实验排错助教。你必须仅依据“已知信息”回答，不得编造。
回答中的每个关键结论句末尾必须用引用编号标注来源，格式为 [n]，n 来自已知信息块的编号。
如果已知信息不足以支持结论，回答“无法从已知信息确定”。

已知信息:
{context} """),
    ("user", "{question}")
])

# 自定义的提示词参数
chain_type_kwargs = {
    "prompt": prompt_template,
}

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
    """
    若模型未输出任何 [n]，则按句末补一个默认引用编号。
    """
    if re.search(r"\[\d+\]", answer or ""):
        return answer
    if not answer:
        return answer
    # 对中文句号/问号/叹号进行补标
    answer = re.sub(r"([。！？!?])", r"\1" + f" [{default_id}]", answer)
    # 若没有句末标点，给最后追加
    if f"[{default_id}]" not in answer:
        answer = answer.rstrip() + f" [{default_id}]"
    return answer


def RAGAgent(message):
    try:
        docs = retriever.invoke(message)
    except Exception:
        docs = retriever.get_relevant_documents(message)

    citations = build_citations(docs)
    context = build_numbered_context(docs, citations)

    messages = prompt_template.format_messages(context=context, question=message)
    resp = llm.invoke(messages)
    answer = getattr(resp, "content", str(resp))
    if citations:
        answer = _ensure_inline_citations(answer, default_id=citations[0]["id"])

    # 兜底：如果模型未内联引用，也在回答末尾追加引用列表
    if citations:
        footer = "\n".join([f"[{c['id']}] {c['source']}" for c in citations])
        answer = answer.rstrip() + "\n\n引用：\n" + footer

    return {"answer": answer, "citations": citations}


if __name__ == "__main__":
    context="如何制作网线？"
    answer= RAGAgent(context)
    print(answer)
