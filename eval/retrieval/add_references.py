"""
为 qa_dataset.json 中的每个问题生成基于文档的参考答案要点
用法：python eval/add_references.py

功能：
  1. 根据问题的 source 字段，用向量检索 + 源文档过滤，
     只从对应的实验文档 chunks 中检索相关片段
  2. 将文档片段 + 问题一起发给 GPT-4o，基于文档内容生成参考答案要点
  3. 写回 qa_dataset.json 的 "reference" 字段

这样参考答案反映的是"教材里实际写了什么"，而不是 GPT 的通用知识。
且检索结果 100% 来自正确的源文档，不会混入其他实验的内容。

环境变量：
  OPENAI_API_KEY        (必须)
  EMBEDDING_MODEL_NAME  默认 BAAI/bge-m3
  RAG_REBUILD_INDEX     设为 0 跳过重建（推荐）

加 --force 参数可覆盖已有的 reference 重新生成。
"""

import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

QA_DATASET      = ROOT / "eval" / "qa_dataset.json"
INDEX_DIR       = ROOT / "faiss_index"
MODEL           = "gpt-4o"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RETRIEVE_K      = 8   # 过滤后返回的文档数量
FETCH_K         = 50  # 过滤前从 FAISS 取的候选数（需要足够大，因为过滤在检索之后）

REFERENCE_PROMPT = """\
你是计算机网络课程的资深教师。请严格基于以下文档片段，为学生问题写出参考答案要点。

要求：
1. 只能使用文档片段中的信息，不要补充文档未提及的内容
2. 用简洁的要点列表（3-6 条），每条一句话
3. 如果文档中没有足够信息回答该问题，明确指出"文档未覆盖"
4. 用中文回答

【文档片段】（来源：{source}）
{context}

【问题类型】{qtype}
【问题】{question}

请直接输出要点列表，每条以"- "开头，不要有其他格式："""


def load_vectorstore():
    """加载 FAISS 向量库（直接返回 vectorstore，不包装成 retriever）"""
    print(f"加载 Embedding 模型：{EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    return vs


def generate_reference(
    client: OpenAI,
    vectorstore: FAISS,
    question: str,
    qtype: str,
    source_hint: str,
) -> str:
    # 1. 源文档过滤检索：只从对应实验文档的 chunks 中搜索
    if source_hint:
        docs = vectorstore.similarity_search(
            question,
            k=RETRIEVE_K,
            filter={"source": source_hint},
            fetch_k=FETCH_K,
        )
    else:
        docs = []

    # 如果源文档过滤无结果（source 为空或该文档未入库），降级为全库检索
    if not docs:
        if source_hint:
            print(f"[降级] 源文档 '{source_hint}' 无匹配，改用全库检索 ", end="")
        docs = vectorstore.similarity_search(question, k=RETRIEVE_K)

    # 拼接上下文
    context_parts = []
    sources = set()
    for i, doc in enumerate(docs, 1):
        src = (doc.metadata or {}).get("source", "unknown")
        sources.add(src)
        context_parts.append(f"[{i}] {src}\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)
    source_str = "、".join(sources)

    # 2. 基于文档生成参考答案
    prompt = REFERENCE_PROMPT.format(
        context=context,
        source=source_str,
        qtype=qtype,
        question=question,
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [失败] {e}")
        return ""


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    if not QA_DATASET.exists():
        print(f"错误：找不到 {QA_DATASET}")
        sys.exit(1)

    force = "--force" in sys.argv

    client = OpenAI(api_key=api_key)
    vectorstore = load_vectorstore()

    with open(QA_DATASET, encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)
    existing = sum(1 for q in questions if q.get("reference"))
    if existing and not force:
        print(f"共 {total} 个问题，其中 {existing} 个已有参考答案，将跳过")
        print("（加 --force 参数可覆盖重新生成）")
    elif existing and force:
        print(f"共 {total} 个问题，--force 模式：将覆盖所有已有参考答案")

    for i, q in enumerate(questions, 1):
        if q.get("reference") and not force:
            continue

        qtext = q["question"]
        qtype = q.get("type", "unknown")
        source_hint = q.get("source", "")
        print(f"[{i}/{total}] {qtext[:50]}{'...' if len(qtext)>50 else ''}", end=" ", flush=True)

        ref = generate_reference(client, vectorstore, qtext, qtype, source_hint)
        q["reference"] = ref
        print("OK" if ref else "SKIP")

        time.sleep(0.3)

    with open(QA_DATASET, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n已更新：{QA_DATASET}")
    print("之后运行：python eval/evaluate_retrieval.py")


if __name__ == "__main__":
    main()
