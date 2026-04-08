"""
检索方案量化评测脚本
用法：python eval/evaluate_retrieval.py

功能：
  对比三种检索方案，用 GPT-4o 作为裁判打分，输出 CSV

对比方案（统一返回 6 条文档，公平对比）：
  A: 纯相似度检索（Similarity, k=6）
  B: MMR 检索（k=6, fetch_k=30）
  C: MMR + BGE Reranker 精排（粗检索 k=20, fetch_k=50 → 精排 top_n=6）

输入：eval/qa_dataset.json（由 generate_qa_dataset.py 生成并经你筛选）
输出：eval/results_TIMESTAMP.csv + 控制台汇总

环境变量：
  OPENAI_API_KEY   (必须，用于 GPT-4o 打分)
  DEEPSEEK_API_KEY (必须，用于生成回答)
  EMBEDDING_MODEL_NAME  默认 BAAI/bge-m3
  RERANKER_MODEL_NAME   默认 BAAI/bge-reranker-v2-m3
  RAG_REBUILD_INDEX     设为 0 跳过重建（推荐）
"""

import os
import sys
import json
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

# ── 依赖导入 ───────────────────────────────────────────
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import SystemMessage, HumanMessage

# DeepSeek client（复用项目已有实现）
from agentic_rag.llm_config import build_chat_llm

# ── 配置 ──────────────────────────────────────────────
QA_DATASET      = ROOT / "eval" / "qa_dataset.json"
INDEX_DIR       = ROOT / "faiss_index"
OUTPUT_DIR      = ROOT / "eval"
JUDGE_MODEL     = "gpt-4o"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
SNIPPET_LEN     = 300   # 送给评委的检索片段长度
ANSWER_SNIPPET  = 1200  # 送给评委的回答截断长度
# ─────────────────────────────────────────────────────

# ── 打分 Prompt ────────────────────────────────────────
JUDGE_SYSTEM = """\
你是计算机网络课程的专家评审，负责客观评估 AI 助教回答的质量。
你的打分必须严格、一致，不偏袒任何方案。
请严格对照每个维度的 5 级锚点描述打分，不要凭主观印象。"""

# 带参考答案的模板
JUDGE_USER_TEMPLATE_WITH_REF = """\
请对以下 AI 助教的回答从 4 个维度打分（每项 1-5 分整数）。

【学生问题】
{question}

【参考答案要点】（用于判断完整性的客观基准）
{reference}

【AI 回答】
{answer}

【AI 检索到的参考资料】（AI 依据此内容生成回答）
{context}

─── 评分标准（请严格对照打分）──────────────────────
1. 相关性（relevance）
   5=完全紧扣问题，无跑题内容
   4=主要内容相关，有少量无关内容
   3=部分相关，但有明显偏题
   2=仅少量内容与问题相关
   1=回答与问题几乎无关

2. 忠实性（faithfulness）
   5=所有结论均能在参考资料中找到依据，无捏造
   4=绝大部分有据可查，有 1 处无法确认来源
   3=有 2-3 处无据内容或轻微矛盾
   2=多处内容缺乏支撑或与资料矛盾
   1=大量编造，与资料严重不符

3. 完整性（completeness）
   5=参考答案要点全部被覆盖
   4=覆盖 80% 以上要点，缺少次要细节
   3=覆盖约 60% 要点，有明显遗漏
   2=仅覆盖少数要点，核心内容缺失
   1=几乎未覆盖任何要点

4. 技术准确性（technical_accuracy）
   5=IP地址/子网掩码/命令语法/协议行为等技术事实完全正确
   4=技术内容基本正确，有 1 处小瑕疵
   3=有 1-2 处明显技术错误
   2=多处技术错误，影响理解
   1=技术内容严重错误

综合（overall）= 加权平均，权重：
  相关性 0.2 + 忠实性 0.3 + 完整性 0.3 + 技术准确 0.2
  四舍五入到整数。
────────────────────────────────────────────────────

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "overall": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""

# 无参考答案的降级模板
JUDGE_USER_TEMPLATE_NO_REF = """\
请对以下 AI 助教的回答从 4 个维度打分（每项 1-5 分整数）。

【学生问题】
{question}

【AI 回答】
{answer}

【AI 检索到的参考资料】（AI 依据此内容生成回答）
{context}

─── 评分标准（请严格对照打分）──────────────────────
1. 相关性（relevance）
   5=完全紧扣问题，无跑题内容
   4=主要内容相关，有少量无关内容
   3=部分相关，但有明显偏题
   2=仅少量内容与问题相关
   1=回答与问题几乎无关

2. 忠实性（faithfulness）
   5=所有结论均能在参考资料中找到依据，无捏造
   4=绝大部分有据可查，有 1 处无法确认来源
   3=有 2-3 处无据内容或轻微矛盾
   2=多处内容缺乏支撑或与资料矛盾
   1=大量编造，与资料严重不符

3. 完整性（completeness）
   5=作为计算机网络课程助教，回答覆盖了该问题所有关键知识点
   4=覆盖大部分关键知识点，缺少次要细节
   3=覆盖部分知识点，有明显遗漏
   2=仅覆盖少数知识点，核心内容缺失
   1=几乎未覆盖任何关键知识点

4. 技术准确性（technical_accuracy）
   5=IP地址/子网掩码/命令语法/协议行为等技术事实完全正确
   4=技术内容基本正确，有 1 处小瑕疵
   3=有 1-2 处明显技术错误
   2=多处技术错误，影响理解
   1=技术内容严重错误

综合（overall）= 加权平均，权重：
  相关性 0.2 + 忠实性 0.3 + 完整性 0.3 + 技术准确 0.2
  四舍五入到整数。
────────────────────────────────────────────────────

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "overall": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""
# ─────────────────────────────────────────────────────


# ══ 检索方案定义 ══════════════════════════════════════

FINAL_K = 6  # 所有方案统一返回的文档数，确保公平对比


def build_retrievers(embeddings: HuggingFaceEmbeddings):
    """构建三种检索器，返回字典。

    所有方案统一返回 FINAL_K 条文档以保证公平对比。
    方案 C 的 base retriever 过检索（k=20, fetch_k=50），
    给 Reranker 充足的候选池进行精选。
    """
    vs = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

    # 方案 A：纯相似度
    retriever_a = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": FINAL_K},
    )

    # 方案 B：MMR
    retriever_b = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": FINAL_K, "fetch_k": 30, "lambda_mult": 0.5},
    )

    # 方案 C：MMR + Reranker（粗检索 20 条 → 精排取 FINAL_K 条）
    print(f"  正在加载 Reranker 模型：{RERANKER_MODEL} ...")
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=FINAL_K)
    retriever_c = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5},
        ),
    )

    return {
        "A_Similarity": retriever_a,
        "B_MMR":        retriever_b,
        "C_MMR+Rerank": retriever_c,
    }


# ══ 回答生成 ══════════════════════════════════════════

ANSWER_SYSTEM = """\
你是网络实验排错助教。请仅依据已知信息回答问题，不得编造。
如果已知信息不足，回答"无法从已知信息确定"。"""

def generate_answer(llm, question: str, docs: list) -> str:
    context_parts = []
    for i, doc in enumerate(docs, 1):
        src = (doc.metadata or {}).get("source", "unknown")
        context_parts.append(f"[{i}] {src}\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM + f"\n\n已知信息:\n{context}"),
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))


# ══ GPT-4o 打分 ═══════════════════════════════════════

def judge_answer(
    openai_client: OpenAI,
    question: str,
    answer: str,
    docs: list,
    reference: str = "",
) -> Dict[str, Any]:
    context_snippet = "\n---\n".join(
        d.page_content.strip()[:SNIPPET_LEN] for d in docs
    )
    answer_snippet = answer[:ANSWER_SNIPPET]

    if reference:
        user_msg = JUDGE_USER_TEMPLATE_WITH_REF.format(
            question=question,
            reference=reference,
            answer=answer_snippet,
            context=context_snippet,
        )
    else:
        user_msg = JUDGE_USER_TEMPLATE_NO_REF.format(
            question=question,
            answer=answer_snippet,
            context=context_snippet,
        )
    try:
        resp = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown 包裹
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        return scores
    except Exception as e:
        print(f"    [打分失败] {e}")
        return {
            "relevance": None, "faithfulness": None,
            "completeness": None, "technical_accuracy": None,
            "overall": None, "comment": f"打分失败: {e}",
        }


# ══ 主流程 ════════════════════════════════════════════

def main():
    # 检查必要环境变量
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    # 加载问题数据集
    if not QA_DATASET.exists():
        print(f"错误：找不到问题数据集 {QA_DATASET}")
        print("请先运行 python eval/generate_qa_dataset.py 并筛选问题")
        sys.exit(1)

    with open(QA_DATASET, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"加载 {len(questions)} 个测试问题")

    # 初始化模型
    print("\n初始化模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    retrievers = build_retrievers(embeddings)
    llm = build_chat_llm(temperature=0)
    openai_client = OpenAI(api_key=openai_key)
    print("模型初始化完成\n")

    # 开始评测
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = OUTPUT_DIR / f"results_{timestamp}.csv"

    fieldnames = [
        "question_id", "question", "question_type", "scheme",
        "retrieved_docs_count", "answer_length",
        "relevance", "faithfulness", "completeness", "technical_accuracy",
        "overall", "comment",
    ]

    rows = []
    scheme_names = list(retrievers.keys())
    total = len(questions) * len(scheme_names)
    done = 0

    for q_item in questions:
        qid       = q_item.get("id", "?")
        question  = q_item.get("question", "")
        qtype     = q_item.get("type", "unknown")
        reference = q_item.get("reference", "")

        print(f"\n── 问题 {qid}：{question[:50]}{'...' if len(question)>50 else ''}")

        for scheme_name, retriever in retrievers.items():
            done += 1
            print(f"  [{done}/{total}] 方案 {scheme_name} ...", end=" ", flush=True)

            # 1. 检索
            try:
                docs = retriever.invoke(question)
            except Exception:
                docs = retriever.get_relevant_documents(question)

            # 2. 生成回答
            answer = generate_answer(llm, question, docs)

            # 3. GPT-4o 打分
            scores = judge_answer(openai_client, question, answer, docs, reference)

            print(f"overall={scores.get('overall', '?')}")

            rows.append({
                "question_id":         qid,
                "question":            question,
                "question_type":       qtype,
                "scheme":              scheme_name,
                "retrieved_docs_count": len(docs),
                "answer_length":       len(answer),
                "relevance":           scores.get("relevance"),
                "faithfulness":        scores.get("faithfulness"),
                "completeness":        scores.get("completeness"),
                "technical_accuracy":  scores.get("technical_accuracy"),
                "overall":             scores.get("overall"),
                "comment":             scores.get("comment", ""),
            })

            time.sleep(0.5)  # 避免 API 频率限制

    # 保存 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n结果已保存到：{output_csv}")

    # 打印汇总表
    print("\n" + "=" * 65)
    print("方案对比汇总（各维度平均分）")
    print("=" * 65)
    print(f"{'方案':<18} {'相关性':>6} {'忠实性':>6} {'完整性':>6} {'技术准确':>8} {'综合':>6}")
    print("-" * 65)

    for scheme_name in scheme_names:
        scheme_rows = [r for r in rows if r["scheme"] == scheme_name]
        def avg(key):
            vals = [r[key] for r in scheme_rows if r[key] is not None]
            return f"{sum(vals)/len(vals):.2f}" if vals else "N/A"
        print(
            f"{scheme_name:<18} {avg('relevance'):>6} {avg('faithfulness'):>6} "
            f"{avg('completeness'):>6} {avg('technical_accuracy'):>8} {avg('overall'):>6}"
        )

    print("=" * 65)
    print(f"\n原始数据（含每题评语）见：{output_csv}")


if __name__ == "__main__":
    main()
