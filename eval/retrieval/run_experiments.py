"""
运行检索实验：按 experiment_config.py 中定义的实验组合逐个评测

用法：
  python eval/retrieval/run_experiments.py                              # 运行所有实验
  python eval/retrieval/run_experiments.py --only baseline_similarity   # 只跑指定实验
  python eval/retrieval/run_experiments.py --only baseline_similarity,baseline_mmr
  python eval/retrieval/run_experiments.py --dataset eval/qa_dataset_small.json
  python eval/retrieval/run_experiments.py --workers 4                  # 每个实验内并行处理 4 道题

--workers 说明：
  默认值 1（顺序执行）。设为 3-5 可显著提速，瓶颈是 DeepSeek/GPT-4o 的 API 响应时延。
  线程数过高可能触发 API 限速（429），建议不超过 6。
  断点续跑（--resume）与并行完全兼容。

每个实验独立生成一个 CSV 文件保存到 eval/retrieval/results/，不会覆盖。

环境变量：
  OPENAI_API_KEY    (必须，用于 GPT-4o 打分)
  DEEPSEEK_API_KEY  (必须，用于 DeepSeek 生成回答)
"""

import os
import sys
import json
import csv
import pickle
import time
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# 让 experiment_config 可直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from experiment_config import (
    INDEX_VARIANTS, RETRIEVER_VARIANTS, EXPERIMENTS, EVAL_CONFIG,
)
from agentic_rag.llm_config import build_chat_llm

# ── 路径 ─────────────────────────────────────────────────
QA_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

# ── 多线程共享锁 ─────────────────────────────────────────
_ckpt_lock = threading.Lock()    # 断点文件写入锁
_print_lock = threading.Lock()   # 终端输出锁，防止并发 print 行内容交错


def parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    """解析命令行参数。"""
    args = {
        "only": None,
        "resume": False,
        "start_index": 1,
        "start_qid": None,
        "dataset": str(QA_DATASET),
        "workers": 1,
    }

    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--only":
            if i + 1 >= len(argv):
                raise ValueError("--only 需要一个逗号分隔的实验名列表")
            args["only"] = set(argv[i + 1].split(","))
            i += 2
        elif token == "--resume":
            args["resume"] = True
            i += 1
        elif token == "--start-index":
            if i + 1 >= len(argv):
                raise ValueError("--start-index 需要一个正整数")
            try:
                args["start_index"] = int(argv[i + 1])
            except ValueError as e:
                raise ValueError("--start-index 必须是整数") from e
            if args["start_index"] < 1:
                raise ValueError("--start-index 必须 >= 1")
            i += 2
        elif token == "--start-qid":
            if i + 1 >= len(argv):
                raise ValueError("--start-qid 需要一个题号")
            args["start_qid"] = str(argv[i + 1]).strip()
            i += 2
        elif token == "--dataset":
            if i + 1 >= len(argv):
                raise ValueError("--dataset 需要一个文件路径")
            args["dataset"] = argv[i + 1].strip()
            i += 2
        elif token == "--workers":
            if i + 1 >= len(argv):
                raise ValueError("--workers 需要一个正整数")
            try:
                args["workers"] = int(argv[i + 1])
            except ValueError as e:
                raise ValueError("--workers 必须是整数") from e
            if args["workers"] < 1:
                raise ValueError("--workers 必须 >= 1")
            i += 2
        else:
            raise ValueError(f"未知参数：{token}")

    if args["start_qid"] and args["start_index"] != 1:
        raise ValueError("--start-qid 与 --start-index 不能同时使用")

    if args["workers"] > 1 and args["start_qid"] is not None:
        raise ValueError("--workers > 1 时不支持 --start-qid，请改用 --start-index")

    return args


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", name).strip("_")


def _load_checkpoint_rows(checkpoint_path: Path) -> List[Dict[str, Any]]:
    if not checkpoint_path.exists():
        return []
    rows = []
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_checkpoint_row(checkpoint_path: Path, row: Dict[str, Any]) -> None:
    with _ckpt_lock:
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ══ 综合分权重（公式 8）═══════════════════════════════════
RETRIEVAL_OVERALL_WEIGHTS = {
    "relevance": 0.2,
    "faithfulness": 0.3,
    "completeness": 0.3,
    "technical_accuracy": 0.2,
}


def _compute_retrieval_overall(scores: Dict[str, Any]) -> Optional[float]:
    """按 RETRIEVAL_OVERALL_WEIGHTS 计算综合分（浮点，不做整数舍入）。

    保留浮点可保证 avg(overall_i) == sum(w_j * avg(dim_j))，
    使汇总表格与公式严格自洽。
    """
    vals = [scores.get(k) for k in RETRIEVAL_OVERALL_WEIGHTS]
    if any(v is None for v in vals):
        return None
    return round(sum(v * w for v, w in zip(vals, RETRIEVAL_OVERALL_WEIGHTS.values())), 3)


# ══ 打分 Prompt ══════════════════════════════════════════

JUDGE_SYSTEM = """\
你是计算机网络课程的专家评审，负责客观评估 AI 助教回答的质量。
你的打分必须严格、一致，不偏袒任何方案。
请严格对照每个维度的 5 级锚点描述打分，不要凭主观印象。"""

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

────────────────────────────────────────────────────

请严格按以下 JSON 格式输出（只需 4 个维度分 + 理由），不要有任何其他文字：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""

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

────────────────────────────────────────────────────

请严格按以下 JSON 格式输出（只需 4 个维度分 + 理由），不要有任何其他文字：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""


# ══ 检索器构建 ═══════════════════════════════════════════

def build_retriever(
    vectorstore: FAISS,
    retriever_config: dict,
    reranker_model: str,
    chunks: Optional[list] = None,
):
    """根据配置构建检索器。"""
    rtype = retriever_config["type"]
    k = retriever_config["k"]

    if rtype == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    elif rtype == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": retriever_config.get("fetch_k", 30),
                "lambda_mult": retriever_config.get("lambda_mult", 0.5),
            },
        )

    elif rtype == "mmr_rerank":
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

        base_k = retriever_config.get("base_k", 20)
        fetch_k = retriever_config.get("fetch_k", 50)

        cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)
        base = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": base_k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        )
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base,
        )

    elif rtype == "hybrid":
        return _build_ensemble_retriever(vectorstore, retriever_config, chunks)

    elif rtype == "hybrid_rerank":
        return _build_hybrid_rerank_retriever(
            vectorstore, retriever_config, reranker_model, chunks,
        )

    else:
        raise ValueError(f"未知的检索器类型：{rtype}")


def _build_ensemble_retriever(vectorstore, config, chunks):
    """构建 BM25 + Dense 的 RRF 融合检索器。"""
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        raise ImportError(
            "BM25 混合检索需要 rank_bm25 包，请运行：pip install rank-bm25"
        )

    if chunks is None:
        raise ValueError(
            "BM25 混合检索需要 chunks.pkl，请先运行 build_indices.py"
        )

    dense_k = config.get("dense_k", config.get("k", 6))
    fusion_k = config.get("fusion_k", config.get("k", 6))

    # 中文分词
    try:
        import jieba
        preprocess = lambda text: list(jieba.cut(text))
    except ImportError:
        print("  [提示] 未安装 jieba，BM25 将使用空格分词（中文效果较差）")
        preprocess = lambda text: text.split()

    bm25 = BM25Retriever.from_documents(
        chunks, preprocess_func=preprocess, k=config.get("bm25_k", fusion_k),
    )
    dense = vectorstore.as_retriever(
        search_type=config.get("dense_search_type", "similarity"),
        search_kwargs={"k": dense_k},
    )

    bm25_w = config.get("bm25_weight", 0.4)
    dense_w = config.get("dense_weight", 0.6)

    return _RRFRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_w, dense_w],
        k=fusion_k,
    )


class _RRFRetriever:
    """简单的 Reciprocal Rank Fusion 融合检索器。"""

    def __init__(self, retrievers, weights, k=6):
        self.retrievers = retrievers
        self.weights = weights
        self.k = k

    def invoke(self, query: str):
        doc_scores: Dict[str, dict] = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = doc.page_content[:200]
                if key not in doc_scores:
                    doc_scores[key] = {"doc": doc, "score": 0.0}
                doc_scores[key]["score"] += weight / (rank + 60)  # RRF 常数 60

        sorted_items = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        return [item["doc"] for item in sorted_items[: self.k]]

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


class _HybridRerankRetriever:
    """先做 BM25 + Dense 融合，再用 cross-encoder 精排。"""

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


def _build_hybrid_rerank_retriever(vectorstore, config, reranker_model, chunks):
    """构建 BM25 + Dense + RRF + Rerank 检索器。"""
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

    fusion_config = dict(config)
    fusion_config["fusion_k"] = config.get("fusion_k", max(config.get("k", 6) * 3, 18))
    base_retriever = _build_ensemble_retriever(vectorstore, fusion_config, chunks)

    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=config["k"])
    return _HybridRerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        k=config["k"],
    )


# ══ 回答生成 ════════════════════════════════════════════

ANSWER_SYSTEM = """\
你是网络实验排错助教。请仅依据已知信息回答问题，不得编造。
如果已知信息不足，回答"无法从已知信息确定"。"""


def generate_answer(llm, question: str, docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = (doc.metadata or {}).get("source", "unknown")
        parts.append(f"[{i}] {src}\n{doc.page_content.strip()}")
    context = "\n\n".join(parts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM + f"\n\n已知信息:\n{context}"),
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))


# ══ GPT-4o 打分 ═════════════════════════════════════════

def judge_answer(
    client: OpenAI,
    question: str,
    answer: str,
    docs: list,
    reference: str = "",
) -> Dict[str, Any]:
    snippet_len = EVAL_CONFIG.get("snippet_len", 300)
    answer_snippet_len = EVAL_CONFIG.get("answer_snippet", 1200)

    context_snippet = "\n---\n".join(
        d.page_content.strip()[:snippet_len] for d in docs
    )
    answer_snippet = answer[:answer_snippet_len]

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
        resp = client.chat.completions.create(
            model=EVAL_CONFIG.get("judge_model", "gpt-4o"),
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        scores["overall"] = _compute_retrieval_overall(scores)
        return scores
    except Exception as e:
        print(f"    [打分失败] {e}")
        return {
            "relevance": None, "faithfulness": None,
            "completeness": None, "technical_accuracy": None,
            "overall": None, "comment": f"打分失败: {e}",
        }


# ══ 单题处理（可并发调用）═══════════════════════════════

def _process_question_task(
    retriever,
    llm,
    openai_client: OpenAI,
    q_item: dict,
    idx: int,
    total: int,
    exp_name: str,
    index_name: str,
    retriever_name: str,
    checkpoint_path: Path,
    stagger_delay: float = 0.0,
) -> Dict[str, Any]:
    """执行单道题的完整流程：检索 → 生成 → 打分，线程安全。

    stagger_delay: 任务开始前的错开延迟（秒），并行模式下由调用方传入，
                   用于将各线程的首批 API 请求分散开，避免瞬时并发触发限速。
    """
    if stagger_delay > 0:
        time.sleep(stagger_delay)

    qid = q_item.get("id", "?")
    question = q_item.get("question", "")
    qtype = q_item.get("type", "unknown")
    reference = q_item.get("reference", "")

    try:
        docs = retriever.invoke(question)
    except Exception:
        docs = retriever.get_relevant_documents(question)

    answer = generate_answer(llm, question, docs)
    scores = judge_answer(openai_client, question, answer, docs, reference)

    # 整行一次性输出，避免并发时行内容交错
    with _print_lock:
        q_preview = question[:40] + ("..." if len(question) > 40 else "")
        print(f"  [{idx}/{total}] Q{qid}: {q_preview} overall={scores.get('overall', '?')}")

    row = {
        "experiment": exp_name,
        "question_id": qid,
        "question": question,
        "question_type": qtype,
        "index": index_name,
        "retriever": retriever_name,
        "retrieved_docs_count": len(docs),
        "answer_length": len(answer),
        "relevance": scores.get("relevance"),
        "faithfulness": scores.get("faithfulness"),
        "completeness": scores.get("completeness"),
        "technical_accuracy": scores.get("technical_accuracy"),
        "overall": scores.get("overall"),
        "comment": scores.get("comment", ""),
    }
    _append_checkpoint_row(checkpoint_path, row)
    return row


# ══ 单个实验运行 ═════════════════════════════════════════

def run_single_experiment(
    exp: dict,
    questions: list,
    embeddings: HuggingFaceEmbeddings,
    llm,
    openai_client: OpenAI,
    resume: bool = False,
    start_index: int = 1,
    start_qid: Optional[str] = None,
    workers: int = 1,
):
    """运行单个实验，返回 (rows, output_csv_path)。"""
    exp_name = exp["name"]
    index_name = exp["index"]
    retriever_name = exp["retriever"]

    index_config = INDEX_VARIANTS[index_name]
    retriever_config = RETRIEVER_VARIANTS[retriever_name]

    print(f"\n{'='*65}")
    print(f"实验：{exp_name}  (workers={workers})")
    print(f"  索引：{index_name} | 检索：{retriever_name}")
    print(f"{'='*65}")

    # 1. 加载索引
    index_dir = ROOT / index_config["index_dir"]
    if not index_dir.exists():
        print(f"  [跳过] 索引目录不存在：{index_dir}")
        print(f"  请先运行：python eval/retrieval/build_indices.py --only {index_name}")
        return [], None

    vs = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )

    # 2. 加载 chunks（BM25 需要）
    chunks = None
    chunks_path = index_dir / "chunks.pkl"
    if retriever_config["type"] in {"hybrid", "hybrid_rerank"}:
        if not chunks_path.exists():
            print(f"  [跳过] BM25 需要 chunks.pkl，但文件不存在：{chunks_path}")
            print(f"  请先运行：python eval/retrieval/build_indices.py --only {index_name}")
            return [], None
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

    # 3. 构建检索器
    reranker_model = EVAL_CONFIG.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    retriever = build_retriever(vs, retriever_config, reranker_model, chunks)

    # 4. 断点信息
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{_safe_name(exp_name)}.jsonl"
    checkpoint_rows = _load_checkpoint_rows(checkpoint_path) if resume else []
    completed_qids: set = {str(r.get("question_id")) for r in checkpoint_rows}
    rows = list(checkpoint_rows)

    if resume:
        print(f"  [断点续跑] 已加载 {len(checkpoint_rows)} 条历史结果：{checkpoint_path}")
    elif checkpoint_path.exists():
        print(f"  [提示] 存在历史断点文件但未启用 --resume：{checkpoint_path}")

    # 5. 筛选待处理题目
    total = len(questions)
    started_by_qid = start_qid is None
    pending: List[tuple] = []  # (原始序号, q_item)

    for i, q_item in enumerate(questions, 1):
        qid_str = str(q_item.get("id", "?"))

        if i < start_index:
            continue
        if not started_by_qid:
            if qid_str != start_qid:
                continue
            started_by_qid = True
        if qid_str in completed_qids:
            print(f"  [{i}/{total}] Q{qid_str}: 已在断点中，跳过")
            continue
        pending.append((i, q_item))

    if start_qid and not started_by_qid:
        raise ValueError(f"--start-qid={start_qid} 未在数据集中找到")

    # 6. 执行评测（顺序 or 并发）
    if workers == 1:
        for i, q_item in pending:
            row = _process_question_task(
                retriever, llm, openai_client,
                q_item, i, total,
                exp_name, index_name, retriever_name,
                checkpoint_path,
            )
            rows.append(row)
            completed_qids.add(str(q_item.get("id", "?")))
            time.sleep(0.5)  # 顺序模式下拉开相邻请求间距，降低 API 限速风险
    else:
        print(f"  [并行] 提交 {len(pending)} 道题，最多 {workers} 线程同时运行…")
        # 同一批次内各线程错开 1.0s，将每批首个 API 请求分散，降低瞬时并发触发限速的概率
        # 使用 slot % workers 让错开量始终在 [0, workers-1] 秒范围内
        _STAGGER_INTERVAL = 1.0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    _process_question_task,
                    retriever, llm, openai_client,
                    q_item, i, total,
                    exp_name, index_name, retriever_name,
                    checkpoint_path,
                    (slot % workers) * _STAGGER_INTERVAL,  # stagger_delay
                ): i
                for slot, (i, q_item) in enumerate(pending)
            }
            for future in as_completed(future_to_idx):
                try:
                    row = future.result()
                    rows.append(row)
                except Exception as exc:
                    orig_idx = future_to_idx[future]
                    with _print_lock:
                        # 用 format_exc() 转成字符串再 print，确保走 stdout 并受锁保护
                        print(f"  [错误] 第 {orig_idx} 题处理失败：{exc}")
                        print(traceback.format_exc(), end="")

    # 7. 保存结果 CSV（按 question_id 排序保持可读性）
    rows.sort(key=lambda r: r.get("question_id", 0))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = RESULTS_DIR / f"{exp_name}_{timestamp}.csv"

    fieldnames = [
        "experiment", "question_id", "question", "question_type",
        "index", "retriever",
        "retrieved_docs_count", "answer_length",
        "relevance", "faithfulness", "completeness", "technical_accuracy",
        "overall", "comment",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  结果已保存：{output_csv}")
    return rows, output_csv


# ══ 汇总打印 ════════════════════════════════════════════

def print_summary(all_rows: Dict[str, list]):
    """打印所有实验的汇总对比表。"""
    print(f"\n\n{'='*75}")
    print("实验方案对比汇总（各维度平均分）")
    print(f"{'='*75}")
    print(f"{'实验名称':<28} {'相关性':>6} {'忠实性':>6} {'完整性':>6} {'技术准确':>8} {'综合':>6}")
    print(f"{'-'*75}")

    for exp_name, rows in all_rows.items():
        if not rows:
            continue

        def avg(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return f"{sum(vals)/len(vals):.2f}" if vals else "N/A"

        print(
            f"{exp_name:<28} {avg('relevance'):>6} {avg('faithfulness'):>6} "
            f"{avg('completeness'):>6} {avg('technical_accuracy'):>8} {avg('overall'):>6}"
        )

    print(f"{'='*75}")


# ══ 主流程 ══════════════════════════════════════════════

def main():
    # 检查环境变量
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    # 解析参数
    try:
        cli_args = parse_cli_args(sys.argv[1:])
    except ValueError as e:
        print(f"参数错误：{e}")
        print("用法示例：")
        print("  python eval/retrieval/run_experiments.py")
        print("  python eval/retrieval/run_experiments.py --only baseline_similarity")
        print("  python eval/retrieval/run_experiments.py --resume")
        print("  python eval/retrieval/run_experiments.py --dataset eval/qa_dataset_small.json")
        print("  python eval/retrieval/run_experiments.py --start-index 37")
        print("  python eval/retrieval/run_experiments.py --start-qid 37")
        print("  python eval/retrieval/run_experiments.py --workers 4")
        sys.exit(1)

    dataset_path = Path(cli_args["dataset"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        print(f"错误：找不到问题数据集 {dataset_path}")
        sys.exit(1)

    experiments_to_run = [
        e for e in EXPERIMENTS
        if cli_args["only"] is None or e["name"] in cli_args["only"]
    ]

    if not experiments_to_run:
        print("没有要运行的实验。可用实验：")
        for e in EXPERIMENTS:
            print(f"  - {e['name']} (索引: {e['index']}, 检索: {e['retriever']})")
        sys.exit(1)

    # 加载数据集
    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"加载 {len(questions)} 个测试问题（{dataset_path}）")
    print(f"将运行 {len(experiments_to_run)} 个实验：{', '.join(e['name'] for e in experiments_to_run)}")
    if cli_args["resume"]:
        print("启用断点续跑：--resume")
    if cli_args["start_index"] != 1:
        print(f"从题目序号开始：--start-index {cli_args['start_index']}")
    if cli_args["start_qid"] is not None:
        print(f"从题号开始：--start-qid {cli_args['start_qid']}")
    if cli_args["workers"] > 1:
        print(f"并行线程数：--workers {cli_args['workers']}")

    # 初始化共享资源
    print("\n初始化模型...")
    embedding_model = EVAL_CONFIG.get("embedding_model", "BAAI/bge-m3")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    llm = build_chat_llm(temperature=0)
    openai_client = OpenAI(api_key=openai_key)
    print("模型初始化完成\n")

    # 逐个运行实验（实验间顺序执行，实验内并发）
    all_rows = {}
    for exp in experiments_to_run:
        rows, csv_path = run_single_experiment(
            exp,
            questions,
            embeddings,
            llm,
            openai_client,
            resume=cli_args["resume"],
            start_index=cli_args["start_index"],
            start_qid=cli_args["start_qid"],
            workers=cli_args["workers"],
        )
        all_rows[exp["name"]] = rows

    # 汇总
    print_summary(all_rows)
    print(f"\n原始数据见：{RESULTS_DIR}/")
    print("详细对比请运行：python eval/retrieval/compare_experiments.py")


if __name__ == "__main__":
    main()
