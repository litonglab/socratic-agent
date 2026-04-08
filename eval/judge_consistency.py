"""
LLM 裁判自一致性验证实验

对 GPT-4o 裁判进行自一致性验证：对同一 (question, answer) 对独立打分 N 次，
统计各维度的标准差和 ICC（组内相关系数），验证打分稳定性。

用法：
  python eval/judge_consistency.py                     # 默认 25 题 × 3 次打分
  python eval/judge_consistency.py --n-samples 20      # 指定抽样数
  python eval/judge_consistency.py --n-repeats 5       # 指定重复打分次数
  python eval/judge_consistency.py --dataset eval/qa_dataset_topo_balanced.json
  python eval/judge_consistency.py --experiment baseline_mmr_rerank  # 指定实验方案

输出：
  eval/results/judge_consistency_YYYYMMDD_HHMMSS.csv   — 逐题逐次打分明细
  eval/results/judge_consistency_YYYYMMDD_HHMMSS.json  — 汇总统计
  eval/results/judge_consistency_YYYYMMDD_HHMMSS.png   — 偏差分布箱线图

环境变量：
  OPENAI_API_KEY   (必须，用于 GPT-4o 打分)
  DEEPSEEK_API_KEY (必须，用于 DeepSeek 生成回答)
"""

import os
import sys
import json
import csv
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "retrieval"))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from run_experiments import (
    build_retriever, generate_answer, JUDGE_SYSTEM,
    JUDGE_USER_TEMPLATE_WITH_REF, JUDGE_USER_TEMPLATE_NO_REF,
)
from experiment_config import RETRIEVER_VARIANTS, INDEX_VARIANTS, EVAL_CONFIG
from agentic_rag.llm_config import build_chat_llm

# ── 配置 ─────────────────────────────────────────────
QA_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"
SCORE_KEYS = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
SCORE_NAMES_ZH = {
    "relevance": "相关性", "faithfulness": "忠实性",
    "completeness": "完整性", "technical_accuracy": "技术准确性",
    "overall": "综合",
}


def stratified_sample(questions: list, n: int, seed: int = 42) -> list:
    """按 question type 分层抽样。"""
    rng = random.Random(seed)
    by_type: Dict[str, list] = {}
    for q in questions:
        t = q.get("type", "unknown")
        by_type.setdefault(t, []).append(q)

    # 每类至少 n // len(by_type) 题，多余的随机补
    per_type = max(1, n // len(by_type))
    sampled = []
    remaining = []
    for t, qs in by_type.items():
        rng.shuffle(qs)
        sampled.extend(qs[:per_type])
        remaining.extend(qs[per_type:])

    need = n - len(sampled)
    if need > 0:
        rng.shuffle(remaining)
        sampled.extend(remaining[:need])

    return sampled[:n]


def judge_answer_once(
    client: OpenAI,
    question: str,
    answer: str,
    context_snippet: str,
    reference: str = "",
) -> Dict[str, Any]:
    """调用 GPT-4o 打分一次（temperature=0 但每次独立 session）。"""
    if reference:
        user_msg = JUDGE_USER_TEMPLATE_WITH_REF.format(
            question=question,
            reference=reference,
            answer=answer[:1200],
            context=context_snippet,
        )
    else:
        user_msg = JUDGE_USER_TEMPLATE_NO_REF.format(
            question=question,
            answer=answer[:1200],
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
        return json.loads(raw)
    except Exception as e:
        print(f"    [打分失败] {e}")
        return {k: None for k in SCORE_KEYS}


def compute_icc(scores_matrix: List[List[float]]) -> float:
    """计算 ICC(2,1)（双向随机效应，单个评分者一致性）。

    scores_matrix: n_subjects × n_raters，每个元素是分数。
    """
    import numpy as np

    data = np.array(scores_matrix, dtype=float)
    n, k = data.shape  # n=题数, k=评分次数

    if n < 2 or k < 2:
        return float("nan")

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_total = ((data - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_cols = ss_cols / (k - 1)

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    return float(icc)


def main():
    parser = argparse.ArgumentParser(description="LLM 裁判自一致性验证")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(QA_DATASET),
        help="题库路径（默认 eval/qa_dataset.json）",
    )
    parser.add_argument("--n-samples", type=int, default=25, help="抽样题数 (默认 25)")
    parser.add_argument("--n-repeats", type=int, default=3, help="每题重复打分次数 (默认 3)")
    parser.add_argument("--experiment", type=str, default="baseline_mmr_rerank",
                        help="使用的实验方案 (默认 baseline_mmr_rerank)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 检查环境变量
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        print(f"错误：找不到题库文件 {dataset_path}")
        sys.exit(1)

    # 加载数据集
    with open(dataset_path, encoding="utf-8") as f:
        all_questions = json.load(f)
    sampled = stratified_sample(all_questions, args.n_samples, seed=args.seed)
    print(f"题库：{dataset_path}")
    print(f"从 {len(all_questions)} 题中分层抽样 {len(sampled)} 题")
    type_dist = {}
    for q in sampled:
        t = q.get("type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"  类型分布：{type_dist}")

    # 初始化模型
    print("\n初始化模型...")
    embedding_model = EVAL_CONFIG.get("embedding_model", "BAAI/bge-m3")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    # 确定索引和检索器
    exp_name = args.experiment
    # 从实验名推断索引和检索器
    index_name = "baseline"
    retriever_name = "mmr_rerank"
    for exp_cfg in [
        {"name": "baseline_similarity", "index": "baseline", "retriever": "similarity"},
        {"name": "baseline_mmr", "index": "baseline", "retriever": "mmr"},
        {"name": "baseline_mmr_rerank", "index": "baseline", "retriever": "mmr_rerank"},
        {"name": "large_mmr_rerank", "index": "large_chunk", "retriever": "mmr_rerank"},
        {"name": "enriched_mmr_rerank", "index": "enriched", "retriever": "mmr_rerank"},
    ]:
        if exp_cfg["name"] == exp_name:
            index_name = exp_cfg["index"]
            retriever_name = exp_cfg["retriever"]
            break

    index_config = INDEX_VARIANTS[index_name]
    retriever_config = RETRIEVER_VARIANTS[retriever_name]
    index_dir = ROOT / index_config["index_dir"]

    vs = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    reranker_model = EVAL_CONFIG.get("reranker_model", "BAAI/bge-reranker-v2-m3")
    retriever = build_retriever(vs, retriever_config, reranker_model)

    llm = build_chat_llm(temperature=0)
    openai_client = OpenAI(api_key=openai_key)
    print("模型初始化完成\n")

    # 逐题评测
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_csv = RESULTS_DIR / f"judge_consistency_{timestamp}.csv"
    summary_json = RESULTS_DIR / f"judge_consistency_{timestamp}.json"

    fieldnames = [
        "question_id", "question_type", "repeat", *SCORE_KEYS, "comment",
    ]
    all_rows = []

    # 收集每题每维度的分数矩阵（用于 ICC 计算）
    # scores_by_dim[dim] = [[q1_r1, q1_r2, ...], [q2_r1, q2_r2, ...], ...]
    scores_by_dim = {k: [] for k in SCORE_KEYS}

    total = len(sampled) * args.n_repeats
    done = 0

    for qi, q_item in enumerate(sampled):
        qid = q_item.get("id", "?")
        question = q_item.get("question", "")
        qtype = q_item.get("type", "unknown")
        reference = q_item.get("reference", "")

        print(f"\n── 问题 {qi+1}/{len(sampled)} (ID={qid}): {question[:50]}...")

        # 1. 检索 + 生成回答（只做一次，保证打分对象相同）
        try:
            docs = retriever.invoke(question)
        except Exception:
            docs = retriever.get_relevant_documents(question)

        answer = generate_answer(llm, question, docs)
        snippet_len = EVAL_CONFIG.get("snippet_len", 300)
        context_snippet = "\n---\n".join(
            d.page_content.strip()[:snippet_len] for d in docs
        )

        # 2. 重复打分 N 次
        repeat_scores = {k: [] for k in SCORE_KEYS}

        for r in range(args.n_repeats):
            done += 1
            print(f"  打分 {r+1}/{args.n_repeats} [{done}/{total}]...", end=" ", flush=True)
            scores = judge_answer_once(
                openai_client, question, answer, context_snippet, reference,
            )
            print(f"overall={scores.get('overall', '?')}")

            row = {
                "question_id": qid,
                "question_type": qtype,
                "repeat": r + 1,
                **{k: scores.get(k) for k in SCORE_KEYS},
                "comment": scores.get("comment", ""),
            }
            all_rows.append(row)

            for k in SCORE_KEYS:
                v = scores.get(k)
                repeat_scores[k].append(v if v is not None else float("nan"))

            time.sleep(1.0)  # 避免 API 频率限制

        for k in SCORE_KEYS:
            scores_by_dim[k].append(repeat_scores[k])

    # 保存逐题明细 CSV
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n明细已保存：{detail_csv}")

    # 计算统计汇总
    import numpy as np

    summary = {
        "experiment": args.experiment,
        "n_samples": len(sampled),
        "n_repeats": args.n_repeats,
        "timestamp": timestamp,
        "dimensions": {},
    }

    print(f"\n{'='*65}")
    print("LLM 裁判自一致性验证结果")
    print(f"{'='*65}")
    print(f"实验方案：{args.experiment} | 抽样：{len(sampled)} 题 | 重复：{args.n_repeats} 次")
    print(f"\n{'维度':<12} {'平均标准差':>10} {'最大标准差':>10} {'ICC':>8} {'判定':>6}")
    print(f"{'-'*65}")

    for dim in SCORE_KEYS:
        matrix = scores_by_dim[dim]
        # 过滤掉含 NaN 的行
        valid_matrix = [row for row in matrix if all(not np.isnan(v) for v in row)]

        if not valid_matrix:
            print(f"{SCORE_NAMES_ZH.get(dim, dim):<12} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
            continue

        stds = [float(np.std(row)) for row in valid_matrix]
        mean_std = float(np.mean(stds))
        max_std = float(np.max(stds))
        icc = compute_icc(valid_matrix)

        verdict = "通过" if mean_std <= 0.5 and icc >= 0.7 else "待检查"

        summary["dimensions"][dim] = {
            "mean_std": round(mean_std, 4),
            "max_std": round(max_std, 4),
            "icc": round(icc, 4),
            "verdict": verdict,
        }

        print(f"{SCORE_NAMES_ZH.get(dim, dim):<12} {mean_std:>10.4f} {max_std:>10.4f} {icc:>8.4f} {verdict:>6}")

    print(f"{'='*65}")
    print(f"通过标准：平均标准差 ≤ 0.5 且 ICC ≥ 0.7")

    # 保存汇总 JSON
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    # 生成箱线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 6))

        # 每个维度的标准差分布
        dim_stds = []
        dim_labels = []
        for dim in SCORE_KEYS:
            matrix = scores_by_dim[dim]
            valid = [row for row in matrix if all(not np.isnan(v) for v in row)]
            if valid:
                stds = [float(np.std(row)) for row in valid]
                dim_stds.append(stds)
                dim_labels.append(SCORE_NAMES_ZH.get(dim, dim))

        if dim_stds:
            bp = ax.boxplot(dim_stds, labels=dim_labels, patch_artist=True)
            colors = ["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="阈值 (0.5)")
            ax.set_ylabel("标准差")
            ax.set_title(f"GPT-4o 裁判自一致性 — 各维度打分标准差分布\n"
                         f"({len(sampled)} 题 × {args.n_repeats} 次打分)")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            png_path = RESULTS_DIR / f"judge_consistency_{timestamp}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"箱线图已保存：{png_path}")

    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
