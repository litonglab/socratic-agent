"""
问题分类器准确性评估

使用 qa_dataset.json 中的 type 标签作为金标准，
调用 classify_unified() 获取预测分类，计算混淆矩阵和各类指标。

类型映射：
  qa_dataset.type         →  classify_unified category
  "concept"               →  THEORY_CONCEPT
  "troubleshooting"       →  LAB_TROUBLESHOOTING
  "config"                →  CONFIG_REVIEW
  "calculation"           →  CALCULATION

用法：
  python eval/classification_accuracy.py

输出：
  eval/results/classification_YYYYMMDD_HHMMSS.csv   — 逐题分类结果
  eval/results/classification_YYYYMMDD_HHMMSS.json  — 指标汇总
  eval/results/classification_YYYYMMDD_HHMMSS.png   — 混淆矩阵热力图

环境变量：
  DEEPSEEK_API_KEY (必须，用于 classify_unified 调用 LLM)
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from agentic_rag.agent import classify_unified

# ── 配置 ─────────────────────────────────────────────
QA_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"

# 金标准映射
TYPE_TO_CATEGORY = {
    "concept": "THEORY_CONCEPT",
    "troubleshooting": "LAB_TROUBLESHOOTING",
    "config": "CONFIG_REVIEW",
    "calculation": "CALCULATION",
}

CATEGORY_LABELS = ["THEORY_CONCEPT", "LAB_TROUBLESHOOTING", "CONFIG_REVIEW", "CALCULATION"]
CATEGORY_ZH = {
    "THEORY_CONCEPT": "理论概念",
    "LAB_TROUBLESHOOTING": "实验排错",
    "CONFIG_REVIEW": "配置操作",
    "CALCULATION": "计算分析",
}


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """计算各类 Precision, Recall, F1, 以及 Accuracy 和 Macro-F1。"""
    # 混淆矩阵
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            matrix[t][p] += 1

    # 各类指标
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # 总体指标
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {k: dict(v) for k, v in matrix.items()},
        "total": len(y_true),
        "correct": correct,
    }


def main():
    # 加载数据集
    with open(QA_DATASET, encoding="utf-8") as f:
        questions = json.load(f)

    # 过滤有 type 标签的题目
    valid_questions = [q for q in questions if q.get("type") in TYPE_TO_CATEGORY]
    print(f"加载 {len(valid_questions)} 个有类型标签的题目")
    type_dist = {}
    for q in valid_questions:
        t = q["type"]
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"  类型分布：{type_dist}")

    # 逐题分类
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    detail_rows = []

    total = len(valid_questions)
    for i, q_item in enumerate(valid_questions):
        qid = q_item.get("id", "?")
        question = q_item.get("question", "")
        true_type = q_item["type"]
        true_category = TYPE_TO_CATEGORY[true_type]

        print(f"  [{i+1}/{total}] Q{qid}: {question[:40]}...", end=" ", flush=True)

        # 调用分类器（空历史、默认 state）
        try:
            result = classify_unified(question, history=[], state={})
            pred_category = result.get("category", "")
            relevance = result.get("relevance", True)
            hint_decision = result.get("hint_decision", "")
        except Exception as e:
            print(f"[分类失败: {e}]")
            pred_category = "UNKNOWN"
            relevance = None
            hint_decision = ""

        correct = pred_category == true_category
        print(f"真={true_category[:8]} 预={pred_category[:8]} {'✓' if correct else '✗'}")

        y_true.append(true_category)
        y_pred.append(pred_category)

        detail_rows.append({
            "question_id": qid,
            "question": question[:100],
            "true_type": true_type,
            "true_category": true_category,
            "pred_category": pred_category,
            "correct": correct,
            "relevance": relevance,
            "hint_decision": hint_decision,
        })

        time.sleep(0.3)  # 避免 API 频率限制

    # 保存逐题明细
    detail_csv = RESULTS_DIR / f"classification_{timestamp}.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"\n明细已保存：{detail_csv}")

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, CATEGORY_LABELS)

    print(f"\n{'='*65}")
    print("问题分类准确性评估结果")
    print(f"{'='*65}")
    print(f"总题数：{metrics['total']} | 正确：{metrics['correct']} | "
          f"准确率：{metrics['accuracy']:.1%} | Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"\n{'类别':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"{'-'*65}")

    for label in CATEGORY_LABELS:
        c = metrics["per_class"].get(label, {})
        zh = CATEGORY_ZH.get(label, label)
        print(
            f"{zh:<20} {c.get('precision', 0):>10.4f} {c.get('recall', 0):>10.4f} "
            f"{c.get('f1', 0):>10.4f} {c.get('support', 0):>10}"
        )

    # 混淆矩阵
    print(f"\n混淆矩阵（行=真实，列=预测）：")
    cm = metrics["confusion_matrix"]
    header = f"{'':>14}"
    for label in CATEGORY_LABELS:
        header += f" {CATEGORY_ZH.get(label, label)[:4]:>8}"
    print(header)
    for actual in CATEGORY_LABELS:
        line = f"{CATEGORY_ZH.get(actual, actual)[:6]:>14}"
        for pred in CATEGORY_LABELS:
            line += f" {cm.get(actual, {}).get(pred, 0):>8}"
        print(line)

    print(f"{'='*65}")

    # 保存汇总 JSON
    summary_json = RESULTS_DIR / f"classification_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    # 混淆矩阵热力图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(8, 6))

        cm_array = np.array([
            [cm.get(actual, {}).get(pred, 0) for pred in CATEGORY_LABELS]
            for actual in CATEGORY_LABELS
        ])

        im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        labels_zh = [CATEGORY_ZH.get(l, l) for l in CATEGORY_LABELS]
        ax.set(xticks=np.arange(len(labels_zh)),
               yticks=np.arange(len(labels_zh)),
               xticklabels=labels_zh,
               yticklabels=labels_zh,
               ylabel="真实类别",
               xlabel="预测类别",
               title=f"问题分类混淆矩阵\n(准确率: {metrics['accuracy']:.1%}, Macro-F1: {metrics['macro_f1']:.4f})")

        # 在格子中标注数字
        thresh = cm_array.max() / 2
        for i in range(len(CATEGORY_LABELS)):
            for j in range(len(CATEGORY_LABELS)):
                ax.text(j, i, format(cm_array[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm_array[i, j] > thresh else "black",
                        fontsize=14)

        fig.tight_layout()
        png_path = RESULTS_DIR / f"classification_{timestamp}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵热力图：{png_path}")

    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
