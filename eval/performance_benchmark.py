"""
系统性能基准测试

测量 Agent 端到端响应延迟，并分解为各阶段耗时：
  - 分类（classify_unified）
  - 检索（RAG retrieval）
  - 重排（Reranker）
  - LLM 生成（DeepSeek）
  - 总端到端

统计 P50/P95/P99 延迟，验证是否满足教学场景需求（< 15s）。

用法：
  python eval/performance_benchmark.py                    # 默认 40 题
  python eval/performance_benchmark.py --n-samples 50     # 指定题数
  python eval/performance_benchmark.py --dataset eval/qa_dataset_topo_balanced.json

输出：
  eval/results/performance_YYYYMMDD_HHMMSS.csv   — 逐题耗时明细
  eval/results/performance_YYYYMMDD_HHMMSS.json  — 统计汇总
  eval/results/performance_YYYYMMDD_HHMMSS.png   — 延迟分布图

环境变量：
  DEEPSEEK_API_KEY (必须)
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
from typing import Dict, List
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

# ── 配置 ─────────────────────────────────────────────
QA_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"


class TimingContext:
    """用于收集各阶段耗时的上下文管理器。"""
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._stack: List[tuple] = []

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.timings[name] = self.timings.get(name, 0) + elapsed


def stratified_sample(questions: list, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    by_type: Dict[str, list] = {}
    for q in questions:
        t = q.get("type", "unknown")
        by_type.setdefault(t, []).append(q)

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


def benchmark_single_query(question: str) -> Dict:
    """对单个问题进行端到端性能测量。"""
    from agentic_rag.agent import (
        _prepare_context, Agent, _find_action,
        _execute_tool_action, _format_citations, _get_client,
    )

    timings = {}

    # 阶段 1: 前置处理（含分类 LLM 调用）
    t0 = time.perf_counter()
    state = {}
    ctx = _prepare_context(question, [], state, None, False, False)
    timings["classification"] = round(time.perf_counter() - t0, 3)

    if ctx.early_reply is not None:
        return {
            "total": timings["classification"],
            "classification": timings["classification"],
            "retrieval": 0,
            "generation": 0,
            "tool_calls": 0,
            "early_exit": True,
        }

    # 阶段 2: Agent 循环（包含检索 + 生成）
    bot = Agent(ctx.final_prompt, [])
    tool_traces = []
    last_citations = []
    total_retrieval = 0
    total_generation = 0
    tool_call_count = 0

    next_prompt = question
    for i in range(5):
        # LLM 生成
        t0 = time.perf_counter()
        result = bot(next_prompt)
        gen_time = time.perf_counter() - t0
        total_generation += gen_time

        # 检查工具调用
        action_match = _find_action(result)
        if action_match:
            tool_call_count += 1
            t0 = time.perf_counter()
            next_prompt = _execute_tool_action(
                action_match, ctx.contextual_actions, tool_traces, last_citations,
            )
            retrieval_time = time.perf_counter() - t0
            total_retrieval += retrieval_time
            continue
        break

    timings["retrieval"] = round(total_retrieval, 3)
    timings["generation"] = round(total_generation, 3)
    timings["total"] = round(
        timings["classification"] + timings["retrieval"] + timings["generation"], 3
    )
    timings["tool_calls"] = tool_call_count
    timings["early_exit"] = False

    return timings


def main():
    parser = argparse.ArgumentParser(description="系统性能基准测试")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(QA_DATASET),
        help="题库路径（默认 eval/qa_dataset.json）",
    )
    parser.add_argument("--n-samples", type=int, default=40, help="测试题数 (默认 40)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        print(f"错误：找不到题库文件 {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    sampled = stratified_sample(all_questions, args.n_samples, seed=args.seed)
    print(f"题库：{dataset_path}")
    print(f"从 {len(all_questions)} 题中抽样 {len(sampled)} 题进行性能测试")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 预热（第一次加载模型较慢，不计入统计）
    print("预热中（加载模型）...", end=" ", flush=True)
    try:
        from agentic_rag.agent import _get_client
        _get_client()
        from agentic_rag.rag import RAGAgent
        RAGAgent("test", category="THEORY_CONCEPT", hint_level=0)
    except Exception:
        pass
    print("完成\n")

    # 逐题测试
    detail_rows = []
    total = len(sampled)

    for i, q_item in enumerate(sampled):
        qid = q_item.get("id", "?")
        question = q_item.get("question", "")
        qtype = q_item.get("type", "unknown")

        print(f"  [{i+1}/{total}] Q{qid}: {question[:40]}...", end=" ", flush=True)

        try:
            timings = benchmark_single_query(question)
            print(f"total={timings['total']:.1f}s "
                  f"(分类={timings['classification']:.1f}s "
                  f"检索={timings['retrieval']:.1f}s "
                  f"生成={timings['generation']:.1f}s "
                  f"工具={timings['tool_calls']}次)")
        except Exception as e:
            print(f"[失败: {e}]")
            timings = {
                "total": None, "classification": None,
                "retrieval": None, "generation": None,
                "tool_calls": 0, "early_exit": False,
            }

        detail_rows.append({
            "question_id": qid,
            "question_type": qtype,
            "total_s": timings.get("total"),
            "classification_s": timings.get("classification"),
            "retrieval_s": timings.get("retrieval"),
            "generation_s": timings.get("generation"),
            "tool_calls": timings.get("tool_calls", 0),
            "early_exit": timings.get("early_exit", False),
        })

    # 保存逐题明细
    detail_csv = RESULTS_DIR / f"performance_{timestamp}.csv"
    fieldnames = [
        "question_id", "question_type",
        "total_s", "classification_s", "retrieval_s", "generation_s",
        "tool_calls", "early_exit",
    ]
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"\n明细已保存：{detail_csv}")

    # 统计汇总
    import numpy as np

    valid_rows = [r for r in detail_rows if r["total_s"] is not None and not r["early_exit"]]
    if not valid_rows:
        print("无有效数据")
        return

    summary = {"timestamp": timestamp, "n_samples": len(sampled), "n_valid": len(valid_rows)}

    print(f"\n{'='*65}")
    print("系统性能基准测试结果")
    print(f"{'='*65}")
    print(f"有效样本：{len(valid_rows)} / {len(sampled)}")

    stages = [
        ("total_s", "端到端总延迟"),
        ("classification_s", "分类阶段"),
        ("retrieval_s", "检索+重排阶段"),
        ("generation_s", "LLM 生成阶段"),
    ]

    print(f"\n{'阶段':<16} {'平均':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'最大':>8}")
    print(f"{'-'*65}")

    for key, label in stages:
        vals = [r[key] for r in valid_rows if r.get(key) is not None]
        if not vals:
            continue
        arr = np.array(vals)
        stats = {
            "mean": round(float(np.mean(arr)), 3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "max": round(float(np.max(arr)), 3),
            "min": round(float(np.min(arr)), 3),
        }
        summary[key] = stats
        print(
            f"{label:<16} {stats['mean']:>7.2f}s {stats['p50']:>7.2f}s "
            f"{stats['p95']:>7.2f}s {stats['p99']:>7.2f}s {stats['max']:>7.2f}s"
        )

    # 工具调用统计
    tool_counts = [r["tool_calls"] for r in valid_rows]
    avg_tools = np.mean(tool_counts)
    print(f"\n平均工具调用次数：{avg_tools:.1f}")

    # 可用性判定
    total_vals = [r["total_s"] for r in valid_rows if r.get("total_s") is not None]
    p95 = np.percentile(total_vals, 95)
    verdict = "通过" if p95 < 15 else "未通过"
    summary["verdict"] = verdict
    summary["threshold_s"] = 15
    print(f"\n可用性判定：P95={p95:.2f}s {'<' if p95 < 15 else '>='} 15s → {verdict}")

    # 按题型分组
    print(f"\n{'='*65}")
    print("按题型分组延迟")
    print(f"{'='*65}")

    types = set(r["question_type"] for r in valid_rows)
    by_type_stats = {}
    for qtype in sorted(types):
        type_rows = [r for r in valid_rows if r["question_type"] == qtype]
        type_totals = [r["total_s"] for r in type_rows if r.get("total_s") is not None]
        if type_totals:
            avg = np.mean(type_totals)
            p50 = np.percentile(type_totals, 50)
            by_type_stats[qtype] = {"avg": round(avg, 3), "p50": round(p50, 3), "n": len(type_totals)}
            print(f"  {qtype:<20} n={len(type_totals):>3}  avg={avg:.2f}s  p50={p50:.2f}s")

    summary["by_type"] = by_type_stats
    print(f"{'='*65}")

    # 保存汇总
    summary_json = RESULTS_DIR / f"performance_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    # 可视化
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. 延迟分布直方图
        ax1 = axes[0]
        total_vals_arr = np.array(total_vals)
        ax1.hist(total_vals_arr, bins=15, color="#3498db", alpha=0.8, edgecolor="white")
        ax1.axvline(x=15, color="red", linestyle="--", alpha=0.7, label="阈值 (15s)")
        ax1.axvline(x=np.mean(total_vals_arr), color="green", linestyle="-",
                    alpha=0.7, label=f"均值 ({np.mean(total_vals_arr):.1f}s)")
        ax1.set_xlabel("端到端延迟 (s)")
        ax1.set_ylabel("题数")
        ax1.set_title("端到端延迟分布")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # 2. 各阶段耗时占比饼图
        ax2 = axes[1]
        stage_means = []
        stage_labels_pie = []
        stage_colors = ["#e74c3c", "#3498db", "#2ecc71"]
        for key, label in [("classification_s", "分类"), ("retrieval_s", "检索+重排"), ("generation_s", "LLM生成")]:
            vals = [r[key] for r in valid_rows if r.get(key) is not None]
            if vals:
                stage_means.append(np.mean(vals))
                stage_labels_pie.append(label)

        if stage_means:
            wedges, texts, autotexts = ax2.pie(
                stage_means, labels=stage_labels_pie, autopct="%1.1f%%",
                colors=stage_colors[:len(stage_means)], startangle=90,
            )
            ax2.set_title("各阶段耗时占比")

        # 3. 按题型箱线图
        ax3 = axes[2]
        type_data = []
        type_labels_box = []
        for qtype in sorted(types):
            vals = [r["total_s"] for r in valid_rows
                    if r["question_type"] == qtype and r.get("total_s") is not None]
            if vals:
                type_data.append(vals)
                type_labels_box.append(qtype)

        if type_data:
            bp = ax3.boxplot(type_data, labels=type_labels_box, patch_artist=True)
            box_colors = plt.cm.Set2(np.linspace(0, 1, len(type_data)))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax3.set_ylabel("端到端延迟 (s)")
            ax3.set_title("按题型延迟分布")
            ax3.grid(axis="y", alpha=0.3)

        fig.suptitle("系统性能基准测试", fontsize=14, y=1.02)
        fig.tight_layout()
        png_path = RESULTS_DIR / f"performance_{timestamp}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"图表已保存：{png_path}")

    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
