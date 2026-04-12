"""
检索实验统计分析与可视化

对现有检索实验结果进行增强统计分析：
  - 配对 Wilcoxon 签名秩检验（最优方案 vs 各方案）
  - 效应量（Cliff's delta）
  - Chunk size 最优曲线图
  - 按题型交叉分析雷达图
  - 完整对比表（含统计显著性标注）

用法：
  python eval/retrieval/statistical_analysis.py              # 分析所有结果
  python eval/retrieval/statistical_analysis.py --latest     # 每实验只取最新结果

输出：
  eval/retrieval/results/statistical_analysis_TIMESTAMP/
    ├── summary_table.csv          — 完整对比表（含显著性）
    ├── significance_tests.csv     — 配对检验结果
    ├── chunk_size_curve.png       — Chunk size 最优曲线图
    ├── radar_by_type.png          — 按题型雷达图
    └── overall_comparison.png     — 综合分数柱状图
"""

import csv
import re
import sys
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORE_KEYS = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
SCORE_NAMES_ZH = {
    "relevance": "相关性", "faithfulness": "忠实性",
    "completeness": "完整性", "technical_accuracy": "技术准确性",
    "overall": "综合",
}

RETRIEVAL_OVERALL_WEIGHTS = {
    "relevance": 0.2,
    "faithfulness": 0.3,
    "completeness": 0.3,
    "technical_accuracy": 0.2,
}


def _recalc_overall(row: dict) -> None:
    """从四维原始分重算 overall（覆盖 CSV 中可能存在的旧整数值）。"""
    vals = [row.get(k) for k in RETRIEVAL_OVERALL_WEIGHTS]
    if any(v is None for v in vals):
        row["overall"] = None
    else:
        row["overall"] = round(
            sum(v * w for v, w in zip(vals, RETRIEVAL_OVERALL_WEIGHTS.values())), 3
        )

# chunk size 映射（实验名 → chunk_size）
CHUNK_SIZE_MAP = {
    "baseline_hybrid_rerank": 500,
    "chunk800_hybrid_rerank": 800,
    "chunk1000_hybrid_rerank": 1000,
    "chunk1200_hybrid_rerank": 1200,
    "chunk1500_hybrid_rerank": 1500,
}


def load_all_results(latest_only: bool = False) -> Dict[str, List[dict]]:
    """加载所有结果 CSV，返回 {实验名: [rows]}。"""
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"未找到结果文件，请先运行 run_experiments.py")
        sys.exit(1)

    exp_files = defaultdict(list)
    for f in csv_files:
        # 排除统计分析自身的输出
        if "statistical_analysis" in f.name or "judge_consistency" in f.name:
            continue
        m = re.match(r"(.+)_(\d{8}_\d{6})\.csv$", f.name)
        if m:
            exp_files[m.group(1)].append(f)

    all_results = {}
    for exp_name, files in exp_files.items():
        files_to_load = [files[-1]] if latest_only else files
        rows = []
        for csv_path in files_to_load:
            with open(csv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in SCORE_KEYS:
                        val = row.get(key, "")
                        row[key] = float(val) if val and val != "None" else None
                    _recalc_overall(row)
                    rows.append(row)
        if rows:
            all_results[exp_name] = rows

    return all_results


def get_paired_scores(
    results_a: List[dict], results_b: List[dict], dim: str,
) -> Tuple[List[float], List[float]]:
    """按 question_id 配对两组分数，返回 (scores_a, scores_b)。"""
    by_qid_a = {}
    for r in results_a:
        qid = str(r.get("question_id", ""))
        val = r.get(dim)
        if val is not None:
            by_qid_a[qid] = val

    by_qid_b = {}
    for r in results_b:
        qid = str(r.get("question_id", ""))
        val = r.get(dim)
        if val is not None:
            by_qid_b[qid] = val

    common_qids = sorted(set(by_qid_a.keys()) & set(by_qid_b.keys()))
    sa = [by_qid_a[q] for q in common_qids]
    sb = [by_qid_b[q] for q in common_qids]
    return sa, sb


def wilcoxon_test(x: List[float], y: List[float]) -> Tuple[float, float]:
    """配对 Wilcoxon 签名秩检验，返回 (statistic, p_value)。"""
    from scipy.stats import wilcoxon
    diff = [a - b for a, b in zip(x, y)]
    # 过滤零差异
    nonzero = [(a, b) for a, b in zip(x, y) if a != b]
    if len(nonzero) < 5:
        return float("nan"), float("nan")
    x_nz = [a for a, _ in nonzero]
    y_nz = [b for _, b in nonzero]
    stat, p = wilcoxon(x_nz, y_nz)
    return float(stat), float(p)


def cliffs_delta(x: List[float], y: List[float]) -> Tuple[float, str]:
    """计算 Cliff's delta 效应量。"""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float("nan"), "N/A"

    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    d = (more - less) / (n_x * n_y)

    abs_d = abs(d)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return d, magnitude


def significance_label(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    parser = argparse.ArgumentParser(description="检索实验统计分析与可视化")
    parser.add_argument("--latest", action="store_true", help="每个实验只取最新一次结果")
    parser.add_argument("--best", type=str, default=None,
                        help="指定最优方案名（默认自动选 overall 最高的）")
    args = parser.parse_args()

    all_results = load_all_results(latest_only=args.latest)
    print(f"加载了 {len(all_results)} 个实验的结果")

    if not all_results:
        print("没有可分析的结果")
        sys.exit(1)

    # 确定最优方案
    def overall_avg(rows):
        vals = [r["overall"] for r in rows if r["overall"] is not None]
        return sum(vals) / len(vals) if vals else 0

    if args.best:
        best_name = args.best
    else:
        best_name = max(all_results.keys(), key=lambda n: overall_avg(all_results[n]))
    print(f"最优方案（基准）：{best_name} (overall avg = {overall_avg(all_results[best_name]):.3f})")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"statistical_analysis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══ 1. 完整对比表 ═══════════════════════════════════════
    sorted_exps = sorted(all_results.keys(), key=lambda n: overall_avg(all_results[n]), reverse=True)

    summary_rows = []
    for exp_name in sorted_exps:
        rows = all_results[exp_name]
        entry = {"experiment": exp_name, "n": len(rows)}
        for dim in SCORE_KEYS:
            vals = [r[dim] for r in rows if r[dim] is not None]
            entry[f"{dim}_mean"] = round(np.mean(vals), 3) if vals else None
            entry[f"{dim}_std"] = round(np.std(vals), 3) if vals else None
        summary_rows.append(entry)

    summary_csv = out_dir / "summary_table.csv"
    fieldnames = ["experiment", "n"]
    for dim in SCORE_KEYS:
        fieldnames.extend([f"{dim}_mean", f"{dim}_std"])
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n完整对比表：{summary_csv}")

    # ═══ 2. 配对显著性检验 ═══════════════════════════════════
    sig_rows = []
    print(f"\n{'='*85}")
    print(f"配对 Wilcoxon 检验：{best_name} vs 其他方案")
    print(f"{'='*85}")
    print(f"{'对比方案':<28} {'维度':<12} {'Δ均值':>8} {'p值':>10} {'显著性':>6} {'Cliff δ':>8} {'效应量':>10}")
    print(f"{'-'*85}")

    for exp_name in sorted_exps:
        if exp_name == best_name:
            continue
        for dim in ["overall"]:
            sa, sb = get_paired_scores(all_results[best_name], all_results[exp_name], dim)
            if len(sa) < 5:
                continue

            stat, p = wilcoxon_test(sa, sb)
            delta, magnitude = cliffs_delta(sa, sb)
            mean_diff = np.mean(sa) - np.mean(sb)
            sig = significance_label(p)

            sig_rows.append({
                "best": best_name,
                "compared": exp_name,
                "dimension": dim,
                "n_pairs": len(sa),
                "mean_diff": round(mean_diff, 4),
                "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
                "p_value": round(p, 6) if not np.isnan(p) else None,
                "significance": sig,
                "cliffs_delta": round(delta, 4),
                "effect_size": magnitude,
            })

            print(
                f"{exp_name:<28} {SCORE_NAMES_ZH.get(dim, dim):<12} "
                f"{mean_diff:>+8.4f} {p:>10.6f} {sig:>6} "
                f"{delta:>+8.4f} {magnitude:>10}"
            )

    # 对所有维度也做检验
    print(f"\n{'='*85}")
    print(f"各维度详细检验")
    print(f"{'='*85}")
    for exp_name in sorted_exps:
        if exp_name == best_name:
            continue
        for dim in SCORE_KEYS:
            if dim == "overall":
                continue
            sa, sb = get_paired_scores(all_results[best_name], all_results[exp_name], dim)
            if len(sa) < 5:
                continue
            stat, p = wilcoxon_test(sa, sb)
            delta, magnitude = cliffs_delta(sa, sb)
            mean_diff = np.mean(sa) - np.mean(sb)
            sig = significance_label(p)

            sig_rows.append({
                "best": best_name,
                "compared": exp_name,
                "dimension": dim,
                "n_pairs": len(sa),
                "mean_diff": round(mean_diff, 4),
                "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
                "p_value": round(p, 6) if not np.isnan(p) else None,
                "significance": sig,
                "cliffs_delta": round(delta, 4),
                "effect_size": magnitude,
            })

    sig_csv = out_dir / "significance_tests.csv"
    sig_fields = ["best", "compared", "dimension", "n_pairs", "mean_diff",
                  "wilcoxon_stat", "p_value", "significance", "cliffs_delta", "effect_size"]
    with open(sig_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sig_fields)
        writer.writeheader()
        writer.writerows(sig_rows)
    print(f"\n显著性检验结果：{sig_csv}")

    # ═══ 3. 可视化 ═══════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 3a. 综合分数柱状图（带误差棒和显著性标注）
        fig, ax = plt.subplots(figsize=(12, 6))
        exp_names = sorted_exps
        means = []
        stds = []
        for name in exp_names:
            vals = [r["overall"] for r in all_results[name] if r["overall"] is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        colors = ["#2ecc71" if n == best_name else "#3498db" for n in exp_names]
        bars = ax.bar(range(len(exp_names)), means, yerr=stds, capsize=4,
                      color=colors, alpha=0.8, edgecolor="white")

        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in exp_names],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("综合分数 (1-5)")
        ax.set_title("检索方案综合分数对比（含标准差误差棒）")
        ax.set_ylim(0, 5.5)
        ax.grid(axis="y", alpha=0.3)

        # 在柱状图上标注均值
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / "overall_comparison.png", dpi=150)
        plt.close()

        # 3b. Chunk size 最优曲线图
        chunk_data = []
        for exp_name, chunk_size in sorted(CHUNK_SIZE_MAP.items(), key=lambda x: x[1]):
            if exp_name in all_results:
                rows = all_results[exp_name]
                for dim in SCORE_KEYS:
                    vals = [r[dim] for r in rows if r[dim] is not None]
                    if vals:
                        chunk_data.append({
                            "chunk_size": chunk_size,
                            "dim": dim,
                            "mean": np.mean(vals),
                            "std": np.std(vals),
                        })

        if chunk_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            dim_colors = {
                "relevance": "#e74c3c", "faithfulness": "#3498db",
                "completeness": "#2ecc71", "technical_accuracy": "#f39c12",
                "overall": "#9b59b6",
            }

            for dim in SCORE_KEYS:
                pts = [d for d in chunk_data if d["dim"] == dim]
                if pts:
                    xs = [p["chunk_size"] for p in pts]
                    ys = [p["mean"] for p in pts]
                    errs = [p["std"] for p in pts]
                    ax.errorbar(xs, ys, yerr=errs, marker="o", label=SCORE_NAMES_ZH.get(dim, dim),
                                color=dim_colors.get(dim, "gray"), capsize=3, linewidth=2)

            ax.set_xlabel("Chunk Size (字符)")
            ax.set_ylabel("平均分 (1-5)")
            ax.set_title("Chunk Size 对检索质量的影响\n(固定 Hybrid + Reranker)")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            ax.set_ylim(2.5, 5.2)
            fig.tight_layout()
            fig.savefig(out_dir / "chunk_size_curve.png", dpi=150)
            plt.close()

        # 3c. 按题型雷达图（展示 top-3 方案）
        all_types = set()
        for rows in all_results.values():
            for r in rows:
                all_types.add(r.get("question_type", "unknown"))
        all_types = sorted(all_types)

        if len(all_types) >= 3:
            top_3 = sorted_exps[:3]
            fig, axes = plt.subplots(1, len(top_3), figsize=(6 * len(top_3), 6),
                                     subplot_kw=dict(polar=True))
            if len(top_3) == 1:
                axes = [axes]

            radar_colors = ["#e74c3c", "#3498db", "#2ecc71"]
            for idx, (exp_name, ax_r, color) in enumerate(zip(top_3, axes, radar_colors)):
                rows = all_results[exp_name]
                type_scores = {}
                for qtype in all_types:
                    typed_rows = [r for r in rows if r.get("question_type") == qtype]
                    vals = [r["overall"] for r in typed_rows if r["overall"] is not None]
                    type_scores[qtype] = np.mean(vals) if vals else 0

                labels = list(type_scores.keys())
                values = list(type_scores.values())
                values.append(values[0])  # 闭合

                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                angles.append(angles[0])

                ax_r.plot(angles, values, "o-", color=color, linewidth=2)
                ax_r.fill(angles, values, alpha=0.2, color=color)
                ax_r.set_xticks(angles[:-1])
                ax_r.set_xticklabels(labels, fontsize=9)
                ax_r.set_ylim(0, 5)
                ax_r.set_title(exp_name.replace("_", "\n"), fontsize=10, pad=20)

            fig.suptitle("Top-3 方案按题型综合分数雷达图", fontsize=14, y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / "radar_by_type.png", dpi=150, bbox_inches="tight")
            plt.close()

        print(f"\n可视化图表已保存到：{out_dir}/")

    except ImportError as e:
        print(f"\n[提示] 缺少可视化依赖 ({e})，跳过图表生成")
        print("安装：pip install matplotlib scipy")

    # ═══ 4. 打印按题型交叉分析 ═══════════════════════════════
    print(f"\n{'='*85}")
    print("按题型交叉分析（overall 平均分）")
    print(f"{'='*85}")

    all_types = set()
    for rows in all_results.values():
        for r in rows:
            all_types.add(r.get("question_type", "unknown"))
    all_types = sorted(all_types)

    header = f"{'实验名称':<28}"
    for t in all_types:
        header += f" {t:>12}"
    print(header)
    print("-" * 85)

    for exp_name in sorted_exps:
        line = f"{exp_name:<28}"
        for qtype in all_types:
            typed_rows = [r for r in all_results[exp_name] if r.get("question_type") == qtype]
            vals = [r["overall"] for r in typed_rows if r["overall"] is not None]
            avg = f"{np.mean(vals):.2f}" if vals else "N/A"
            line += f" {avg:>12}"
        print(line)

    print(f"{'='*85}")
    print(f"\n所有结果已保存到：{out_dir}/")


if __name__ == "__main__":
    main()
