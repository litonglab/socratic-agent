"""
汇总对比所有实验结果

用法：
  python eval/retrieval/compare_experiments.py              # 汇总所有结果
  python eval/retrieval/compare_experiments.py --latest     # 每个实验只取最新一次

读取 eval/retrieval/results/ 下的所有 CSV 文件，按实验名称汇总各维度平均分。
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SCORE_KEYS = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
DISPLAY_NAMES = {"relevance": "相关性", "faithfulness": "忠实性", "completeness": "完整性",
                 "technical_accuracy": "技术准确", "overall": "综合"}

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


def load_all_results(latest_only: bool = False) -> dict:
    """加载所有结果 CSV，返回 {实验名: [rows]}。"""
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"未找到结果文件，请先运行 run_experiments.py")
        print(f"搜索目录：{RESULTS_DIR}")
        sys.exit(1)

    # 按实验名分组，如果 latest_only 则只保留每个实验的最新文件
    exp_files = defaultdict(list)
    for f in csv_files:
        # 文件名格式：{exp_name}_{timestamp}.csv
        # 提取实验名（去掉最后的 _YYYYMMDD_HHMMSS）
        m = re.match(r"(.+)_(\d{8}_\d{6})\.csv$", f.name)
        if m:
            exp_name = m.group(1)
            exp_files[exp_name].append(f)
        else:
            # 旧格式文件（如 results_20260302_183746.csv），作为单独实验
            exp_files[f.stem].append(f)

    all_results = {}
    for exp_name, files in exp_files.items():
        files_to_load = [files[-1]] if latest_only else files  # 已排序，最后一个是最新的

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


def print_comparison(all_results: dict):
    """打印对比表。"""
    if not all_results:
        print("没有可对比的结果")
        return

    # 按综合分排序
    def overall_avg(rows):
        vals = [r["overall"] for r in rows if r["overall"] is not None]
        return sum(vals) / len(vals) if vals else 0

    sorted_exps = sorted(all_results.keys(), key=lambda n: overall_avg(all_results[n]), reverse=True)

    print(f"\n{'='*80}")
    print("检索实验方案对比汇总（各维度平均分）")
    print(f"{'='*80}")
    print(f"{'实验名称':<30} {'题数':>4} {'相关性':>6} {'忠实性':>6} {'完整性':>6} {'技术准确':>8} {'综合':>6}")
    print(f"{'-'*80}")

    for exp_name in sorted_exps:
        rows = all_results[exp_name]
        n = len(rows)

        def avg(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return f"{sum(vals)/len(vals):.2f}" if vals else "N/A"

        print(
            f"{exp_name:<30} {n:>4} {avg('relevance'):>6} {avg('faithfulness'):>6} "
            f"{avg('completeness'):>6} {avg('technical_accuracy'):>8} {avg('overall'):>6}"
        )

    print(f"{'='*80}")

    # 按问题类型细分
    print(f"\n{'='*80}")
    print("按问题类型细分")
    print(f"{'='*80}")

    # 收集所有问题类型
    all_types = set()
    for rows in all_results.values():
        for r in rows:
            all_types.add(r.get("question_type", "unknown"))

    for qtype in sorted(all_types):
        print(f"\n── 类型：{qtype} ──")
        print(f"{'实验名称':<30} {'题数':>4} {'相关性':>6} {'忠实性':>6} {'完整性':>6} {'技术准确':>8} {'综合':>6}")
        print(f"{'-'*80}")

        for exp_name in sorted_exps:
            rows = [r for r in all_results[exp_name]
                    if r.get("question_type", "unknown") == qtype]
            if not rows:
                continue
            n = len(rows)

            def avg(key):
                vals = [r[key] for r in rows if r[key] is not None]
                return f"{sum(vals)/len(vals):.2f}" if vals else "N/A"

            print(
                f"{exp_name:<30} {n:>4} {avg('relevance'):>6} {avg('faithfulness'):>6} "
                f"{avg('completeness'):>6} {avg('technical_accuracy'):>8} {avg('overall'):>6}"
            )

    print(f"\n{'='*80}")
    print(f"数据来源：{RESULTS_DIR}/")


def main():
    latest_only = "--latest" in sys.argv
    if latest_only:
        print("模式：每个实验只取最新一次结果")

    all_results = load_all_results(latest_only)
    print(f"加载了 {len(all_results)} 个实验的结果")

    print_comparison(all_results)


if __name__ == "__main__":
    main()
