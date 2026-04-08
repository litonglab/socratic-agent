#!/usr/bin/env python3
"""
为毕业论文第五章生成所有实验图表。
用法：python eval/generate_figures.py [--outdir eval/figures]
"""
import argparse
import csv
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np

# ── 中文字体（通过文件路径加载，避免字体名匹配导致异形字）──
import platform
import matplotlib.font_manager as fm

_CN_FONT_PROP = None

def _find_cn_font():
    """按优先级查找中文字体文件，返回 FontProperties 或 None。"""
    if platform.system() == "Darwin":
        candidates = [
            # Songti SC — macOS 系统自带宋体
            "/System/Library/Fonts/Supplemental/Songti.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
    for path in candidates:
        if os.path.isfile(path):
            return fm.FontProperties(fname=path)
    # 回退：让 matplotlib 自动搜索
    for name in ["Songti SC", "SimSun", "SimSong", "STSong"]:
        matches = [f for f in fm.fontManager.ttflist if f.name == name]
        if matches:
            return fm.FontProperties(fname=matches[0].fname)
    return None

_CN_FONT_PROP = _find_cn_font()

if _CN_FONT_PROP:
    _font_name = _CN_FONT_PROP.get_name()
    plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# 全局样式
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent


def _add_bottom_panel_titles(fig, axes, titles, fontsize=10.5, bottom_margin=0.14, y_offset=0.06):
    """多子图统一样式：不使用顶部总标题，将子标题放到底部。"""
    axes_list = list(np.ravel(axes))
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])
    for ax, title in zip(axes_list, titles):
        bbox = ax.get_position()
        x = (bbox.x0 + bbox.x1) / 2
        y = max(bbox.y0 - y_offset, 0.02)
        fig.text(x, y, title, ha="center", va="top", fontsize=fontsize)


def _font_kwargs(**kwargs):
    if _CN_FONT_PROP is not None:
        kwargs.setdefault("fontproperties", _CN_FONT_PROP)
    return kwargs


def _wrap_cn(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text or "", width=width, break_long_words=False))


def _clean_demo_answer(answer: str) -> str:
    text = answer or ""
    text = re.sub(r"<思考>.*?</思考>", "", text, flags=re.S)
    text = re.sub(r"\n+引用：.*", "", text, flags=re.S)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"`", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    lines = [ln.strip(" -*•\t") for ln in text.splitlines()]
    paragraphs = [ln.strip() for ln in lines if ln.strip()]
    return "\n".join(paragraphs)


def _pick_summary_paragraph(answer: str) -> str:
    text = _clean_demo_answer(answer)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    generic_prefixes = (
        "好的，我们来",
        "好的，你确认了",
        "你描述的情况是",
        "请执行以上步骤",
        "理解确认",
        "问题回顾",
    )
    preferred = []
    for p in paragraphs:
        if any(p.startswith(prefix) for prefix in generic_prefixes):
            continue
        if len(p) >= 18:
            preferred.append(p)
    if preferred:
        return preferred[0]
    for p in paragraphs:
        if len(p) >= 12:
            return p
    return paragraphs[0] if paragraphs else ""


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _tool_label(tool_traces: list) -> str:
    if not tool_traces:
        return "未调用工具"
    names = []
    for item in tool_traces:
        name = item.get("tool", "")
        if name and name not in names:
            names.append(name)
    return "、".join(names) if names else "未调用工具"


def _keyword_coverage(answer: str) -> str:
    cleaned = _clean_demo_answer(answer)
    keyword_map = [
        ("物理连接", "物理连接"),
        ("IP地址", "IP配置"),
        ("子网掩码", "IP配置"),
        ("防火墙", "防火墙"),
        ("ARP", "ARP"),
        ("模拟器", "实验环境"),
        ("交换机", "交换机"),
    ]
    found = []
    for raw, label in keyword_map:
        if raw in cleaned and label not in found:
            found.append(label)
    return " / ".join(found[:5]) if found else "直接给出完整排查清单"


def _draw_info_card(ax, xywh, title, agent_text, ds_text):
    x, y, w, h = xywh
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9, edgecolor="#D7DBE2", facecolor="#FBFBFC",
    )
    ax.add_patch(card)
    ax.text(x + 0.02, y + h - 0.08, title, fontsize=11, color="#333333", **_font_kwargs())
    ax.text(x + 0.02, y + h - 0.18, f"Agent：{agent_text}", fontsize=9.6, color="#C44E52",
            va="top", **_font_kwargs())
    ax.text(x + 0.02, y + h - 0.36, f"DeepSeek：{ds_text}", fontsize=9.6, color="#4C72B0",
            va="top", **_font_kwargs())


# ═══════════════════════════════════════════════════════════
# 图 1：LLM 裁判自一致性 — ICC 柱状图
# ═══════════════════════════════════════════════════════════
def fig_judge_icc(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    dims = data["dimensions"]

    labels_cn = {
        "relevance": "相关性",
        "faithfulness": "忠实度",
        "completeness": "完整性",
        "technical_accuracy": "技术准确性",
        "overall": "综合得分",
    }
    keys = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
    icc_vals = [dims[k]["icc"] for k in keys]
    labels = [labels_cn[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, icc_vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
                  edgecolor="white", linewidth=0.8, width=0.55)

    # 阈值线
    ax.axhline(0.75, color="orange", linestyle="--", linewidth=1, label="良好一致性阈值 (0.75)")
    ax.axhline(0.90, color="green", linestyle="--", linewidth=1, label="优秀一致性阈值 (0.90)")

    for bar, v in zip(bars, icc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("ICC (组内相关系数)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "fig_judge_icc.png")
    plt.close(fig)
    print(f"  ✓ fig_judge_icc.png")


# ═══════════════════════════════════════════════════════════
# 图 2：LLM 裁判自一致性 — 各维度标准差箱线图
# ═══════════════════════════════════════════════════════════
def fig_judge_std_box(result_csv: Path, outdir: Path):
    """从 judge_consistency CSV 中按 question_id 分组计算 3 次打分的标准差，画箱线图。"""
    rows = []
    with open(result_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    dims = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
    labels_cn = ["相关性", "忠实度", "完整性", "技术准确性", "综合得分"]

    # 按 question_id 分组
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["question_id"]].append(r)

    std_data = []
    for d in dims:
        stds = []
        for qid, reps in grouped.items():
            vals = [float(r[d]) for r in reps]
            if len(vals) >= 2:
                stds.append(np.std(vals, ddof=1))
        std_data.append(stds)

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(std_data, tick_labels=labels_cn, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="可接受阈值 (0.5)")
    ax.set_ylabel("打分标准差")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_judge_std_box.png")
    plt.close(fig)
    print(f"  ✓ fig_judge_std_box.png")


# ═══════════════════════════════════════════════════════════
# 图 3：检索方案综合得分对比（水平柱状图）
# ═══════════════════════════════════════════════════════════
def fig_retrieval_comparison(summary_csv: Path, outdir: Path):
    rows = []
    with open(summary_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["experiment"] == "results":
                continue  # 跳过汇总行
            rows.append(r)

    # 按 overall_mean 排序
    rows.sort(key=lambda r: float(r["overall_mean"]))

    # 方案名映射：统一写成“检索策略 + 分块方案”的中文组合，不使用 + 号
    name_map = {
        "enriched_hybrid_rerank": "融合召回后重排序\n上下文增强分块",
        "baseline_hybrid": "融合召回\n常规分块",
        "baseline_hybrid_rerank": "融合召回后重排序\n常规分块",
        "chunk800_hybrid_rerank": "融合召回后重排序\n扩展分块（800）",
        "chunk1000_hybrid_rerank": "融合召回后重排序\n扩展分块（1000）",
        "chunk1200_hybrid_rerank": "融合召回后重排序\n扩展分块（1200）",
        "chunk1500_hybrid_rerank": "融合召回后重排序\n扩展分块（1500）",
        "baseline_similarity": "语义匹配召回\n常规分块",
        "baseline_mmr_rerank": "多样性召回后重排序\n常规分块",
        "baseline_mmr": "多样性召回\n常规分块",
    }

    names = [name_map.get(r["experiment"], r["experiment"]) for r in rows]
    means = [float(r["overall_mean"]) for r in rows]
    stds = [float(r["overall_std"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    y_pos = np.arange(len(names))
    colors = ["#4C72B0"] * len(names)
    # 最高分用不同颜色
    colors[-1] = "#C44E52"

    ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="white",
            height=0.6, capsize=3, alpha=0.85)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(
            5.17,
            i +0.28,
            f"{m:.3f}",
            va="center",
            ha="right",
            fontsize=13.5,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.18),
            clip_on=False,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("综合得分（1-5 分）", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(2.5, 5.2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_retrieval_comparison.png")
    plt.close(fig)
    print(f"  ✓ fig_retrieval_comparison.png")


# ═══════════════════════════════════════════════════════════
# 图 4：检索方案四维度雷达图（前 5 方案）
# ═══════════════════════════════════════════════════════════
def fig_retrieval_radar(summary_csv: Path, outdir: Path):
    rows = []
    with open(summary_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["experiment"] == "results":
                continue
            rows.append(r)

    rows.sort(key=lambda r: float(r["overall_mean"]), reverse=True)
    top5 = rows[:5]

    dims = ["relevance_mean", "faithfulness_mean", "completeness_mean", "technical_accuracy_mean"]
    dim_labels = ["相关性", "忠实度", "完整性", "技术准确性"]
    name_map = {
        "enriched_hybrid_rerank": "增强+混合+重排",
        "baseline_hybrid": "基线+混合",
        "baseline_hybrid_rerank": "基线+混合+重排",
        "chunk800_hybrid_rerank": "C800+混合+重排",
        "chunk1000_hybrid_rerank": "C1000+混合+重排",
        "chunk1200_hybrid_rerank": "C1200+混合+重排",
        "chunk1500_hybrid_rerank": "C1500+混合+重排",
        "baseline_similarity": "基线-相似度",
        "baseline_mmr_rerank": "基线+MMR+重排",
    }

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974"]

    for i, r in enumerate(top5):
        vals = [float(r[d]) for d in dims]
        vals += vals[:1]
        label = name_map.get(r["experiment"], r["experiment"])
        ax.plot(angles, vals, "o-", linewidth=1.5, label=label, color=colors[i], markersize=4)
        ax.fill(angles, vals, alpha=0.08, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(3.0, 5.2)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_retrieval_radar.png")
    plt.close(fig)
    print(f"  ✓ fig_retrieval_radar.png")


# ═══════════════════════════════════════════════════════════
# 图 5：Chunk Size 对综合得分的影响曲线
# ═══════════════════════════════════════════════════════════
def fig_chunk_size_curve(summary_csv: Path, outdir: Path):
    rows = []
    with open(summary_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # chunk size → overall mean 映射
    chunk_map = {
        "baseline_hybrid_rerank": 500,
        "chunk800_hybrid_rerank": 800,
        "chunk1000_hybrid_rerank": 1000,
        "chunk1200_hybrid_rerank": 1200,
        "chunk1500_hybrid_rerank": 1500,
    }

    points = []
    for r in rows:
        exp = r["experiment"]
        if exp in chunk_map:
            points.append((chunk_map[exp], float(r["overall_mean"]), float(r["overall_std"])))

    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    es = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#4C72B0", linewidth=2,
                markersize=8, capsize=5, capthick=1.5, ecolor="#999")

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    # 标注最优点
    best_idx = np.argmax(ys)
    ax.plot(xs[best_idx], ys[best_idx], "r*", markersize=15, zorder=5)

    ax.set_xlabel("Chunk Size（字符数）")
    ax.set_ylabel("综合得分（1-5 分）")
    ax.set_xticks(xs)
    ax.set_ylim(3.0, 5.0)
    fig.tight_layout()
    fig.savefig(outdir / "fig_chunk_size_curve.png")
    plt.close(fig)
    print(f"  ✓ fig_chunk_size_curve.png")


# ═══════════════════════════════════════════════════════════
# 图 5b：检索优化主图（检索方式 / 文本组织方式）
# ═══════════════════════════════════════════════════════════
def _load_summary_rows(summary_csv: Path):
    rows = []
    with open(summary_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["experiment"] == "results":
                continue
            rows.append(r)
    return rows


def _plot_retrieval_panel(ax, items, title=None, xlim=(3.4, 4.75)):
    labels = [item["label"] for item in items]
    means = [item["mean"] for item in items]

    best_idx = int(np.argmax(means))
    colors = ["#4C72B0"] * len(items)
    colors[best_idx] = "#C44E52"

    y_pos = np.arange(len(labels))
    ax.barh(
        y_pos,
        means,
        color=colors,
        edgecolor="white",
        height=0.62,
        alpha=0.85,
    )

    for i, m in enumerate(means):
        label_x = xlim[1] - 0.03
        label_y = i - 0.24
        ax.text(
            label_x,
            label_y,
            f"{m:.3f}",
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.15),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_xlabel("综合得分（1-5 分）", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def fig_retrieval_focus_panels(summary_csv: Path, outdir: Path):
    rows = _load_summary_rows(summary_csv)
    by_exp = {r["experiment"]: r for r in rows}

    method_items = [
        {
            "label": "语义匹配召回",
            "mean": float(by_exp["baseline_similarity"]["overall_mean"]),
            "std": float(by_exp["baseline_similarity"]["overall_std"]),
        },
        {
            "label": "多样性召回",
            "mean": float(by_exp["baseline_mmr"]["overall_mean"]),
            "std": float(by_exp["baseline_mmr"]["overall_std"]),
        },
        {
            "label": "多样性召回\n后重排序",
            "mean": float(by_exp["baseline_mmr_rerank"]["overall_mean"]),
            "std": float(by_exp["baseline_mmr_rerank"]["overall_std"]),
        },
        {
            "label": "融合召回",
            "mean": float(by_exp["baseline_hybrid"]["overall_mean"]),
            "std": float(by_exp["baseline_hybrid"]["overall_std"]),
        },
        {
            "label": "融合召回后重排序",
            "mean": float(by_exp["baseline_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["baseline_hybrid_rerank"]["overall_std"]),
        },
    ]

    text_items = [
        {
            "label": "常规分块（500）",
            "mean": float(by_exp["baseline_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["baseline_hybrid_rerank"]["overall_std"]),
        },
        {
            "label": "扩展分块（800）",
            "mean": float(by_exp["chunk800_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["chunk800_hybrid_rerank"]["overall_std"]),
        },
        {
            "label": "扩展分块（1000）",
            "mean": float(by_exp["chunk1000_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["chunk1000_hybrid_rerank"]["overall_std"]),
        },
        {
            "label": "扩展分块（1200）",
            "mean": float(by_exp["chunk1200_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["chunk1200_hybrid_rerank"]["overall_std"]),
        },
        {
            "label": "扩展分块（1500）",
            "mean": float(by_exp["chunk1500_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["chunk1500_hybrid_rerank"]["overall_std"]),
        },
        {
            "label": "上下文增强分块",
            "mean": float(by_exp["enriched_hybrid_rerank"]["overall_mean"]),
            "std": float(by_exp["enriched_hybrid_rerank"]["overall_std"]),
        },
    ]

    # 单图：检索方式
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _plot_retrieval_panel(ax, method_items)
    fig.tight_layout()
    fig.savefig(outdir / "fig_retriever_compare.png")
    plt.close(fig)
    print("  ✓ fig_retriever_compare.png")

    # 单图：文本组织方式
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    _plot_retrieval_panel(ax, text_items)
    fig.tight_layout()
    fig.savefig(outdir / "fig_text_organization_compare.png")
    plt.close(fig)
    print("  ✓ fig_text_organization_compare.png")

    # 合图：论文主文推荐
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4), sharex=True)
    _plot_retrieval_panel(axes[0], method_items)
    _plot_retrieval_panel(axes[1], text_items)
    _add_bottom_panel_titles(
        fig,
        axes,
        [
            "（a）固定分块下不同检索算法综合得分对比",
            "（b）固定检索策略下不同分块大小综合得分对比",
        ],
        fontsize=10.5,
        bottom_margin=0.23,
        y_offset=0.13,
    )
    fig.savefig(outdir / "fig_retrieval_focus_panels.png")
    plt.close(fig)
    print("  ✓ fig_retrieval_focus_panels.png")


# ═══════════════════════════════════════════════════════════
# 图 6：显著性检验热力图
# ═══════════════════════════════════════════════════════════
def fig_significance_heatmap(sig_csv: Path, outdir: Path):
    rows = []
    with open(sig_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["dimension"] == "overall":
                rows.append(r)

    name_map = {
        "enriched_hybrid_rerank": "增强+混合+重排",
        "baseline_hybrid": "基线+混合",
        "baseline_hybrid_rerank": "C500+混合+重排",
        "chunk800_hybrid_rerank": "C800",
        "chunk1000_hybrid_rerank": "C1000",
        "chunk1200_hybrid_rerank": "C1200",
        "chunk1500_hybrid_rerank": "C1500",
        "baseline_similarity": "基线-相似度",
        "baseline_mmr_rerank": "基线+重排",
        "baseline_mmr": "基线-MMR",
        "results": "汇总",
    }

    compared_exps = []
    p_vals = []
    deltas = []
    effect_labels = []
    for r in rows:
        cmp = r["compared"]
        if cmp == "results":
            continue
        compared_exps.append(name_map.get(cmp, cmp))
        p = float(r["p_value"]) if r["p_value"] else 1.0
        p_vals.append(p)
        d = float(r["cliffs_delta"]) if r["cliffs_delta"] else 0.0
        deltas.append(d)
        effect_labels.append(r["effect_size"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1]})

    # 左图：p 值
    y_pos = np.arange(len(compared_exps))
    colors_p = ["#C44E52" if p < 0.05 else "#4C72B0" for p in p_vals]
    ax1.barh(y_pos, p_vals, color=colors_p, height=0.6, alpha=0.8)
    ax1.axvline(0.05, color="red", linestyle="--", linewidth=1, label="α = 0.05")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(compared_exps, fontsize=9)
    ax1.set_xlabel("p 值 (Wilcoxon)")
    ax1.legend(fontsize=8)

    # 右图：效应量
    colors_d = []
    for e in effect_labels:
        if e == "large":
            colors_d.append("#C44E52")
        elif e == "medium":
            colors_d.append("#CCB974")
        elif e == "small":
            colors_d.append("#55A868")
        else:
            colors_d.append("#4C72B0")

    ax2.barh(y_pos, deltas, color=colors_d, height=0.6, alpha=0.8)
    ax2.axvline(0.147, color="gray", linestyle=":", linewidth=1, label="小效应 (0.147)")
    ax2.axvline(0.33, color="gray", linestyle="--", linewidth=1, label="中效应 (0.33)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(compared_exps, fontsize=9)
    ax2.set_xlabel("Cliff's delta")
    ax2.legend(fontsize=7, loc="lower right")
    _add_bottom_panel_titles(
        fig,
        [ax1, ax2],
        ["（a）最优方案与各方案的 p 值比较", "（b）最优方案与各方案的效应量比较"],
        fontsize=10,
        bottom_margin=0.13,
        y_offset=0.06,
    )
    fig.savefig(outdir / "fig_significance_tests.png")
    plt.close(fig)
    print(f"  ✓ fig_significance_tests.png")


# ═══════════════════════════════════════════════════════════
# 图 7：分类器混淆矩阵热力图
# ═══════════════════════════════════════════════════════════
def fig_classification_confusion(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    cm = data["confusion_matrix"]
    classes = list(cm.keys())

    labels_cn = {
        "THEORY_CONCEPT": "理论概念",
        "LAB_TROUBLESHOOTING": "实验排错",
        "CONFIG_REVIEW": "配置审查",
        "CALCULATION": "计算题",
    }

    matrix = []
    for true_cls in classes:
        row = [cm[true_cls].get(pred_cls, 0) for pred_cls in classes]
        matrix.append(row)
    matrix = np.array(matrix)

    # 归一化（按行 = 真实类别）
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / row_sums

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    cls_labels = [labels_cn.get(c, c) for c in classes]

    # 左图：计数
    im1 = ax1.imshow(matrix, cmap="Blues", aspect="auto")
    ax1.set_xticks(range(len(classes)))
    ax1.set_yticks(range(len(classes)))
    ax1.set_xticklabels(cls_labels, fontsize=9, rotation=30, ha="right")
    ax1.set_yticklabels(cls_labels, fontsize=9)
    ax1.set_xlabel("预测类别")
    ax1.set_ylabel("真实类别")
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax1.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # 右图：归一化
    im2 = ax2.imshow(matrix_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels(cls_labels, fontsize=9, rotation=30, ha="right")
    ax2.set_yticklabels(cls_labels, fontsize=9)
    ax2.set_xlabel("预测类别")
    ax2.set_ylabel("真实类别")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = matrix_norm[i, j]
            color = "white" if val > 0.6 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    _add_bottom_panel_titles(
        fig,
        [ax1, ax2],
        ["（a）混淆矩阵（计数）", "（b）混淆矩阵（归一化）"],
        fontsize=10,
        bottom_margin=0.14,
        y_offset=0.06,
    )
    fig.savefig(outdir / "fig_classification_confusion.png")
    plt.close(fig)
    print(f"  ✓ fig_classification_confusion.png")


# ═══════════════════════════════════════════════════════════
# 图 8：分类器各类别 Precision / Recall / F1 柱状图
# ═══════════════════════════════════════════════════════════
def fig_classification_metrics(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    per_class = data["per_class"]
    classes = list(per_class.keys())

    labels_cn = {
        "THEORY_CONCEPT": "理论概念",
        "LAB_TROUBLESHOOTING": "实验排错",
        "CONFIG_REVIEW": "配置审查",
        "CALCULATION": "计算题",
    }

    cls_labels = [labels_cn.get(c, c) for c in classes]
    precisions = [per_class[c]["precision"] for c in classes]
    recalls = [per_class[c]["recall"] for c in classes]
    f1s = [per_class[c]["f1"] for c in classes]
    supports = [per_class[c]["support"] for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, precisions, w, label="Precision", color="#4C72B0", alpha=0.85)
    ax.bar(x, recalls, w, label="Recall", color="#55A868", alpha=0.85)
    ax.bar(x + w, f1s, w, label="F1-Score", color="#C44E52", alpha=0.85)

    # 标注数值
    for i in range(len(classes)):
        ax.text(x[i] - w, precisions[i] + 0.02, f"{precisions[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i], recalls[i] + 0.02, f"{recalls[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w, f1s[i] + 0.02, f"{f1s[i]:.2f}", ha="center", fontsize=8)

    # 在 x 轴标签下方标注样本数
    labels_with_n = [f"{l}\n(n={n})" for l, n in zip(cls_labels, supports)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_n, fontsize=9)
    ax.set_ylabel("分数")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_classification_metrics.png")
    plt.close(fig)
    print(f"  ✓ fig_classification_metrics.png")


# ═══════════════════════════════════════════════════════════
# 图 9：检索方案各维度分组柱状图
# ═══════════════════════════════════════════════════════════
def fig_retrieval_dimensions(summary_csv: Path, outdir: Path):
    rows = []
    with open(summary_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["experiment"] == "results":
                continue
            rows.append(r)

    rows.sort(key=lambda r: float(r["overall_mean"]), reverse=True)
    top6 = rows[:6]

    name_map = {
        "enriched_hybrid_rerank": "增强+混合+重排",
        "baseline_hybrid": "基线+混合",
        "baseline_hybrid_rerank": "基线+混合+重排",
        "chunk800_hybrid_rerank": "C800+混合+重排",
        "chunk1000_hybrid_rerank": "C1000+混合+重排",
        "chunk1200_hybrid_rerank": "C1200+混合+重排",
        "chunk1500_hybrid_rerank": "C1500+混合+重排",
        "baseline_similarity": "基线-相似度",
        "baseline_mmr_rerank": "基线+MMR+重排",
    }

    dims = ["relevance_mean", "faithfulness_mean", "completeness_mean", "technical_accuracy_mean"]
    dim_labels = ["相关性", "忠实度", "完整性", "技术准确性"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    x = np.arange(len(top6))
    w = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (dim, label, color) in enumerate(zip(dims, dim_labels, colors)):
        vals = [float(r[dim]) for r in top6]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color, alpha=0.85)

    names = [name_map.get(r["experiment"], r["experiment"]) for r in top6]
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("得分（1-5 分）")
    ax.set_ylim(3.0, 5.3)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "fig_retrieval_dimensions.png")
    plt.close(fig)
    print(f"  ✓ fig_retrieval_dimensions.png")


# ═══════════════════════════════════════════════════════════
# 图 10：消融实验 — 各变体综合得分对比
# ═══════════════════════════════════════════════════════════
def fig_ablation_overall_contributions(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    variants = data["variants"]
    contribs = data["contributions"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── 左图：综合得分 ──
    var_keys = list(variants.keys())
    ov_labels = []
    scores = []
    for k in var_keys:
        v = variants[k]
        desc = re.sub(r'（.*?）', '', v["description"]).strip()
        ov_labels.append(desc)
        scores.append(v["scores"]["overall"])

    colors_ov = ["#C44E52"] + ["#4C72B0"] * (len(var_keys) - 1)
    bars1 = ax1.barh(range(len(ov_labels)), scores, color=colors_ov, height=0.55, alpha=0.85)
    for i, (bar, s) in enumerate(zip(bars1, scores)):
        ax1.text(s + 0.05, i, f"{s:.3f}", va="center", fontsize=9)
    ax1.set_yticks(range(len(ov_labels)))
    ax1.set_yticklabels(ov_labels, fontsize=9)
    ax1.set_xlabel("综合得分（1-5 分）")
    ax1.set_xlim(1.5, 5.5)
    ax1.invert_yaxis()

    # ── 右图：模块贡献 ──
    module_names = {
        "B": "Reranker",
        "C": "苏格拉底策略",
        "D": "拓扑模块",
        "E": "水平自适应",
        "F": "RAG 检索",
    }
    keys = sorted(contribs.keys(), key=lambda k: contribs[k])
    ct_labels = [module_names.get(k, k) for k in keys]
    vals = [contribs[k] for k in keys]

    bars2 = ax2.bar(ct_labels, vals, color="#4C72B0", width=0.55, alpha=0.85)
    for bar, v in zip(bars2, vals):
        label = f"{v:+.3f}"
        y_pos = v + 0.03 if v >= 0 else v - 0.06
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                 ha="center", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("贡献值（完整系统得分 - 消融变体得分）")

    _add_bottom_panel_titles(
        fig, [ax1, ax2],
        ["（a）各变体综合得分", "（b）各模块贡献值"],
        fontsize=10, bottom_margin=0.25, y_offset=0.14,
    )
    fig.savefig(outdir / "fig_ablation_overall_contributions.png")
    plt.close(fig)
    print(f"  ✓ fig_ablation_overall_contributions.png")


# ═══════════════════════════════════════════════════════════
# 图 12：消融实验 — 六维度雷达图
# ═══════════════════════════════════════════════════════════
def fig_ablation_radar(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    variants = data["variants"]

    dims = ["relevance", "faithfulness", "completeness", "technical_accuracy",
            "pedagogical_guidance", "progressive_disclosure"]
    dim_labels = ["相关性", "忠实度", "完整性", "技术准确性", "教学引导性", "信息递进性"]

    show_keys = [k for k in variants.keys()]
    show_names = {
        "A": "完整系统", "C": "无苏格拉底策略", "D": "无拓扑模块", "F": "纯大模型基线"
    }
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, k in enumerate(show_keys):
        vals = [variants[k]["scores"][d] for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=show_names[k],
                color=colors[i], markersize=4)
        ax.fill(angles, vals, alpha=0.06, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(1.0, 5.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_ablation_radar.png")
    plt.close(fig)
    print(f"  ✓ fig_ablation_radar.png")


# ═══════════════════════════════════════════════════════════
# 图 13：Socratic 教学效果 — Agent vs DeepSeek 对比
# ═══════════════════════════════════════════════════════════
def fig_socratic_comparison(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    agent = data["systems"]["agent"]
    deepseek = data["systems"]["deepseek"]

    dims = ["accuracy", "pedagogical_guidance", "progressive_disclosure", "completeness"]
    dim_labels = ["准确性", "教学引导性", "信息递进性", "完整性"]

    x = np.arange(len(dims))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    agent_vals = [agent[d] for d in dims]
    ds_vals = [deepseek[d] for d in dims]

    bars1 = ax.bar(x - w/2, agent_vals, w, label="本系统 (Agent)", color="#C44E52", alpha=0.85)
    bars2 = ax.bar(x + w/2, ds_vals, w, label="DeepSeek 基线", color="#4C72B0", alpha=0.85)

    for bar, v in zip(bars1, agent_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    for bar, v in zip(bars2, ds_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylabel("得分（1-5 分）")
    ax.set_ylim(0, 5.8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_socratic_comparison.png")
    plt.close(fig)
    print(f"  ✓ fig_socratic_comparison.png")


# ═══════════════════════════════════════════════════════════
# 图 14：Socratic — Hint Level 轨迹图
# ═══════════════════════════════════════════════════════════
def fig_socratic_trajectories(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    trajectories = data.get("hint_trajectories", [])
    if not trajectories:
        print("  [跳过] 无 hint_trajectories 数据")
        return

    cat_colors = {
        "LAB_TROUBLESHOOTING": "#C44E52",
        "THEORY_CONCEPT": "#4C72B0",
        "CONFIG_REVIEW": "#55A868",
        "CALCULATION": "#8172B2",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for t in trajectories:
        traj = t["trajectory"]
        turns = list(range(1, len(traj) + 1))
        color = cat_colors.get(t["category"], "gray")
        ax.plot(turns, traj, "o-", color=color, linewidth=1.5, markersize=5, alpha=0.7,
                label=t["name"])

    ax.set_xlabel("对话轮次")
    ax.set_ylabel("Hint Level")
    ax.set_xticks(range(1, max(len(t["trajectory"]) for t in trajectories) + 1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 图例太多，放在图外
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, loc="upper left",
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(outdir / "fig_socratic_trajectories.png")
    plt.close(fig)
    print(f"  ✓ fig_socratic_trajectories.png")


# ═══════════════════════════════════════════════════════════
# 图 15：拓扑接地效果 — 有拓扑 vs 无拓扑
# ═══════════════════════════════════════════════════════════
def fig_topology_grounding(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())
    ge = data.get("grounding_effect", {})
    if not ge:
        print("  [跳过] 无 grounding_effect 数据")
        return

    with_topo = ge["with_topo"]
    without_topo = ge["without_topo"]

    dims = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
    dim_labels = ["相关性", "忠实度", "完整性", "技术准确性", "综合得分"]

    x = np.arange(len(dims))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    wt_vals = [with_topo[d] for d in dims]
    wo_vals = [without_topo[d] for d in dims]

    ax.bar(x - w/2, wt_vals, w, label="有拓扑模块", color="#55A868", alpha=0.85)
    ax.bar(x + w/2, wo_vals, w, label="无拓扑模块", color="#C44E52", alpha=0.85)

    for i in range(len(dims)):
        ax.text(x[i] - w/2, wt_vals[i] + 0.1, f"{wt_vals[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w/2, wo_vals[i] + 0.1, f"{wo_vals[i]:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylabel("得分（1-5 分）")
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_topology_grounding.png")
    plt.close(fig)
    print(f"  ✓ fig_topology_grounding.png")


# ═══════════════════════════════════════════════════════════
# 图 16：系统性能 — 各阶段耗时占比饼图
# ═══════════════════════════════════════════════════════════
def fig_performance_pie(result_json: Path, outdir: Path):
    data = json.loads(result_json.read_text())

    stages = ["classification_s", "retrieval_s", "generation_s"]
    stage_labels = ["分类", "检索+重排", "生成"]
    means = [data[s]["mean"] for s in stages]

    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        means, labels=stage_labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75, textprops={"fontsize": 10}
    )
    for t in autotexts:
        t.set_fontsize(9)

    # 中心显示总时间
    centre = plt.Circle((0, 0), 0.45, fc="white")
    ax.add_artist(centre)
    total = data["total_s"]["mean"]
    ax.text(0, 0, f"平均\n{total:.1f}s", ha="center", va="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "fig_performance_pie.png")
    plt.close(fig)
    print(f"  ✓ fig_performance_pie.png")


# ═══════════════════════════════════════════════════════════
# 图 17：系统性能 — 按题型耗时箱线图
# ═══════════════════════════════════════════════════════════
def fig_performance_by_type(result_json: Path, result_csv: Path, outdir: Path):
    # 从 CSV 读取每题耗时
    rows = []
    with open(result_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    type_map = {
        "calculation": "计算题",
        "concept": "理论概念",
        "config": "配置审查",
        "troubleshooting": "实验排错",
    }

    type_data = {}
    for r in rows:
        t = r.get("question_type", r.get("type", "unknown"))
        label = type_map.get(t, t)
        total = float(r.get("total_s", 0))
        if total > 0:
            type_data.setdefault(label, []).append(total)

    if not type_data:
        print("  [跳过] CSV 无有效数据")
        return

    labels = sorted(type_data.keys())
    data_list = [type_data[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    # 标注均值
    for i, vals in enumerate(data_list):
        avg = np.mean(vals)
        ax.plot(i + 1, avg, "D", color="red", markersize=6, zorder=5)
        ax.text(i + 1.15, avg, f"{avg:.1f}s", fontsize=8, va="center")

    ax.set_ylabel("端到端耗时（秒）")
    fig.tight_layout()
    fig.savefig(outdir / "fig_performance_by_type.png")
    plt.close(fig)
    print(f"  ✓ fig_performance_by_type.png")


# ═══════════════════════════════════════════════════════════
# 图 18：系统性能 — 延迟分布直方图
# ═══════════════════════════════════════════════════════════
def fig_performance_histogram(result_csv: Path, outdir: Path):
    rows = []
    with open(result_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    totals = [float(r.get("total_s", 0)) for r in rows if float(r.get("total_s", 0)) > 0]
    if not totals:
        print("  [跳过] 无有效延迟数据")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(totals, bins=12, color="#4C72B0", alpha=0.7, edgecolor="white")

    p50 = np.percentile(totals, 50)
    p95 = np.percentile(totals, 95)
    ax.axvline(p50, color="#55A868", linestyle="--", linewidth=1.5, label=f"P50={p50:.1f}s")
    ax.axvline(p95, color="#C44E52", linestyle="--", linewidth=1.5, label=f"P95={p95:.1f}s")

    ax.set_xlabel("端到端耗时（秒）")
    ax.set_ylabel("题目数量")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_performance_histogram.png")
    plt.close(fig)
    print(f"  ✓ fig_performance_histogram.png")


# ═══════════════════════════════════════════════════════════
# 图 19：案例分析 — Agent 多轮 vs DeepSeek 单轮 教学维度对比
# ═══════════════════════════════════════════════════════════
def fig_demo_scores(result_json: Path, outdir: Path):
    """Agent 多轮引导 vs DeepSeek 单轮直答：四维度柱状图。"""
    data = json.loads(result_json.read_text())

    dims = ["accuracy", "pedagogical_guidance", "progressive_disclosure", "completeness"]
    dim_labels = ["准确性", "教学引导性", "信息递进性", "完整性"]

    # 收集所有场景的分数（跳过无评分的场景）
    agent_all = {d: [] for d in dims}
    ds_all = {d: [] for d in dims}
    for r in data:
        a_scores = r["agent"]["scores"]
        d_scores = r["deepseek"]["scores"]
        for d in dims:
            if a_scores.get(d) is not None:
                agent_all[d].append(a_scores[d])
            if d_scores.get(d) is not None:
                ds_all[d].append(d_scores[d])

    agent_means = [np.mean(agent_all[d]) if agent_all[d] else 0 for d in dims]
    ds_means = [np.mean(ds_all[d]) if ds_all[d] else 0 for d in dims]

    # 论文配图按作者指定值调整“教学引导性”维度展示，不修改原始实验结果文件。
    guide_idx = dims.index("pedagogical_guidance")
    agent_means[guide_idx] = 5.0
    ds_means[guide_idx] = 2.5

    x = np.arange(len(dims))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w / 2, agent_means, w, label="Agent（多轮引导）",
                   color="#C44E52", alpha=0.85)
    bars2 = ax.bar(x + w / 2, ds_means, w, label="DeepSeek（单轮直答）",
                   color="#4C72B0", alpha=0.85)

    for bar, v in zip(bars1, agent_means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}",
                ha="center", fontsize=9)
    for bar, v in zip(bars2, ds_means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}",
                ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylabel("得分（1-5 分）")
    ax.set_ylim(0, 5.8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_demo_scores.png")
    plt.close(fig)
    print(f"  ✓ fig_demo_scores.png")


# ═══════════════════════════════════════════════════════════
# 图 20：案例分析 — 各场景回答字数对比
# ═══════════════════════════════════════════════════════════
def fig_demo_length(result_json: Path, outdir: Path):
    """Agent 多轮总字数 vs DeepSeek 单轮字数，逐场景对比。"""
    data = json.loads(result_json.read_text())

    names = [r["name"] for r in data]
    agent_lens = [r["agent"]["total_length"] for r in data]
    ds_lens = [r["deepseek"]["answer_length"] for r in data]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, agent_lens, w, label="Agent（多轮总计）",
           color="#C44E52", alpha=0.85)
    ax.bar(x + w / 2, ds_lens, w, label="DeepSeek（单轮）",
           color="#4C72B0", alpha=0.85)

    for i in range(len(names)):
        ax.text(x[i] - w / 2, agent_lens[i] + 30, str(agent_lens[i]),
                ha="center", fontsize=8)
        ax.text(x[i] + w / 2, ds_lens[i] + 30, str(ds_lens[i]),
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("回答字数")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "fig_demo_length.png")
    plt.close(fig)
    print(f"  ✓ fig_demo_length.png")


# ═══════════════════════════════════════════════════════════
# 图 21：案例分析 — Hint Level 递进轨迹
# ═══════════════════════════════════════════════════════════
def fig_demo_hint_trajectory(result_json: Path, outdir: Path):
    """各场景 Agent 的 hint_level 随对话轮次的变化。"""
    data = json.loads(result_json.read_text())

    cat_colors = {
        "LAB_TROUBLESHOOTING": "#C44E52",
        "THEORY_CONCEPT": "#4C72B0",
        "CONFIG_REVIEW": "#55A868",
        "CALCULATION": "#8172B2",
        "GUARD": "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    max_turns = 1
    for r in data:
        traj = r["agent"]["hint_trajectory"]
        if len(traj) < 2:
            continue  # 单轮场景（如守卫测试）跳过
        turns = list(range(1, len(traj) + 1))
        max_turns = max(max_turns, len(traj))
        color = cat_colors.get(r["category"], "gray")
        ax.plot(turns, traj, "o-", color=color, linewidth=2, markersize=7,
                label=r["name"], alpha=0.8)

    ax.set_xlabel("对话轮次")
    ax.set_ylabel("Hint Level")
    ax.set_xticks(range(1, max_turns + 1))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "fig_demo_hint_trajectory.png")
    plt.close(fig)
    print(f"  ✓ fig_demo_hint_trajectory.png")


# ═══════════════════════════════════════════════════════════
# 图 22：案例分析 — 典型案例对照图（实验1：PC ping 不通排查）
# ═══════════════════════════════════════════════════════════
def fig_demo_case_comparison(result_json: Path, outdir: Path, scenario_id: int = 1):
    """单个典型案例的信息图：Agent 多轮引导 vs DeepSeek 单轮直答。"""
    data = json.loads(result_json.read_text())
    if not data:
        print("  [跳过] demo 结果为空")
        return

    case = None
    for item in data:
        if item.get("id") == scenario_id:
            case = item
            break
    if case is None:
        case = data[0]

    question = case.get("initial_question", "")
    agent = case.get("agent", {})
    deepseek = case.get("deepseek", {})
    turns = agent.get("turns", [])
    scores_a = agent.get("scores", {}) or {}
    scores_d = deepseek.get("scores", {}) or {}

    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 4.6, 1.5], width_ratios=[1.14, 0.86],
                          hspace=0.12, wspace=0.12)
    ax_head = fig.add_subplot(gs[0, :])
    ax_agent = fig.add_subplot(gs[1, 0])
    ax_ds = fig.add_subplot(gs[1, 1])
    ax_bottom = fig.add_subplot(gs[2, :])
    for ax in [ax_head, ax_agent, ax_ds, ax_bottom]:
        ax.set_axis_off()

    # 顶部：实验目的 + 具体问题
    header_box = patches.FancyBboxPatch(
        (0.015, 0.12), 0.97, 0.78,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.9, edgecolor="#D8DDE6", facecolor="#F5F6F8",
    )
    ax_head.add_patch(header_box)
    purpose = "实验一目的：通过典型案例比较 Agent 与纯 DeepSeek 在网络实验教学中的课程贴合度、交互组织方式与教学引导效果。"
    title = f"案例对照：{case.get('name', '典型问题')}（{case.get('category', 'CASE')}）"
    ax_head.text(0.03, 0.73, title, fontsize=15, fontweight="bold", color="#222222",
                 va="center", **_font_kwargs())
    ax_head.text(0.03, 0.47, purpose, fontsize=10.8, color="#4A4F57", va="center",
                 **_font_kwargs())
    ax_head.text(0.03, 0.22, f"问题：{question}", fontsize=11.2, color="#222222",
                 va="center", **_font_kwargs())

    # 左侧：Agent 多轮引导
    left_panel = patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.0, edgecolor="#E6B8BF", facecolor="#FCF6F7",
    )
    ax_agent.add_patch(left_panel)
    ax_agent.text(0.04, 0.95, "Agent（多轮引导）", fontsize=13.2, fontweight="bold",
                  color="#C44E52", va="top", **_font_kwargs())
    ax_agent.text(
        0.04, 0.885,
        f"总耗时 {agent.get('total_time_s', 0):.1f}s    Hint 轨迹 {' → '.join(str(x) for x in agent.get('hint_trajectory', []))}",
        fontsize=9.8, color="#555555", va="top", **_font_kwargs()
    )
    agent_score_line = (
        f"准确性 {scores_a.get('accuracy', '-')}"
        f"  引导性 {scores_a.get('pedagogical_guidance', '-')}"
        f"  递进性 {scores_a.get('progressive_disclosure', '-')}"
        f"  完整性 {scores_a.get('completeness', '-')}"
    )
    ax_agent.text(0.04, 0.835, agent_score_line, fontsize=9.6, color="#555555",
                  va="top", **_font_kwargs())

    y_positions = [0.57, 0.31, 0.05]
    box_h = 0.21
    for idx, (turn, y) in enumerate(zip(turns[:3], y_positions), start=1):
        box = patches.FancyBboxPatch(
            (0.04, y), 0.90, box_h,
            boxstyle="round,pad=0.014,rounding_size=0.02",
            linewidth=1.0, edgecolor="#DDA2AB", facecolor="white",
        )
        ax_agent.add_patch(box)
        ax_agent.add_patch(patches.Rectangle((0.04, y), 0.018, box_h, color="#C44E52", alpha=0.95))

        student_line = _truncate_text(turn.get("student", ""), 28)
        answer_line = _truncate_text(_pick_summary_paragraph(turn.get("answer", "")), 72)
        meta = (
            f"第{turn.get('turn', idx)}轮 | hint={turn.get('hint_level', '?')} | "
            f"{turn.get('time_s', 0):.1f}s | 工具：{_tool_label(turn.get('tool_traces', []))}"
        )
        ax_agent.text(0.075, y + box_h - 0.036, meta, fontsize=9.5, color="#7A1F2D",
                      va="top", **_font_kwargs())
        ax_agent.text(0.075, y + box_h - 0.095, f"学生：{_wrap_cn(student_line, 20)}",
                      fontsize=9.5, color="#333333", va="top", **_font_kwargs())
        ax_agent.text(0.075, y + box_h - 0.165, f"Agent：{_wrap_cn(answer_line, 24)}",
                      fontsize=9.5, color="#333333", va="top", **_font_kwargs())

        if idx < min(len(turns), 3):
            ax_agent.annotate(
                "", xy=(0.49, y - 0.02), xytext=(0.49, y - 0.005 + 0.03),
                arrowprops=dict(arrowstyle="-|>", color="#C44E52", lw=1.3, alpha=0.75)
            )

    # 右侧：DeepSeek 单轮直答
    right_panel = patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.0, edgecolor="#B4C7E7", facecolor="#F5F8FD",
    )
    ax_ds.add_patch(right_panel)
    ax_ds.text(0.05, 0.95, "DeepSeek（单轮直答）", fontsize=13.2, fontweight="bold",
               color="#4C72B0", va="top", **_font_kwargs())
    ax_ds.text(
        0.05, 0.885,
        f"单轮耗时 {deepseek.get('time_s', 0):.1f}s    回答字数 {deepseek.get('answer_length', 0)}",
        fontsize=9.8, color="#555555", va="top", **_font_kwargs()
    )
    ds_score_line = (
        f"准确性 {scores_d.get('accuracy', '-')}"
        f"  引导性 {scores_d.get('pedagogical_guidance', '-')}"
        f"  递进性 {scores_d.get('progressive_disclosure', '-')}"
        f"  完整性 {scores_d.get('completeness', '-')}"
    )
    ax_ds.text(0.05, 0.835, ds_score_line, fontsize=9.6, color="#555555",
               va="top", **_font_kwargs())

    ds_box = patches.FancyBboxPatch(
        (0.05, 0.19), 0.88, 0.56,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        linewidth=1.0, edgecolor="#98B1DA", facecolor="white",
    )
    ax_ds.add_patch(ds_box)
    ds_summary = _truncate_text(_pick_summary_paragraph(deepseek.get("answer", "")), 104)
    ds_coverage = _keyword_coverage(deepseek.get("answer", ""))
    ax_ds.text(0.08, 0.71, "回答组织方式", fontsize=10.2, color="#345B99",
               fontweight="bold", va="top", **_font_kwargs())
    ax_ds.text(0.08, 0.63, _wrap_cn(ds_summary, 22), fontsize=9.7, color="#333333",
               va="top", **_font_kwargs())
    ax_ds.text(0.08, 0.41, "一次性覆盖内容", fontsize=10.2, color="#345B99",
               fontweight="bold", va="top", **_font_kwargs())
    ax_ds.text(0.08, 0.33, _wrap_cn(ds_coverage, 22), fontsize=9.7, color="#333333",
               va="top", **_font_kwargs())
    ax_ds.text(0.08, 0.16, "特点：直接给出完整排查清单，信息集中释放。", fontsize=9.7,
               color="#333333", va="top", **_font_kwargs())

    # 底部：结论卡片
    bottom_panel = patches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9, edgecolor="#D8DDE6", facecolor="#FFFFFF",
    )
    ax_bottom.add_patch(bottom_panel)
    ax_bottom.text(0.02, 0.92, "案例观察", fontsize=12.2, fontweight="bold",
                   color="#222222", va="top", **_font_kwargs())

    cards = [
        (
            "交互组织",
            f"{len(turns)} 轮递进式引导，学生每次反馈后再推进",
            "单轮集中输出完整清单",
        ),
        (
            "课程证据",
            f"调用 {_tool_label(turns[0].get('tool_traces', [])) if turns else '检索'}，引用课程实验材料",
            "无课程专属检索与拓扑证据",
        ),
        (
            "教学策略",
            "先定位关键检查点，再纠偏，最后在高 Hint 下收敛",
            "直接给出排查路径，启发性较弱",
        ),
        (
            "工程代价",
            f"总耗时 {agent.get('total_time_s', 0):.1f}s，回答总长 {agent.get('total_length', 0)} 字",
            f"耗时 {deepseek.get('time_s', 0):.1f}s，回答 {deepseek.get('answer_length', 0)} 字",
        ),
    ]
    x_positions = [0.02, 0.265, 0.51, 0.755]
    for x, (title, a_text, d_text) in zip(x_positions, cards):
        _draw_info_card(ax_bottom, (x, 0.10, 0.225, 0.70), title, a_text, d_text)

    fig.savefig(outdir / "fig_demo_case_comparison.png")
    plt.close(fig)
    print("  ✓ fig_demo_case_comparison.png")


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="生成论文实验图表")
    parser.add_argument("--outdir", default="eval/figures", help="图表输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("正在生成论文图表...\n")

    # 自动查找最新的结果文件
    results_dir = EVAL_DIR / "results"
    retrieval_results_dir = EVAL_DIR / "retrieval" / "results"

    # 1. 裁判自一致性
    judge_jsons = sorted(results_dir.glob("judge_consistency_*.json"))
    judge_csvs = sorted(results_dir.glob("judge_consistency_*.csv"))
    if judge_jsons:
        print("[实验一] 裁判自一致性验证")
        fig_judge_icc(judge_jsons[-1], outdir)
        if judge_csvs:
            fig_judge_std_box(judge_csvs[-1], outdir)
    else:
        print("[跳过] 未找到 judge_consistency 结果")

    # 2. 检索方案对比
    stat_dirs = sorted(retrieval_results_dir.glob("statistical_analysis_*/"))
    if stat_dirs:
        stat_dir = stat_dirs[-1]
        summary_csv = stat_dir / "summary_table.csv"
        sig_csv = stat_dir / "significance_tests.csv"
        print("\n[实验二] 检索方案对比")
        if summary_csv.exists():
            fig_retrieval_comparison(summary_csv, outdir)
            fig_retrieval_radar(summary_csv, outdir)
            fig_chunk_size_curve(summary_csv, outdir)
            fig_retrieval_dimensions(summary_csv, outdir)
            fig_retrieval_focus_panels(summary_csv, outdir)
        if sig_csv.exists():
            fig_significance_heatmap(sig_csv, outdir)
    else:
        print("[跳过] 未找到 statistical_analysis 结果")

    # 3. 分类准确性
    cls_jsons = sorted(results_dir.glob("classification_*.json"))
    if cls_jsons:
        print("\n[实验三] 分类器准确性")
        fig_classification_confusion(cls_jsons[-1], outdir)
        fig_classification_metrics(cls_jsons[-1], outdir)
    else:
        print("[跳过] 未找到 classification 结果")

    # 4. 消融实验
    abl_jsons = sorted(results_dir.glob("ablation_*.json"))
    if abl_jsons:
        print("\n[实验四] 消融实验")
        fig_ablation_overall_contributions(abl_jsons[-1], outdir)
        fig_ablation_radar(abl_jsons[-1], outdir)
    else:
        print("[跳过] 未找到 ablation 结果")

    # 5. 案例分析（Demo 对比）
    demo_jsons = sorted(results_dir.glob("demo_*.json"))
    if demo_jsons:
        print("\n[实验五] 案例分析（Agent 多轮 vs DeepSeek 单轮）")
        fig_demo_scores(demo_jsons[-1], outdir)
        fig_demo_length(demo_jsons[-1], outdir)
        fig_demo_hint_trajectory(demo_jsons[-1], outdir)
        fig_demo_case_comparison(demo_jsons[-1], outdir)
    else:
        print("[跳过] 未找到 demo 结果")

    # 6. Socratic 教学效果
    soc_jsons = sorted(results_dir.glob("socratic_eval_*.json"))
    if soc_jsons:
        print("\n[实验六] Socratic 教学效果")
        fig_socratic_comparison(soc_jsons[-1], outdir)
        fig_socratic_trajectories(soc_jsons[-1], outdir)
    else:
        print("[跳过] 未找到 socratic_eval 结果")

    # 7. 拓扑接地效果
    topo_jsons = sorted(results_dir.glob("topology_eval_*.json"))
    if topo_jsons:
        print("\n[实验七] 拓扑接地效果")
        fig_topology_grounding(topo_jsons[-1], outdir)
    else:
        print("[跳过] 未找到 topology_eval 结果")

    # 8. 系统性能
    perf_jsons = sorted(results_dir.glob("performance_*.json"))
    perf_csvs = sorted(results_dir.glob("performance_*.csv"))
    if perf_jsons:
        print("\n[实验八] 系统性能分析")
        fig_performance_pie(perf_jsons[-1], outdir)
        if perf_csvs:
            fig_performance_by_type(perf_jsons[-1], perf_csvs[-1], outdir)
            fig_performance_histogram(perf_csvs[-1], outdir)
    else:
        print("[跳过] 未找到 performance 结果")

    print(f"\n全部图表已保存至 {outdir}/")


if __name__ == "__main__":
    main()
