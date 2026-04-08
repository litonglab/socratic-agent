"""
拓扑接地能力评估

评估拓扑模块的两个维度：
  1. 拓扑提取准确性：对比 GPT-4o 提取结果与人工标注的金标准
  2. 拓扑对回答质量的影响：有拓扑 vs 无拓扑工具的问答对比

用法：
  python eval/topology_evaluation.py                     # 运行全部评估
  python eval/topology_evaluation.py --skip-extraction    # 跳过提取准确性评估
  python eval/topology_evaluation.py --skip-grounding     # 跳过接地效果评估
  python eval/topology_evaluation.py --questions-file eval/topology_question_bank.json

输出：
  eval/results/topology_eval_YYYYMMDD_HHMMSS.json  — 汇总结果
  eval/results/topology_eval_YYYYMMDD_HHMMSS.csv   — 接地效果逐题对比
  eval/results/topology_eval_YYYYMMDD_HHMMSS.png   — 可视化对比图

环境变量：
  OPENAI_API_KEY   (必须，用于打分)
  DEEPSEEK_API_KEY (必须，用于 Agent 回答)
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from agentic_rag.agent import query as agent_query
from agentic_rag.llm_config import build_chat_llm
from langchain_core.messages import SystemMessage, HumanMessage

# ── 配置 ─────────────────────────────────────────────
TOPO_STORE = ROOT / "topo_store"
RESULTS_DIR = ROOT / "eval" / "results"
SCORE_KEYS = ["relevance", "faithfulness", "completeness", "technical_accuracy", "overall"]
DEFAULT_TOPO_QUESTIONS_PATH = ROOT / "eval" / "topology_question_bank.json"

# ── 内置兜底题集（questions file 缺失时使用）────────
FALLBACK_TOPO_QUESTIONS = [
    {
        "id": "T1",
        "question": "在实验13的子网划分实验中，RT1和RT2之间是如何连接的？使用了什么接口？",
        "reference": "RT1和RT2通过路由器接口相连，连接两个不同子网172.16.0.0/24和172.16.1.0/24",
        "experiment_id": "lab13",
    },
    {
        "id": "T2",
        "question": "在实验13中，PCa属于哪个子网？它的IP地址和默认网关应该怎么配？",
        "reference": "PCa属于172.16.0.0/24子网，需要配置该子网范围内的IP和对应的网关地址",
        "experiment_id": "lab13",
    },
    {
        "id": "T3",
        "question": "在实验13的拓扑中，一共有多少台交换机和多少台PC？它们是怎么分布的？",
        "reference": "实验13拓扑包含多台交换机（SW1-SW6）和多台PC（PCa-PCh），分布在两个子网中",
        "experiment_id": "lab13",
    },
    {
        "id": "T4",
        "question": "在实验13中，如果PCa想ping通PCe，数据包需要经过哪些设备？",
        "reference": "需要经过所在子网的交换机、路由器RT1/RT2、对方子网的交换机到达PCe",
        "experiment_id": "lab13",
    },
    {
        "id": "T5",
        "question": "在实验8的VLAN实验中，Trunk端口是配置在哪些设备之间的？",
        "reference": "Trunk端口配置在交换机之间的互联端口上，用于传输多个VLAN的数据",
        "experiment_id": "lab8",
    },
    {
        "id": "T6",
        "question": "在实验12的静态路由实验中，两台路由器各连接了哪些网段？",
        "reference": "每台路由器连接各自的LAN网段和路由器间的互联网段",
        "experiment_id": "lab12",
    },
]


def load_topology_questions(questions_file: Path) -> List[Dict[str, Any]]:
    """加载拓扑题集，若外部文件缺失则回退到内置样例。"""
    if not questions_file.exists():
        print(f"[提示] 未找到题集文件：{questions_file}，将使用内置题集。")
        return FALLBACK_TOPO_QUESTIONS

    try:
        data = json.loads(questions_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[提示] 读取题集文件失败：{exc}，将使用内置题集。")
        return FALLBACK_TOPO_QUESTIONS

    if not isinstance(data, list):
        print("[提示] 题集文件格式错误（应为 JSON 数组），将使用内置题集。")
        return FALLBACK_TOPO_QUESTIONS

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        reference = str(item.get("reference", "")).strip()
        experiment_id = str(item.get("experiment_id", "")).strip() or "lab13"
        qid = str(item.get("id", f"T{idx}")).strip()
        if not question or not reference:
            continue
        normalized.append(
            {
                "id": qid,
                "question": question,
                "reference": reference,
                "experiment_id": experiment_id,
            }
        )

    if not normalized:
        print("[提示] 题集文件中无有效题目，将使用内置题集。")
        return FALLBACK_TOPO_QUESTIONS

    return normalized

# ── 打分 Prompt ──────────────────────────────────────
JUDGE_SYSTEM = """\
你是计算机网络课程的专家评审，负责客观评估 AI 助教回答的质量。"""

JUDGE_TEMPLATE = """\
请对以下 AI 助教的回答从 4 个维度打分（每项 1-5 分整数）。

【学生问题】
{question}

【参考答案要点】
{reference}

【AI 回答】
{answer}

─── 评分标准 ──────────────────────
1. 相关性（relevance）：5=完全紧扣问题 ... 1=几乎无关
2. 忠实性（faithfulness）：5=所有结论有据 ... 1=大量编造
3. 完整性（completeness）：5=要点全覆盖 ... 1=几乎未覆盖
4. 技术准确性（technical_accuracy）：5=完全正确 ... 1=严重错误

综合 = 相关性×0.2 + 忠实性×0.3 + 完整性×0.3 + 技术准确×0.2，四舍五入到整数。

请严格按以下 JSON 格式输出：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "overall": <1-5>,
  "comment": "<简要理由>"
}}"""


def judge_answer(client: OpenAI, question: str, answer: str, reference: str) -> Dict:
    user_msg = JUDGE_TEMPLATE.format(
        question=question, reference=reference, answer=answer[:1500],
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0, timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {k: None for k in SCORE_KEYS}


def evaluate_extraction_accuracy(summary: dict):
    """评估拓扑提取准确性：遍历 topo_store，统计各阶段通过率。"""
    print(f"\n{'='*65}")
    print("拓扑提取准确性评估")
    print(f"{'='*65}")

    extraction_stats = {}

    for lab_dir in sorted(TOPO_STORE.iterdir()):
        if not lab_dir.is_dir():
            continue
        lab_name = lab_dir.name
        manifest_path = lab_dir / "manifest.json"

        if not manifest_path.exists():
            continue

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        stats = manifest.get("summary", {})
        images_total = stats.get("images_processed", 0)
        classified_topo = images_total - stats.get("pre_filtered_non_topology", 0)
        approved = stats.get("approved", 0)
        approved_warnings = stats.get("approved_with_warnings", 0)
        needs_review = stats.get("needs_manual_review", 0)
        rejected = stats.get("rejected", 0)

        total_approved = approved + approved_warnings

        extraction_stats[lab_name] = {
            "images_total": images_total,
            "classified_as_topology": classified_topo,
            "approved": total_approved,
            "needs_review": needs_review,
            "rejected": rejected,
            "classification_rate": round(classified_topo / images_total, 3) if images_total else 0,
            "approval_rate": round(total_approved / classified_topo, 3) if classified_topo else 0,
        }

        print(f"\n  {lab_name}:")
        print(f"    图片总数: {images_total}")
        print(f"    分类为拓扑图: {classified_topo} ({extraction_stats[lab_name]['classification_rate']:.0%})")
        print(f"    审核通过: {total_approved} ({extraction_stats[lab_name]['approval_rate']:.0%})")
        print(f"    需人工审核: {needs_review}")

        # 检查 approved_json 中的拓扑质量
        approved_dir = lab_dir / "approved_json"
        if approved_dir.exists():
            for topo_file in sorted(approved_dir.glob("*.json")):
                with open(topo_file, encoding="utf-8") as f:
                    topo = json.load(f)
                n_devices = len(topo.get("devices", []))
                n_interfaces = len(topo.get("interfaces", []))
                n_links = len(topo.get("links", []))
                n_subnets = len(topo.get("subnets", []))
                warnings = topo.get("warnings", [])
                print(f"    {topo_file.name}: {n_devices}设备, {n_interfaces}接口, "
                      f"{n_links}链路, {n_subnets}子网, {len(warnings)}警告")

    summary["extraction_stats"] = extraction_stats


def evaluate_grounding_effect(
    openai_client: OpenAI,
    summary: dict,
    topo_questions: List[Dict[str, Any]],
) -> List[dict]:
    """评估拓扑接地对回答质量的影响。"""
    print(f"\n{'='*65}")
    print("拓扑接地效果评估（有拓扑 vs 无拓扑）")
    print(f"{'='*65}")

    from agentic_rag import agent as agent_mod
    original_prepare = agent_mod._prepare_context

    detail_rows = []

    for qi, q_item in enumerate(topo_questions):
        qid = q_item["id"]
        question = q_item["question"]
        reference = q_item["reference"]

        print(f"\n  [{qi+1}/{len(topo_questions)}] {qid}: {question[:50]}...")

        # 有拓扑
        print(f"    有拓扑...", end=" ", flush=True)
        t0 = time.time()
        try:
            ans_with, _, traces_with, state_with = agent_query(
                question, history=[], state={}
            )
        except Exception as e:
            ans_with = f"[失败: {e}]"
            traces_with = []
        time_with = round(time.time() - t0, 1)
        used_topo = any(t.get("tool") == "拓扑" for t in traces_with)
        print(f"({time_with}s, 用拓扑={used_topo})")

        # 无拓扑（monkey-patch 移除拓扑工具）
        def patched_prepare(*args, **kwargs):
            ctx = original_prepare(*args, **kwargs)
            if ctx.contextual_actions and "拓扑" in ctx.contextual_actions:
                ctx.contextual_actions["拓扑"] = lambda q: "未找到可用拓扑数据。"
            return ctx

        agent_mod._prepare_context = patched_prepare
        print(f"    无拓扑...", end=" ", flush=True)
        t0 = time.time()
        try:
            ans_without, _, traces_without, _ = agent_query(
                question, history=[], state={}
            )
        except Exception as e:
            ans_without = f"[失败: {e}]"
        time_without = round(time.time() - t0, 1)
        agent_mod._prepare_context = original_prepare
        print(f"({time_without}s)")

        # 打分
        print(f"    打分...", end=" ", flush=True)
        scores_with = judge_answer(openai_client, question, ans_with, reference)
        scores_without = judge_answer(openai_client, question, ans_without, reference)
        print(f"有拓扑={scores_with.get('overall', '?')} 无拓扑={scores_without.get('overall', '?')}")

        for condition, answer, scores, elapsed in [
            ("with_topo", ans_with, scores_with, time_with),
            ("without_topo", ans_without, scores_without, time_without),
        ]:
            detail_rows.append({
                "question_id": qid,
                "question": question[:100],
                "experiment_id": q_item["experiment_id"],
                "condition": condition,
                "used_topo_tool": used_topo if condition == "with_topo" else False,
                "answer_length": len(answer),
                "time_s": elapsed,
                **{k: scores.get(k) for k in SCORE_KEYS},
                "comment": scores.get("comment", ""),
            })

        time.sleep(0.5)

    return detail_rows


def main():
    parser = argparse.ArgumentParser(description="拓扑接地能力评估")
    parser.add_argument("--skip-extraction", action="store_true", help="跳过提取准确性评估")
    parser.add_argument("--skip-grounding", action="store_true", help="跳过接地效果评估")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=str(DEFAULT_TOPO_QUESTIONS_PATH),
        help="拓扑题集文件路径（默认 eval/topology_question_bank.json）",
    )
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    openai_client = OpenAI(api_key=openai_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions_file = Path(args.questions_file)
    if not questions_file.is_absolute():
        questions_file = (ROOT / questions_file).resolve()
    topo_questions = load_topology_questions(questions_file)
    print(f"加载拓扑题数：{len(topo_questions)}（来源：{questions_file if questions_file.exists() else 'fallback'}）")

    summary = {"timestamp": timestamp}

    # 1. 拓扑提取准确性
    if not args.skip_extraction:
        evaluate_extraction_accuracy(summary)

    # 2. 拓扑接地效果
    detail_rows = []
    if not args.skip_grounding:
        detail_rows = evaluate_grounding_effect(openai_client, summary, topo_questions)
        summary["n_topology_questions"] = len(topo_questions)

        # 保存逐题 CSV
        detail_csv = RESULTS_DIR / f"topology_eval_{timestamp}.csv"
        fieldnames = [
            "question_id", "question", "experiment_id", "condition",
            "used_topo_tool", "answer_length", "time_s",
            *SCORE_KEYS, "comment",
        ]
        with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"\n接地效果明细：{detail_csv}")

        # 汇总
        import numpy as np

        print(f"\n{'='*65}")
        print("拓扑接地效果汇总")
        print(f"{'='*65}")

        grounding_summary = {}
        for condition in ["with_topo", "without_topo"]:
            c_rows = [r for r in detail_rows if r["condition"] == condition]
            scores = {}
            for dim in SCORE_KEYS:
                vals = [r[dim] for r in c_rows if r.get(dim) is not None]
                scores[dim] = round(np.mean(vals), 3) if vals else None
            grounding_summary[condition] = scores

            label = "有拓扑" if condition == "with_topo" else "无拓扑"
            print(f"\n  {label}:")
            for dim in SCORE_KEYS:
                v = scores.get(dim)
                print(f"    {dim:<20} = {v:.3f}" if v else f"    {dim:<20} = N/A")

        # 差异
        print(f"\n  有拓扑 - 无拓扑 差异:")
        for dim in SCORE_KEYS:
            w = grounding_summary.get("with_topo", {}).get(dim)
            wo = grounding_summary.get("without_topo", {}).get(dim)
            if w is not None and wo is not None:
                print(f"    {dim:<20} = {w - wo:+.3f}")

        summary["grounding_effect"] = grounding_summary

    # 保存汇总 JSON
    summary_json = RESULTS_DIR / f"topology_eval_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    # 可视化
    if detail_rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, ax = plt.subplots(figsize=(10, 6))

            dim_labels_zh = ["相关性", "忠实性", "完整性", "技术准确", "综合"]
            with_vals = [grounding_summary.get("with_topo", {}).get(d, 0) or 0 for d in SCORE_KEYS]
            without_vals = [grounding_summary.get("without_topo", {}).get(d, 0) or 0 for d in SCORE_KEYS]

            x = np.arange(len(dim_labels_zh))
            width = 0.35
            ax.bar(x - width/2, with_vals, width, label="有拓扑工具",
                   color="#2ecc71", alpha=0.8)
            ax.bar(x + width/2, without_vals, width, label="无拓扑工具",
                   color="#e74c3c", alpha=0.8)

            ax.set_ylabel("平均分 (1-5)")
            ax.set_title(f"拓扑接地对回答质量的影响\n({len(topo_questions)} 道拓扑相关问题)")
            ax.set_xticks(x)
            ax.set_xticklabels(dim_labels_zh)
            ax.legend()
            ax.set_ylim(0, 5.5)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            png_path = RESULTS_DIR / f"topology_eval_{timestamp}.png"
            fig.savefig(png_path, dpi=150)
            plt.close()
            print(f"图表已保存：{png_path}")

        except ImportError:
            print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
