"""
Socratic 教学效果定量评估

对 Agent（RAG + Socratic）vs 裸 DeepSeek 进行多轮对话的定量评估：
  - 新增 2 个教学维度的 GPT-4o 打分：引导性 + 信息递进性
  - 扩展到 10 个场景（覆盖 4 种题型）
  - Hint Level 轨迹分析
  - 多轮交互统计（解决轮数、token 消耗）

用法：
  python eval/socratic_evaluation.py                    # 运行全部场景
  python eval/socratic_evaluation.py --scenarios 1,2,3  # 只跑指定场景

输出：
  eval/results/socratic_eval_YYYYMMDD_HHMMSS.csv   — 逐轮打分明细
  eval/results/socratic_eval_YYYYMMDD_HHMMSS.json  — 汇总统计
  eval/results/socratic_eval_YYYYMMDD_HHMMSS.png   — 教学维度对比图 + hint轨迹图

环境变量：
  OPENAI_API_KEY   (必须，用于 GPT-4o 打分)
  DEEPSEEK_API_KEY (必须，用于 Agent 和 DeepSeek 回答)
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from agentic_rag.agent import query as agent_query
from agentic_rag.llm_config import build_chat_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── 配置 ─────────────────────────────────────────────
RESULTS_DIR = ROOT / "eval" / "results"

# ── Socratic 对话场景（10 个，覆盖 4 种题型）──────────
SCENARIOS = [
    # ── 实验排错（LAB_TROUBLESHOOTING）×3 ──
    {
        "id": 1,
        "name": "PC ping不通排查",
        "category": "LAB_TROUBLESHOOTING",
        "turns": [
            "我在实验1里，两台 PC 的 IP 地址都配好了，但是 ping 不通，怎么回事？",
            "我检查了一下，网线应该没问题，插得很紧。",
            "你说的防火墙，我打开看了但不知道要改哪里，能告诉我具体步骤吗？",
        ],
    },
    {
        "id": 2,
        "name": "VLAN间通信失败",
        "category": "LAB_TROUBLESHOOTING",
        "turns": [
            "在实验8里，我创建了VLAN 2和VLAN 3，但两个VLAN的PC互相ping不通，为什么？",
            "我已经把端口分配到了对应的VLAN，是不是还需要配什么？",
            "三层交换机上要怎么配置VLAN间路由？给我具体命令。",
        ],
    },
    {
        "id": 3,
        "name": "端口镜像抓不到包",
        "category": "LAB_TROUBLESHOOTING",
        "turns": [
            "在实验5中，我配置了端口镜像但Wireshark抓不到其他端口的包，什么原因？",
            "我确认镜像端口配置了，但不确定源端口和目的端口是否选对了。",
            "那正确的源端口和目的端口应该怎么配？请给具体步骤。",
        ],
    },
    # ── 理论概念（THEORY_CONCEPT）×3 ──
    {
        "id": 4,
        "name": "ARP协议理解",
        "category": "THEORY_CONCEPT",
        "turns": [
            "在实验11里，ARP协议是用来干什么的？我觉得它跟DNS一样，都是把名字转成地址。",
            "哦，那是把IP地址转成MAC地址吗？但为什么需要MAC地址？",
        ],
    },
    {
        "id": 5,
        "name": "交叉线与直通线",
        "category": "THEORY_CONCEPT",
        "turns": [
            "为什么连接两台交换机要用交叉线而不是直通线？",
            "那如果现在的交换机支持MDI/MDI-X自动翻转，是不是就不需要区分了？",
        ],
    },
    {
        "id": 6,
        "name": "VLAN原理",
        "category": "THEORY_CONCEPT",
        "turns": [
            "VLAN到底是什么？跟物理划分子网有什么区别？",
            "那VLAN标签是怎么加到数据帧上的？和Trunk端口有什么关系？",
        ],
    },
    # ── 配置操作（CONFIG_REVIEW）×2 ──
    {
        "id": 7,
        "name": "H3C VLAN配置",
        "category": "CONFIG_REVIEW",
        "turns": [
            "我要在H3C三层交换机上配置VLAN 2，第一步应该输入什么命令？",
            "我输入了sys进入了系统视图，然后应该怎么创建VLAN 2？",
            "我已经用vlan 2创建好了，那怎么把端口GigabitEthernet 1/0/1加进去？",
        ],
    },
    {
        "id": 8,
        "name": "静态路由配置",
        "category": "CONFIG_REVIEW",
        "turns": [
            "在实验12中，怎么在路由器上配置一条到172.16.1.0/24网段的静态路由？",
            "ip route-static命令后面的参数分别是什么意思？",
            "如果有两条路径可以到达目的网段，怎么设置优先级？",
        ],
    },
    # ── 计算分析（CALCULATION）×2 ──
    {
        "id": 9,
        "name": "子网划分计算",
        "category": "CALCULATION",
        "turns": [
            "在实验13中，把172.16.0.0/16划分成4个子网，每个子网的地址范围是多少？",
            "我算出子网掩码是/18，但不太确定每个子网的起始地址怎么算。",
        ],
    },
    {
        "id": 10,
        "name": "可用主机数计算",
        "category": "CALCULATION",
        "turns": [
            "一个/26的子网最多能容纳多少台主机？",
            "为什么要减去2？网络地址和广播地址分别是怎么确定的？",
        ],
    },
]

BASELINE_SYSTEM = "你是一个计算机网络实验课程的助教，请清晰、完整地回答学生的问题。"

# ── GPT-4o 教学质量打分 Prompt ────────────────────────
TEACHING_JUDGE_SYSTEM = """\
你是教育学和计算机网络双重专家。请从教学效果角度评估 AI 助教的回答质量。
你的打分必须严格、一致，根据锚点描述打分。"""

TEACHING_JUDGE_TEMPLATE = """\
请评估以下 AI 助教在多轮对话中的单轮回答的教学质量。

【对话上下文】
{context}

【当前学生提问】
{student_message}

【AI 助教回答】
{assistant_answer}

─── 评分标准（每项 1-5 分整数）──────────────────────
1. 内容准确性（accuracy）
   5=技术内容完全正确
   4=基本正确，有1处小瑕疵
   3=有1-2处明显错误
   2=多处错误
   1=严重错误

2. 引导性（pedagogical_guidance）
   5=通过提问/类比/反问等方式引导学生主动思考，不直接给出完整答案
   4=有引导意识，但部分内容直接给出
   3=以回答为主，偶有引导性提问
   2=基本是直接回答，缺乏引导
   1=完全直接给答案，无任何引导

3. 信息递进性（progressive_disclosure）
   5=严格控制信息量，按学生理解程度逐步揭示，与对话历史高度关联
   4=有递进意识，信息量控制较好
   3=信息量适中，但未充分利用对话历史
   2=一次性给出过多信息，缺乏递进
   1=无视对话上下文，信息量不受控

4. 回答完整性（completeness）
   5=在当前引导层级下，回答内容充分，不遗漏关键信息
   4=大部分关键信息已覆盖
   3=覆盖约60%关键信息
   2=仅覆盖少数关键信息
   1=几乎未覆盖

────────────────────────────────────────────────────

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "accuracy": <1-5>,
  "pedagogical_guidance": <1-5>,
  "progressive_disclosure": <1-5>,
  "completeness": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""

TEACHING_SCORE_KEYS = ["accuracy", "pedagogical_guidance", "progressive_disclosure", "completeness"]


def judge_teaching_quality(
    client: OpenAI,
    context: str,
    student_message: str,
    assistant_answer: str,
) -> Dict[str, Any]:
    """GPT-4o 教学质量打分。"""
    user_msg = TEACHING_JUDGE_TEMPLATE.format(
        context=context,
        student_message=student_message,
        assistant_answer=assistant_answer[:1500],
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": TEACHING_JUDGE_SYSTEM},
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
        return {k: None for k in TEACHING_SCORE_KEYS}


def format_context(turns_so_far: List[Dict]) -> str:
    """格式化对话上下文用于打分。"""
    if not turns_so_far:
        return "（首轮对话，无历史上下文）"
    lines = []
    for t in turns_so_far[-3:]:  # 最多展示最近3轮
        lines.append(f"学生：{t['student'][:200]}")
        lines.append(f"AI助教：{t['agent_answer'][:300]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Socratic 教学效果定量评估")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="要运行的场景 ID（逗号分隔，默认全部）")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    scenarios = SCENARIOS
    if args.scenarios:
        ids = set(int(x) for x in args.scenarios.split(","))
        scenarios = [s for s in SCENARIOS if s["id"] in ids]

    print(f"将运行 {len(scenarios)} 个场景")

    # 初始化
    llm = build_chat_llm(temperature=0)
    openai_client = OpenAI(api_key=openai_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_detail_rows = []
    all_scenario_logs = []

    for scenario in scenarios:
        sid = scenario["id"]
        print(f"\n{'='*70}")
        print(f"场景 {sid}：{scenario['name']} [{scenario['category']}]")
        print(f"{'='*70}")

        agent_history = []
        agent_state = {}
        ds_history = []
        turns_so_far = []

        scenario_log = {
            "id": sid,
            "name": scenario["name"],
            "category": scenario["category"],
            "turns": [],
        }

        for ti, student_msg in enumerate(scenario["turns"], 1):
            print(f"\n  ── 第 {ti} 轮 ──")
            print(f"  学生：{student_msg[:60]}...")

            # 构建打分上下文
            context = format_context(turns_so_far)

            # Agent 调用
            print(f"    Agent...", end=" ", flush=True)
            t0 = time.time()
            try:
                agent_ans, agent_history, tool_traces, agent_state = agent_query(
                    student_msg, history=agent_history, state=agent_state
                )
            except Exception as e:
                agent_ans = f"[调用失败: {e}]"
                tool_traces = []
            agent_t = round(time.time() - t0, 1)
            hint_level = agent_state.get("hint_level", 0)
            print(f"({agent_t}s, hint={hint_level})")

            # DeepSeek 调用
            print(f"    DeepSeek...", end=" ", flush=True)
            t0 = time.time()
            messages = [SystemMessage(content=BASELINE_SYSTEM)] + ds_history + [
                HumanMessage(content=student_msg)
            ]
            try:
                ds_resp = llm.invoke(messages)
                ds_ans = ds_resp.content
                ds_history = ds_history + [
                    HumanMessage(content=student_msg),
                    AIMessage(content=ds_ans),
                ]
            except Exception as e:
                ds_ans = f"[调用失败: {e}]"
            ds_t = round(time.time() - t0, 1)
            print(f"({ds_t}s)")

            # GPT-4o 打分：Agent
            print(f"    打分 Agent...", end=" ", flush=True)
            agent_scores = judge_teaching_quality(
                openai_client, context, student_msg, agent_ans,
            )
            print(f"guidance={agent_scores.get('pedagogical_guidance', '?')}")

            # GPT-4o 打分：DeepSeek
            print(f"    打分 DeepSeek...", end=" ", flush=True)
            ds_scores = judge_teaching_quality(
                openai_client, context, student_msg, ds_ans,
            )
            print(f"guidance={ds_scores.get('pedagogical_guidance', '?')}")

            # 记录
            for system_name, scores, answer, elapsed in [
                ("agent", agent_scores, agent_ans, agent_t),
                ("deepseek", ds_scores, ds_ans, ds_t),
            ]:
                all_detail_rows.append({
                    "scenario_id": sid,
                    "scenario_name": scenario["name"],
                    "category": scenario["category"],
                    "turn": ti,
                    "system": system_name,
                    "hint_level": hint_level if system_name == "agent" else None,
                    "answer_length": len(answer),
                    "time_s": elapsed,
                    **{k: scores.get(k) for k in TEACHING_SCORE_KEYS},
                    "comment": scores.get("comment", ""),
                })

            scenario_log["turns"].append({
                "turn": ti,
                "student": student_msg,
                "hint_level": hint_level,
                "agent_answer": agent_ans,
                "agent_scores": agent_scores,
                "agent_time": agent_t,
                "agent_tools": len(tool_traces),
                "deepseek_answer": ds_ans,
                "deepseek_scores": ds_scores,
                "deepseek_time": ds_t,
            })

            turns_so_far.append({
                "student": student_msg,
                "agent_answer": agent_ans,
            })

            time.sleep(0.5)

        all_scenario_logs.append(scenario_log)

    # 保存逐轮明细 CSV
    detail_csv = RESULTS_DIR / f"socratic_eval_{timestamp}.csv"
    fieldnames = [
        "scenario_id", "scenario_name", "category", "turn", "system",
        "hint_level", "answer_length", "time_s",
        *TEACHING_SCORE_KEYS, "comment",
    ]
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detail_rows)
    print(f"\n明细已保存：{detail_csv}")

    # 汇总统计
    import numpy as np

    summary = {
        "timestamp": timestamp,
        "n_scenarios": len(scenarios),
        "systems": {},
        "by_category": {},
    }

    print(f"\n{'='*75}")
    print("Socratic 教学效果评估汇总")
    print(f"{'='*75}")

    for system in ["agent", "deepseek"]:
        sys_rows = [r for r in all_detail_rows if r["system"] == system]
        sys_label = "Agent (Socratic)" if system == "agent" else "裸 DeepSeek"

        scores_summary = {}
        for dim in TEACHING_SCORE_KEYS:
            vals = [r[dim] for r in sys_rows if r.get(dim) is not None]
            scores_summary[dim] = round(np.mean(vals), 3) if vals else None

        summary["systems"][system] = scores_summary

        print(f"\n  {sys_label}:")
        for dim in TEACHING_SCORE_KEYS:
            v = scores_summary.get(dim)
            print(f"    {dim:<28} = {v:.3f}" if v else f"    {dim:<28} = N/A")

    # Agent vs DeepSeek 差异
    print(f"\n  Agent - DeepSeek 差异:")
    for dim in TEACHING_SCORE_KEYS:
        a = summary["systems"].get("agent", {}).get(dim)
        d = summary["systems"].get("deepseek", {}).get(dim)
        if a is not None and d is not None:
            diff = a - d
            print(f"    {dim:<28} = {diff:+.3f}")

    # 按 category 分组
    categories = set(r["category"] for r in all_detail_rows)
    for cat in sorted(categories):
        cat_summary = {}
        for system in ["agent", "deepseek"]:
            cat_rows = [r for r in all_detail_rows
                        if r["system"] == system and r["category"] == cat]
            for dim in TEACHING_SCORE_KEYS:
                vals = [r[dim] for r in cat_rows if r.get(dim) is not None]
                key = f"{system}_{dim}"
                cat_summary[key] = round(np.mean(vals), 3) if vals else None
        summary["by_category"][cat] = cat_summary

    # Hint Level 轨迹统计
    print(f"\n{'='*75}")
    print("Hint Level 轨迹")
    print(f"{'='*75}")

    hint_trajectories = []
    for log in all_scenario_logs:
        levels = [t["hint_level"] for t in log["turns"]]
        hint_trajectories.append({
            "scenario_id": log["id"],
            "name": log["name"],
            "category": log["category"],
            "trajectory": levels,
            "max_level": max(levels),
            "level_increases": sum(1 for i in range(1, len(levels)) if levels[i] > levels[i-1]),
        })
        print(f"  场景 {log['id']} ({log['name']}): {' → '.join(str(l) for l in levels)}")

    summary["hint_trajectories"] = hint_trajectories

    # 保存汇总 JSON
    summary_json = RESULTS_DIR / f"socratic_eval_{timestamp}.json"
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：Agent vs DeepSeek 教学维度对比
        ax1 = axes[0]
        dim_labels = ["准确性", "引导性", "递进性", "完整性"]
        agent_vals = [summary["systems"].get("agent", {}).get(dim, 0) or 0
                      for dim in TEACHING_SCORE_KEYS]
        ds_vals = [summary["systems"].get("deepseek", {}).get(dim, 0) or 0
                   for dim in TEACHING_SCORE_KEYS]

        x = np.arange(len(dim_labels))
        width = 0.35
        bars1 = ax1.bar(x - width/2, agent_vals, width, label="Agent (Socratic)",
                        color="#2ecc71", alpha=0.8)
        bars2 = ax1.bar(x + width/2, ds_vals, width, label="裸 DeepSeek",
                        color="#3498db", alpha=0.8)

        ax1.set_ylabel("平均分 (1-5)")
        ax1.set_title("教学效果维度对比")
        ax1.set_xticks(x)
        ax1.set_xticklabels(dim_labels)
        ax1.legend()
        ax1.set_ylim(0, 5.5)
        ax1.grid(axis="y", alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                         f"{h:.1f}", ha="center", va="bottom", fontsize=9)

        # 右图：Hint Level 轨迹
        ax2 = axes[1]
        colors_list = plt.cm.Set2(np.linspace(0, 1, len(hint_trajectories)))
        for ht, color in zip(hint_trajectories, colors_list):
            turns = list(range(1, len(ht["trajectory"]) + 1))
            ax2.plot(turns, ht["trajectory"], "o-", color=color,
                     label=f"S{ht['scenario_id']}", linewidth=2, markersize=6)

        ax2.set_xlabel("对话轮次")
        ax2.set_ylabel("Hint Level")
        ax2.set_title("各场景 Hint Level 变化轨迹")
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_ylim(-0.2, 3.5)
        ax2.legend(fontsize=7, ncol=2, loc="upper left")
        ax2.grid(alpha=0.3)

        fig.suptitle("Socratic 教学效果定量评估", fontsize=14, y=1.02)
        fig.tight_layout()
        png_path = RESULTS_DIR / f"socratic_eval_{timestamp}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"图表已保存：{png_path}")

    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
