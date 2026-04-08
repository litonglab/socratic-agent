"""
水平自适应模块专项评估

验证系统能根据学生水平差异化教学：
  - 强学生（score=0.75）→ hint_level=0，更多引导性提问
  - 中等学生（score=0.42）→ hint_level=1，适度引导+部分解释
  - 弱学生（score=0.25）→ hint_level=2，更直接的解释

方法：
  1. 在 SQLite 中预写入 3 个模拟用户的 proficiency 记录
  2. 用相同问题集分别调用 agent_query（传入不同 user_id）
  3. GPT-4o 逐轮打分（引导性、递进性、完整性、准确性）
  4. 对比不同水平学生收到的回答差异

预期结果：
  - 引导性：强学生 > 中等 > 弱学生
  - 完整性：弱学生 > 中等 > 强学生（弱学生收到更直接的答案）
  - 准确性：三者接近（水平适配不影响正确性）

用法：
  python eval/proficiency_evaluation.py
  python eval/proficiency_evaluation.py --scenarios 1,2,3
  python eval/proficiency_evaluation.py --n-turns 2

输出：
  eval/results/proficiency_eval_YYYYMMDD_HHMMSS.csv
  eval/results/proficiency_eval_YYYYMMDD_HHMMSS.json
  eval/results/proficiency_eval_YYYYMMDD_HHMMSS.png
"""

import os
import sys
import json
import csv
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from agentic_rag.agent import query as agent_query
from storage.user_store import upsert_proficiency_score, create_user

# ── 配置 ─────────────────────────────────────────────
RESULTS_DIR = ROOT / "eval" / "results"

CATEGORIES = ["LAB_TROUBLESHOOTING", "THEORY_CONCEPT", "CONFIG_REVIEW", "CALCULATION"]

# ── 模拟学生档案 ─────────────────────────────────────
STUDENT_PROFILES = {
    "strong": {
        "user_id": "__eval_strong_student__",
        "label": "强学生",
        "scores": {
            "LAB_TROUBLESHOOTING": 0.78,
            "THEORY_CONCEPT": 0.72,
            "CONFIG_REVIEW": 0.80,
            "CALCULATION": 0.70,
            "OVERALL": 0.75,
        },
        "confidence": 0.85,
        "interaction_count": 25,
        "expected_hint_level": 0,
    },
    "medium": {
        "user_id": "__eval_medium_student__",
        "label": "中等学生",
        "scores": {
            "LAB_TROUBLESHOOTING": 0.45,
            "THEORY_CONCEPT": 0.40,
            "CONFIG_REVIEW": 0.42,
            "CALCULATION": 0.38,
            "OVERALL": 0.42,
        },
        "confidence": 0.70,
        "interaction_count": 18,
        "expected_hint_level": 1,
    },
    "weak": {
        "user_id": "__eval_weak_student__",
        "label": "弱学生",
        "scores": {
            "LAB_TROUBLESHOOTING": 0.22,
            "THEORY_CONCEPT": 0.28,
            "CONFIG_REVIEW": 0.25,
            "CALCULATION": 0.20,
            "OVERALL": 0.25,
        },
        "confidence": 0.75,
        "interaction_count": 20,
        "expected_hint_level": 2,
    },
}

# ── 测试场景（覆盖 4 种题型，每类 2 题）─────────────
SCENARIOS = [
    {
        "id": 1,
        "category": "LAB_TROUBLESHOOTING",
        "question": "我在实验1里，两台 PC 的 IP 地址都配好了，但是 ping 不通，怎么回事？",
        "followup": "我检查了网线没问题，还有什么可能的原因？",
    },
    {
        "id": 2,
        "category": "LAB_TROUBLESHOOTING",
        "question": "在实验8里，我创建了VLAN 2和VLAN 3，但两个VLAN的PC互相ping不通，为什么？",
        "followup": "端口已经分配到对应VLAN了，还需要配什么？",
    },
    {
        "id": 3,
        "category": "THEORY_CONCEPT",
        "question": "ARP协议是用来干什么的？我觉得它跟DNS一样，都是把名字转成地址。",
        "followup": "那是把IP地址转成MAC地址吗？但为什么需要MAC地址？",
    },
    {
        "id": 4,
        "category": "THEORY_CONCEPT",
        "question": "VLAN到底是什么？跟物理划分子网有什么区别？",
        "followup": "那VLAN标签是怎么加到数据帧上的？",
    },
    {
        "id": 5,
        "category": "CONFIG_REVIEW",
        "question": "我要在H3C三层交换机上配置VLAN 2，第一步应该输入什么命令？",
        "followup": "进入系统视图后怎么创建VLAN 2并把端口加进去？",
    },
    {
        "id": 6,
        "category": "CONFIG_REVIEW",
        "question": "在实验12中，怎么在路由器上配置一条到172.16.1.0/24网段的静态路由？",
        "followup": "ip route-static命令后面的参数分别是什么意思？",
    },
    {
        "id": 7,
        "category": "CALCULATION",
        "question": "把172.16.0.0/16划分成4个子网，每个子网的地址范围是多少？",
        "followup": "我算出子网掩码是/18，但不确定每个子网的起始地址怎么算。",
    },
    {
        "id": 8,
        "category": "CALCULATION",
        "question": "一个/26的子网最多能容纳多少台主机？",
        "followup": "为什么要减去2？网络地址和广播地址分别是怎么确定的？",
    },
]

# ── 学生跟进消息（多轮用）─────────────────────────────
STUDENT_FOLLOWUPS = [
    "我不太理解，能再详细解释一下吗？",
]

# ── GPT-4o 打分 Prompt ──────────────────────────────
JUDGE_SYSTEM = """\
你是教育学和计算机网络双重专家。请从教学效果角度评估 AI 助教的回答质量。
你的打分必须严格、一致，根据锚点描述打分。"""

JUDGE_TEMPLATE = """\
请评估以下 AI 助教在多轮对话中的回答的教学质量。

【学生水平】{student_level}

【对话记录】
{conversation}

─── 评分标准（每项 1-5 分整数）──────────────────────
1. 内容准确性（accuracy）
   5=技术内容完全正确
   4=基本正确，有1处小瑕疵
   3=有1-2处明显错误
   2=多处错误
   1=严重错误

2. 教学引导性（pedagogical_guidance）
   5=通过提问/类比/反问等方式引导学生主动思考，不直接给出完整答案
   4=有引导意识，但部分内容直接给出
   3=以回答为主，偶有引导性提问
   2=基本是直接回答，缺乏引导
   1=完全直接给答案，无任何引导

3. 信息递进性（progressive_disclosure）
   5=严格控制信息量，按学生理解程度逐步揭示
   4=有递进意识，信息量控制较好
   3=信息量适中，但未充分利用对话历史
   2=一次性给出过多信息，缺乏递进
   1=无视对话上下文，信息量不受控

4. 水平适配度（level_adaptation）
   5=回答的深度、术语使用、解释详细程度完全匹配该学生水平
   4=基本匹配，偶尔偏深或偏浅
   3=部分适配，但有明显不匹配之处
   2=回答深度与学生水平不匹配
   1=完全未考虑学生水平

5. 回答完整性（completeness）
   5=在当前教学策略下，回答内容充分，关键信息覆盖到位
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
  "level_adaptation": <1-5>,
  "completeness": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""

SCORE_KEYS = ["accuracy", "pedagogical_guidance", "progressive_disclosure",
              "level_adaptation", "completeness"]


def setup_student_profiles():
    """在 SQLite 中预写入模拟学生的 proficiency 记录。"""
    for profile_name, profile in STUDENT_PROFILES.items():
        uid = profile["user_id"]
        # 先确保用户存在（忽略已存在的情况）
        try:
            create_user({
                "id": uid,
                "username": uid,
                "password_salt": "eval",
                "password_hash": "eval",
            })
        except Exception:
            pass  # 用户已存在
        for cat, score in profile["scores"].items():
            upsert_proficiency_score(
                user_id=uid,
                category=cat,
                score=score,
                confidence=profile["confidence"],
                interaction_count=profile["interaction_count"],
            )
        print(f"  已写入 {profile['label']}（{uid}）: "
              f"OVERALL={profile['scores']['OVERALL']:.2f}, "
              f"预期hint_level={profile['expected_hint_level']}")


def run_scenario(scenario: dict, user_id: str, n_turns: int) -> dict:
    """对单个场景执行多轮对话，返回完整记录。"""
    history = []
    state = {}
    turns_log = []

    # 第 1 轮：原始问题
    t0 = time.time()
    answer, history, tool_traces, state = agent_query(
        scenario["question"], history=history, state=state, user_id=user_id
    )
    elapsed = round(time.time() - t0, 1)
    hint_level = state.get("hint_level", 0)

    turns_log.append({
        "turn": 1,
        "student_msg": scenario["question"],
        "answer": answer,
        "hint_level": hint_level,
        "time_s": elapsed,
    })

    # 第 2 轮：场景特定跟进
    if n_turns >= 2:
        t0 = time.time()
        answer, history, tool_traces, state = agent_query(
            scenario["followup"], history=history, state=state, user_id=user_id
        )
        elapsed = round(time.time() - t0, 1)
        hint_level = state.get("hint_level", 0)

        turns_log.append({
            "turn": 2,
            "student_msg": scenario["followup"],
            "answer": answer,
            "hint_level": hint_level,
            "time_s": elapsed,
        })

    # 第 3 轮（可选）：通用跟进
    if n_turns >= 3:
        followup = STUDENT_FOLLOWUPS[0]
        t0 = time.time()
        answer, history, tool_traces, state = agent_query(
            followup, history=history, state=state, user_id=user_id
        )
        elapsed = round(time.time() - t0, 1)
        hint_level = state.get("hint_level", 0)

        turns_log.append({
            "turn": 3,
            "student_msg": followup,
            "answer": answer,
            "hint_level": hint_level,
            "time_s": elapsed,
        })

    return {
        "scenario_id": scenario["id"],
        "category": scenario["category"],
        "turns": turns_log,
        "hint_trajectory": [t["hint_level"] for t in turns_log],
    }


def format_conversation(turns_log: list) -> str:
    """格式化对话记录用于打分。"""
    lines = []
    for t in turns_log:
        lines.append(f"学生（第{t['turn']}轮）：{t['student_msg']}")
        lines.append(f"AI助教（hint_level={t['hint_level']}）：{t['answer'][:500]}")
        lines.append("")
    return "\n".join(lines)


def judge_response(client: OpenAI, student_level: str, conversation: str) -> Dict[str, Any]:
    """GPT-4o 打分。"""
    user_msg = JUDGE_TEMPLATE.format(
        student_level=student_level,
        conversation=conversation[:3000],
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
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


def main():
    parser = argparse.ArgumentParser(description="水平自适应模块专项评估")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="要运行的场景 ID（逗号分隔，默认全部）")
    parser.add_argument("--n-turns", type=int, default=2,
                        help="每个场景的对话轮数（默认 2）")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    scenarios = SCENARIOS
    if args.scenarios:
        ids = set(int(x) for x in args.scenarios.split(","))
        scenarios = [s for s in SCENARIOS if s["id"] in ids]

    profiles = ["strong", "medium", "weak"]
    total = len(scenarios) * len(profiles)

    print(f"水平自适应模块专项评估")
    print(f"  场景数: {len(scenarios)}, 学生档案: {len(profiles)}, 对话轮数: {args.n_turns}")
    print(f"  总评测数: {total}")
    print()

    # 写入模拟学生档案
    print("写入模拟学生 proficiency 记录...")
    setup_student_profiles()
    print()

    openai_client = OpenAI(api_key=openai_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_logs = []
    done_counter = {"n": 0}
    print_lock = threading.Lock()

    def run_single_task(scenario, profile_name):
        """单个 (场景, 学生) 任务，线程安全。"""
        sid = scenario["id"]
        profile = STUDENT_PROFILES[profile_name]
        uid = profile["user_id"]
        label = profile["label"]

        try:
            result = run_scenario(scenario, uid, args.n_turns)
            hint_traj = result["hint_trajectory"]
        except Exception as e:
            result = {
                "scenario_id": sid,
                "category": scenario["category"],
                "turns": [],
                "hint_trajectory": [],
            }
            hint_traj = []

        # GPT-4o 打分
        if result["turns"]:
            conversation = format_conversation(result["turns"])
            scores = judge_response(
                openai_client,
                student_level=f"{label}（proficiency={profile['scores']['OVERALL']:.2f}）",
                conversation=conversation,
            )
        else:
            scores = {k: None for k in SCORE_KEYS}

        with print_lock:
            done_counter["n"] += 1
            hint_str = ' → '.join(str(h) for h in hint_traj) if hint_traj else "N/A"
            print(f"  [{done_counter['n']}/{total}] 场景{sid} {label}: "
                  f"hint={hint_str}  "
                  f"guidance={scores.get('pedagogical_guidance', '?')} "
                  f"adapt={scores.get('level_adaptation', '?')}")

        row = {
            "scenario_id": sid,
            "category": scenario["category"],
            "profile": profile_name,
            "profile_label": label,
            "proficiency_score": profile["scores"]["OVERALL"],
            "initial_hint_level": profile["expected_hint_level"],
            "hint_trajectory": str(hint_traj),
            "total_answer_length": sum(len(t["answer"]) for t in result["turns"]),
            "total_time_s": sum(t["time_s"] for t in result["turns"]),
            **{k: scores.get(k) for k in SCORE_KEYS},
            "comment": scores.get("comment", ""),
        }
        log = {**result, "profile": profile_name, "scores": scores}
        return row, log

    # 并发执行：每个 (场景, 学生) 组合独立
    max_workers = min(6, total)
    print(f"  并发线程数: {max_workers}")

    tasks = [(s, p) for s in scenarios for p in profiles]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_task, s, p): (s["id"], p)
            for s, p in tasks
        }
        for future in as_completed(futures):
            try:
                row, log = future.result()
                all_rows.append(row)
                all_logs.append(log)
            except Exception as e:
                sid, pname = futures[future]
                print(f"  [错误] 场景{sid} {pname}: {e}")

    # 保存 CSV
    output_csv = RESULTS_DIR / f"proficiency_eval_{timestamp}.csv"
    fieldnames = [
        "scenario_id", "category", "profile", "profile_label",
        "proficiency_score", "initial_hint_level", "hint_trajectory",
        "total_answer_length", "total_time_s",
        *SCORE_KEYS, "comment",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n明细已保存：{output_csv}")

    # 汇总统计
    print(f"\n{'='*75}")
    print("水平自适应评估汇总")
    print(f"{'='*75}")

    dim_zh = {
        "accuracy": "准确性", "pedagogical_guidance": "引导性",
        "progressive_disclosure": "递进性", "level_adaptation": "水平适配",
        "completeness": "完整性",
    }

    summary = {"timestamp": timestamp, "n_scenarios": len(scenarios), "profiles": {}}

    print(f"\n  {'学生类型':<10} {'水平分':>6} {'hint':>5} {'准确性':>6} {'引导性':>6} "
          f"{'递进性':>6} {'适配度':>6} {'完整性':>6}")
    print(f"  {'-'*65}")

    profile_scores = {}
    for pname in profiles:
        p_rows = [r for r in all_rows if r["profile"] == pname]
        label = STUDENT_PROFILES[pname]["label"]
        prof_score = STUDENT_PROFILES[pname]["scores"]["OVERALL"]
        hint = STUDENT_PROFILES[pname]["expected_hint_level"]

        scores = {}
        for d in SCORE_KEYS:
            vals = [r[d] for r in p_rows if r.get(d) is not None]
            scores[d] = round(sum(vals) / len(vals), 3) if vals else None
        profile_scores[pname] = scores

        summary["profiles"][pname] = {
            "label": label,
            "proficiency": prof_score,
            "expected_hint_level": hint,
            "n": len(p_rows),
            "scores": scores,
        }

        def fmt(dim):
            v = scores.get(dim)
            return f"{v:.2f}" if v is not None else "N/A"

        print(f"  {label:<10} {prof_score:>6.2f} {hint:>5} "
              f"{fmt('accuracy'):>6} {fmt('pedagogical_guidance'):>6} "
              f"{fmt('progressive_disclosure'):>6} {fmt('level_adaptation'):>6} "
              f"{fmt('completeness'):>6}")

    # 差异分析
    if "strong" in profile_scores and "weak" in profile_scores:
        print(f"\n  强学生 vs 弱学生 差异:")
        s = profile_scores["strong"]
        w = profile_scores["weak"]
        for d in SCORE_KEYS:
            sv = s.get(d)
            wv = w.get(d)
            if sv is not None and wv is not None:
                delta = sv - wv
                print(f"    {dim_zh[d]:<8}: 强={sv:.2f}  弱={wv:.2f}  Δ={delta:+.3f}")

    # 按题型分组
    print(f"\n{'='*75}")
    print("按题型分组（强学生 vs 弱学生 引导性对比）")
    print(f"{'='*75}")
    for cat in ["LAB_TROUBLESHOOTING", "THEORY_CONCEPT", "CONFIG_REVIEW", "CALCULATION"]:
        strong_rows = [r for r in all_rows
                       if r["profile"] == "strong" and r["category"] == cat
                       and r.get("pedagogical_guidance") is not None]
        weak_rows = [r for r in all_rows
                     if r["profile"] == "weak" and r["category"] == cat
                     and r.get("pedagogical_guidance") is not None]
        if strong_rows and weak_rows:
            s_guide = sum(r["pedagogical_guidance"] for r in strong_rows) / len(strong_rows)
            w_guide = sum(r["pedagogical_guidance"] for r in weak_rows) / len(weak_rows)
            print(f"  {cat:<24}: 强={s_guide:.2f}  弱={w_guide:.2f}  Δ={s_guide-w_guide:+.2f}")

    # Hint Level 轨迹统计
    print(f"\n{'='*75}")
    print("Hint Level 轨迹")
    print(f"{'='*75}")
    for pname in profiles:
        label = STUDENT_PROFILES[pname]["label"]
        p_logs = [l for l in all_logs if l["profile"] == pname and l["hint_trajectory"]]
        if p_logs:
            trajectories = [str(l["hint_trajectory"]) for l in p_logs]
            print(f"  {label}:")
            for log in p_logs:
                traj = " → ".join(str(h) for h in log["hint_trajectory"])
                print(f"    场景{log['scenario_id']}: {traj}")

    # 保存 JSON
    summary_json = RESULTS_DIR / f"proficiency_eval_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    # 可视化
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：三类学生各维度对比（分组柱状图）
        ax1 = axes[0]
        x = np.arange(len(SCORE_KEYS))
        width = 0.25
        colors = {"strong": "#2ecc71", "medium": "#f39c12", "weak": "#e74c3c"}
        labels = {"strong": "强学生", "medium": "中等学生", "weak": "弱学生"}

        for i, pname in enumerate(profiles):
            vals = [profile_scores[pname].get(d, 0) or 0 for d in SCORE_KEYS]
            bars = ax1.bar(x + i * width, vals, width,
                           label=labels[pname], color=colors[pname], alpha=0.8)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                             f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        ax1.set_xticks(x + width)
        ax1.set_xticklabels([dim_zh[d] for d in SCORE_KEYS], fontsize=9)
        ax1.set_ylabel("平均分 (1-5)")
        ax1.set_title("不同水平学生的教学效果对比")
        ax1.legend()
        ax1.set_ylim(0, 5.8)
        ax1.grid(axis="y", alpha=0.3)

        # 右图：Hint Level 起始对比（箱线图样式）
        ax2 = axes[1]
        for i, pname in enumerate(profiles):
            p_logs = [l for l in all_logs if l["profile"] == pname and l["hint_trajectory"]]
            if not p_logs:
                continue
            initial_levels = [l["hint_trajectory"][0] for l in p_logs]
            final_levels = [l["hint_trajectory"][-1] for l in p_logs]
            ax2.scatter([i] * len(initial_levels), initial_levels,
                        color=colors[pname], s=80, zorder=3, label=f"{labels[pname]} 初始")
            ax2.scatter([i + 0.15] * len(final_levels), final_levels,
                        color=colors[pname], s=80, marker="^", alpha=0.5, zorder=3)
            # 连线
            for init, final in zip(initial_levels, final_levels):
                ax2.plot([i, i + 0.15], [init, final],
                         color=colors[pname], alpha=0.3, linewidth=1)

        ax2.set_xticks(range(len(profiles)))
        ax2.set_xticklabels([labels[p] for p in profiles])
        ax2.set_ylabel("Hint Level")
        ax2.set_title("初始 Hint Level 与最终 Level\n(●=初始, ▲=最终)")
        ax2.set_ylim(-0.3, 3.5)
        ax2.set_yticks([0, 1, 2, 3])
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("水平自适应模块评估", fontsize=14, y=1.02)
        fig.tight_layout()
        png_path = RESULTS_DIR / f"proficiency_eval_{timestamp}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"图表已保存：{png_path}")

    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
