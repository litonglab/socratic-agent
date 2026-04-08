"""
Agent 多轮 Socratic 引导 vs 裸 DeepSeek 单轮直答 — 案例分析脚本

对比方式：
  - Agent：多轮对话，模拟学生追问，展示 Socratic 引导 + hint_level 递进
  - 裸 DeepSeek：仅用初始问题单轮回答（无 RAG、无 Socratic、无历史）

评分：GPT-4o 对 Agent 最终轮和 DeepSeek 单轮分别打分

用法：
  python eval/demo_comparison.py                    # 运行全部场景
  python eval/demo_comparison.py --scenarios 1,2,3  # 只跑指定场景

输出：
  - 终端彩色对比
  - eval/results/demo_YYYYMMDD_HHMMSS.md   — Markdown 案例分析报告
  - eval/results/demo_YYYYMMDD_HHMMSS.json — 结构化数据
"""

import os
import sys
import json
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

# ── ANSI 颜色 ────────────────────────────────────────────────────────────────
R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; BLUE = "\033[34m"

# ── 多轮对话场景（模拟学生追问）─────────────────────────────────────────────
SCENARIOS = [
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
        "name": "ARP协议理解",
        "category": "THEORY_CONCEPT",
        "turns": [
            "在实验11里，ARP协议是用来干什么的？我觉得它跟DNS一样，都是把名字转成地址。",
            "哦，那是把IP地址转成MAC地址吗？但为什么需要MAC地址？",
        ],
    },
    {
        "id": 4,
        "name": "H3C VLAN配置",
        "category": "CONFIG_REVIEW",
        "turns": [
            "我要在H3C三层交换机上配置VLAN 2，第一步应该输入什么命令？",
            "我输入了sys进入了系统视图，然后应该怎么创建VLAN 2？",
            "我已经用vlan 2创建好了，那怎么把端口GigabitEthernet 1/0/1加进去？",
        ],
    },
    {
        "id": 5,
        "name": "子网划分计算",
        "category": "CALCULATION",
        "turns": [
            "在实验13中，把172.16.0.0/16划分成4个子网，每个子网的地址范围是多少？",
            "我算出子网掩码是/18，但不太确定每个子网的起始地址怎么算。",
        ],
    },
    {
        "id": 6,
        "name": "无关问题（守卫测试）",
        "category": "GUARD",
        "turns": [
            "今天北京的天气怎么样？",
        ],
    },
]

BASELINE_SYSTEM = "你是一个计算机网络实验课程的助教，请直接、完整地回答学生的问题。"

# ── GPT-4o 案例打分 ──────────────────────────────────────────────────────────
JUDGE_SYSTEM = """\
你是教育学和计算机网络双重专家。请从教学效果角度评估 AI 助教的回答质量。
你的打分必须严格、一致，根据锚点描述打分。"""

JUDGE_TEMPLATE = """\
请评估以下 AI 助教的回答。

【学生初始问题】
{initial_question}

【完整对话过程】
{conversation}

─── 评分标准（每项 1-5 分整数）──────────────────────
1. 内容准确性（accuracy）
   5=技术内容完全正确  4=基本正确，有1处小瑕疵  3=有1-2处明显错误  2=多处错误  1=严重错误

2. 引导性（pedagogical_guidance）
   5=通过提问/类比/反问引导学生主动思考，不直接给完整答案
   4=有引导意识，但部分内容直接给出  3=以回答为主，偶有引导
   2=基本是直接回答  1=完全直接给答案

3. 信息递进性（progressive_disclosure）
   5=严格控制信息量，按学生理解程度逐步揭示
   4=有递进意识  3=信息量适中但未充分利用历史
   2=一次性给出过多信息  1=无视上下文

4. 回答完整性（completeness）
   5=在当前引导层级下内容充分  4=大部分关键信息已覆盖
   3=覆盖约60%关键信息  2=仅覆盖少数  1=几乎未覆盖

────────────────────────────────────────────────────

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "accuracy": <1-5>,
  "pedagogical_guidance": <1-5>,
  "progressive_disclosure": <1-5>,
  "completeness": <1-5>,
  "comment": "<评分简要理由，不超过 80 字>"
}}"""

SCORE_KEYS = ["accuracy", "pedagogical_guidance", "progressive_disclosure", "completeness"]


def judge_quality(client: OpenAI, initial_question: str, conversation: str) -> Dict:
    user_msg = JUDGE_TEMPLATE.format(
        initial_question=initial_question,
        conversation=conversation,
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


# ── 格式化工具 ───────────────────────────────────────────────────────────────

def fmt_tool_traces(tool_traces: list) -> str:
    if not tool_traces:
        return "未调用工具"
    return "、".join(
        f"{t['tool']}（{t['input'][:40].strip()}...）" for t in tool_traces
    )


def divider(char="─", width=80, color=R):
    print(f"{color}{char * width}{R}")


# ── 主流程 ───────────────────────────────────────────────────────────────────

def run_demo(scenarios: List[Dict]):
    print(f"\n{BOLD}{CYAN}{'=' * 80}{R}")
    print(f"{BOLD}{CYAN}   Agent 多轮引导 vs 裸 DeepSeek 单轮直答 — 案例分析{R}")
    print(f"{BOLD}{CYAN}{'=' * 80}{R}\n")

    llm = build_chat_llm(temperature=0)
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_key) if openai_key else None
    if not openai_client:
        print(f"{YELLOW}[提示] 未设置 OPENAI_API_KEY，将跳过 GPT-4o 打分{R}\n")

    results = []

    for scenario in scenarios:
        sid = scenario["id"]
        divider("═", color=CYAN)
        print(f"{BOLD}场景 {sid} / {len(scenarios)}  [{scenario['category']}] {scenario['name']}{R}")
        initial_question = scenario["turns"][0]
        print(f"{BOLD}初始问题：{R}{initial_question}\n")

        # ── Agent 多轮对话 ────────────────────────────────
        print(f"{GREEN}{BOLD}▶ Agent 多轮 Socratic 引导{R}")
        agent_history = []
        agent_state = {}
        agent_turns = []
        agent_total_time = 0

        for ti, student_msg in enumerate(scenario["turns"], 1):
            print(f"  第{ti}轮 学生：{student_msg[:50]}...", end=" ", flush=True)
            t0 = time.time()
            try:
                ans, agent_history, tool_traces, agent_state = agent_query(
                    student_msg, history=agent_history, state=agent_state
                )
            except Exception as e:
                ans = f"[Agent 调用失败: {e}]"
                tool_traces = []
            elapsed = round(time.time() - t0, 1)
            agent_total_time += elapsed
            hint = agent_state.get("hint_level", 0)
            cat = agent_state.get("question_category", "?")
            print(f"({elapsed}s, hint={hint})")

            agent_turns.append({
                "turn": ti,
                "student": student_msg,
                "answer": ans,
                "hint_level": hint,
                "category": cat,
                "tool_traces": tool_traces,
                "time_s": elapsed,
                "answer_length": len(ans),
            })

        # ── 裸 DeepSeek 单轮直答 ─────────────────────────
        print(f"\n{BLUE}{BOLD}▶ 裸 DeepSeek 单轮直答{R}")
        print(f"  问题：{initial_question[:50]}...", end=" ", flush=True)
        t0 = time.time()
        try:
            ds_resp = llm.invoke([
                SystemMessage(content=BASELINE_SYSTEM),
                HumanMessage(content=initial_question),
            ])
            ds_ans = ds_resp.content
        except Exception as e:
            ds_ans = f"[DeepSeek 调用失败: {e}]"
        ds_time = round(time.time() - t0, 1)
        print(f"({ds_time}s, {len(ds_ans)}字)\n")

        # ── GPT-4o 打分 ──────────────────────────────────
        agent_scores, ds_scores = {k: None for k in SCORE_KEYS}, {k: None for k in SCORE_KEYS}
        if openai_client:
            # Agent: 评估完整多轮对话
            agent_conv = "\n".join(
                f"【第{t['turn']}轮】学生：{t['student']}\nAgent：{t['answer'][:500]}"
                for t in agent_turns
            )
            print(f"  GPT-4o 评分 Agent...", end=" ", flush=True)
            agent_scores = judge_quality(openai_client, initial_question, agent_conv)
            print(f"引导性={agent_scores.get('pedagogical_guidance', '?')}")

            # DeepSeek: 评估单轮回答
            ds_conv = f"学生：{initial_question}\nDeepSeek：{ds_ans[:1500]}"
            print(f"  GPT-4o 评分 DeepSeek...", end=" ", flush=True)
            ds_scores = judge_quality(openai_client, initial_question, ds_conv)
            print(f"引导性={ds_scores.get('pedagogical_guidance', '?')}")

        # ── 终端输出对比 ──────────────────────────────────
        print()
        for t in agent_turns:
            print(f"  {GREEN}[Agent 第{t['turn']}轮 | hint={t['hint_level']} | {t['answer_length']}字]{R}")
            print(f"  学生：{t['student']}")
            print(f"  回答：{t['answer'][:200]}...\n")

        print(f"  {BLUE}[DeepSeek 单轮 | {len(ds_ans)}字]{R}")
        print(f"  问题：{initial_question}")
        print(f"  回答：{ds_ans[:200]}...\n")

        hint_trajectory = [t["hint_level"] for t in agent_turns]
        results.append({
            "id": sid,
            "name": scenario["name"],
            "category": scenario["category"],
            "initial_question": initial_question,
            "agent": {
                "turns": agent_turns,
                "total_time_s": round(agent_total_time, 1),
                "hint_trajectory": hint_trajectory,
                "final_hint_level": hint_trajectory[-1],
                "scores": agent_scores,
                "total_length": sum(t["answer_length"] for t in agent_turns),
            },
            "deepseek": {
                "answer": ds_ans,
                "time_s": ds_time,
                "answer_length": len(ds_ans),
                "scores": ds_scores,
            },
        })

        time.sleep(0.3)

    _save_report(results)
    _print_summary(results)
    return results


# ── 保存 Markdown 案例报告 ───────────────────────────────────────────────────

def _save_report(results: list):
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"demo_{ts}.md"
    json_path = out_dir / f"demo_{ts}.json"

    lines = [
        "# Agent 多轮引导 vs 裸 DeepSeek 单轮直答 — 案例分析报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "对比方式：Agent 进行多轮 Socratic 引导对话，裸 DeepSeek 仅对初始问题做单轮回答。",
        "",
    ]

    # 汇总表格
    lines += [
        "## 总览",
        "",
        "| 场景 | 类型 | Agent轮数 | Agent总字数 | DeepSeek字数 | Hint轨迹 | Agent引导性 | DS引导性 |",
        "|------|------|:---------:|:-----------:|:------------:|----------|:-----------:|:--------:|",
    ]
    for r in results:
        a = r["agent"]
        d = r["deepseek"]
        ht = " → ".join(str(h) for h in a["hint_trajectory"])
        a_guide = a["scores"].get("pedagogical_guidance", "-")
        d_guide = d["scores"].get("pedagogical_guidance", "-")
        lines.append(
            f"| {r['name']} | {r['category']} | {len(a['turns'])} | "
            f"{a['total_length']} | {d['answer_length']} | {ht} | {a_guide} | {d_guide} |"
        )
    lines += ["", "---", ""]

    # 逐场景详情
    for r in results:
        a = r["agent"]
        d = r["deepseek"]

        lines += [
            f"## 场景 {r['id']}：{r['name']}（{r['category']}）",
            "",
            f"**初始问题：** {r['initial_question']}",
            "",
            f"### Agent 多轮引导  ⏱ {a['total_time_s']}s | Hint: {' → '.join(str(h) for h in a['hint_trajectory'])}",
            "",
        ]

        # Agent 评分
        if any(v is not None for v in a["scores"].values()):
            lines.append(f"> 评分：准确性={a['scores'].get('accuracy', '-')} | "
                         f"引导性={a['scores'].get('pedagogical_guidance', '-')} | "
                         f"递进性={a['scores'].get('progressive_disclosure', '-')} | "
                         f"完整性={a['scores'].get('completeness', '-')}")
            if a["scores"].get("comment"):
                lines.append(f"> 评语：{a['scores']['comment']}")
            lines.append("")

        for t in a["turns"]:
            tools = fmt_tool_traces(t["tool_traces"])
            lines += [
                f"**第{t['turn']}轮**（hint={t['hint_level']}，{t['answer_length']}字，工具：{tools}）",
                "",
                f"**学生：** {t['student']}",
                "",
                f"**Agent：** {t['answer']}",
                "",
            ]

        lines += [
            f"### 裸 DeepSeek 单轮直答  ⏱ {d['time_s']}s | {d['answer_length']}字",
            "",
        ]

        if any(v is not None for v in d["scores"].values()):
            lines.append(f"> 评分：准确性={d['scores'].get('accuracy', '-')} | "
                         f"引导性={d['scores'].get('pedagogical_guidance', '-')} | "
                         f"递进性={d['scores'].get('progressive_disclosure', '-')} | "
                         f"完整性={d['scores'].get('completeness', '-')}")
            if d["scores"].get("comment"):
                lines.append(f"> 评语：{d['scores']['comment']}")
            lines.append("")

        lines += [
            f"**问题：** {r['initial_question']}",
            "",
            f"**DeepSeek：** {d['answer']}",
            "",
            "---",
            "",
        ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{BOLD}{GREEN}Markdown 报告：{md_path}{R}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{BOLD}{GREEN}JSON 数据：{json_path}{R}")


# ── 终端汇总 ─────────────────────────────────────────────────────────────────

def _print_summary(results: list):
    import numpy as np

    print(f"\n{BOLD}{CYAN}{'=' * 80}{R}")
    print(f"{BOLD}{CYAN}   汇总统计{R}")
    print(f"{BOLD}{CYAN}{'=' * 80}{R}\n")

    # 评分对比
    has_scores = any(
        r["agent"]["scores"].get("accuracy") is not None for r in results
    )
    if has_scores:
        print(f"  {'维度':<20} {'Agent':>10} {'DeepSeek':>10} {'差值':>10}")
        print(f"  {'─' * 52}")
        for dim in SCORE_KEYS:
            a_vals = [r["agent"]["scores"][dim] for r in results
                      if r["agent"]["scores"].get(dim) is not None]
            d_vals = [r["deepseek"]["scores"][dim] for r in results
                      if r["deepseek"]["scores"].get(dim) is not None]
            if a_vals and d_vals:
                a_mean = np.mean(a_vals)
                d_mean = np.mean(d_vals)
                print(f"  {dim:<20} {a_mean:>10.2f} {d_mean:>10.2f} {a_mean - d_mean:>+10.2f}")

    # 字数对比
    print(f"\n  {'指标':<20} {'Agent':>10} {'DeepSeek':>10}")
    print(f"  {'─' * 42}")
    a_lens = [r["agent"]["total_length"] for r in results]
    d_lens = [r["deepseek"]["answer_length"] for r in results]
    print(f"  {'平均总字数':<18} {np.mean(a_lens):>10.0f} {np.mean(d_lens):>10.0f}")
    print(f"  {'平均耗时(s)':<18} "
          f"{np.mean([r['agent']['total_time_s'] for r in results]):>10.1f} "
          f"{np.mean([r['deepseek']['time_s'] for r in results]):>10.1f}")
    print()


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agent多轮 vs DeepSeek单轮 案例分析")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="要运行的场景 ID（逗号分隔，默认全部）")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenarios:
        ids = set(int(x) for x in args.scenarios.split(","))
        scenarios = [s for s in SCENARIOS if s["id"] in ids]

    print(f"将运行 {len(scenarios)} 个场景\n")
    run_demo(scenarios)


if __name__ == "__main__":
    main()
