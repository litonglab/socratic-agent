"""
Agent vs 裸 DeepSeek — 以问题解决为目标的多轮交互对比

对每道题目：
  - Agent 侧：持续多轮对话，用 LLM 模拟学生回应 Agent 的 Socratic 引导，
              直到模拟学生确认理解（【SOLVED】标记）或达到最大轮数。
  - DeepSeek 侧：单轮直接给出完整答案（无引导）。

对比重点：
  - Agent 的引导路径（几轮解决、Hint Level 变化、工具调用）
  - DeepSeek 的直接回答（是否准确、是否有课程针对性）

用法：
  python eval/interactive_comparison.py

输出：
  - 终端彩色展示
  - eval/results/interactive_YYYYMMDD_HHMMSS.md
  - eval/results/interactive_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from agentic_rag.agent import query
from agentic_rag.llm_config import build_chat_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── ANSI 颜色 ────────────────────────────────────────────────────────────────
R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
BLUE = "\033[34m"; MAGENTA = "\033[35m"

MAX_TURNS = 5          # 每道题最多对话轮数
SOLVED_MARKER = "【SOLVED】"

# ── 题目集 ────────────────────────────────────────────────────────────────────
QUESTIONS = [
    {
        "id": 1,
        "category": "实验排错",
        "question": (
            "在实验1中，我已经配置了正确的IP地址和子网掩码，"
            "但两台PC机之间仍无法ping通，可能是什么原因？"
        ),
        "reference": (
            "网线连接错误或松动；Windows防火墙未开放ICMPv4回显请求；"
            "多网卡场景下ping包从非目标网卡发出。"
        ),
    },
    {
        "id": 2,
        "category": "理论概念",
        "question": (
            "在实验11的ARP协议分析中，"
            "为什么实验开始前需要清空两台PC的ARP缓存？"
        ),
        "reference": (
            "清空ARP缓存可确保实验不受之前缓存数据干扰，"
            "使我们能观察到完整的ARP请求与响应过程，保证实验的准确性与可重复性。"
        ),
    },
    {
        "id": 3,
        "category": "配置操作",
        "question": (
            "如何在H3C三层交换机上创建VLAN 2，"
            "并将GigabitEthernet 1/0/1端口加入该VLAN？"
        ),
        "reference": (
            "sys → sys SW1 → vlan 2 → "
            "port GigabitEthernet 1/0/1 → quit"
        ),
    },
    {
        "id": 4,
        "category": "无关问题（守卫测试）",
        "question": "今天北京的天气怎么样？",
        "reference": (
            "预期：Agent 拒绝回答并引导回课程主题；裸 DeepSeek 直接回答。"
        ),
    },
]

# ── 学生模拟器系统提示 ──────────────────────────────────────────────────────
STUDENT_SYSTEM = """\
你是一个正在上计算机网络实验课的大学生，刚遇到一个问题，正在向 AI 助教求助。

【行为准则】
1. 第 1 轮：表现出困惑。对助教的引导性问题给出不完整或略有偏差的猜测，
   不要直接说"不知道"，要有一些思考（即使不太对）。
2. 第 2~3 轮：结合助教给出的线索，回答越来越接近正确答案，
   但可能还有一两处细节不清楚，继续追问。
3. 第 4 轮起：如果助教的回答已经让你完全理解了原始问题，
   在回复最后加上 "【SOLVED】" 标记；否则继续追问。
4. 如果助教要求你查看某个参数或执行某个命令，给出一个合理的"虚构"结果。
5. 回复控制在 1~3 句话，语气自然，像真实学生一样。
6. 如果助教拒绝回答（如无关问题），理解并转向相关话题，在回复末尾加 "【SOLVED】"。\
"""


# ── 核心函数 ──────────────────────────────────────────────────────────────────

def simulate_student(
    original_question: str,
    turn_num: int,
    agent_response: str,
    agent_history: list,
    llm,
) -> str:
    """调用 LLM 模拟学生回复 Agent 的引导。"""
    # 构建上下文
    ctx_parts = [
        f"原始问题：{original_question}",
        f"当前轮次：第 {turn_num} 轮（共最多 {MAX_TURNS} 轮）",
        "",
    ]
    if agent_history:
        ctx_parts.append("最近对话历史：")
        for msg in agent_history[-6:]:
            role = "学生" if isinstance(msg, HumanMessage) else "助教"
            snippet = msg.content[:120].replace("\n", " ")
            ctx_parts.append(f"  {role}：{snippet}…")
        ctx_parts.append("")
    ctx_parts.append(f"助教最新回复：\n{agent_response[:500]}")

    messages = [
        SystemMessage(content=STUDENT_SYSTEM),
        HumanMessage(content="\n".join(ctx_parts)),
    ]
    try:
        resp = llm.invoke(messages)
        return resp.content.strip()
    except Exception as e:
        return f"（模拟学生失败：{e}）"


def run_agent_multiturn(original_question: str, llm) -> dict:
    """
    多轮运行 Agent，模拟学生持续响应，直到：
      - 模拟学生回复含 SOLVED_MARKER，或
      - 达到 MAX_TURNS
    """
    history: list = []
    state: dict = {}
    turns: list = []
    solved = False

    student_msg = original_question  # 第一轮直接发原始问题

    for turn_num in range(1, MAX_TURNS + 1):
        # Agent 回复
        t0 = time.time()
        try:
            agent_ans, history, tool_traces, state = query(
                student_msg, history=history, state=state
            )
        except Exception as e:
            agent_ans = f"[Agent 调用失败: {e}]"
            tool_traces = []
        agent_t = round(time.time() - t0, 1)

        hint_level = state.get("hint_level", 0)
        category   = state.get("question_category", "?")

        turn_record = {
            "turn": turn_num,
            "student_input": student_msg,
            "agent_response": agent_ans,
            "hint_level": hint_level,
            "category": category,
            "tool_count": len(tool_traces),
            "time_s": agent_t,
        }

        # 生成下一轮学生回复（最后一轮不需要）
        if turn_num < MAX_TURNS:
            next_student = simulate_student(
                original_question, turn_num + 1, agent_ans, history, llm
            )
            turn_record["student_followup"] = next_student
            turns.append(turn_record)

            if SOLVED_MARKER in next_student:
                solved = True
                break
            student_msg = next_student
        else:
            turn_record["student_followup"] = ""
            turns.append(turn_record)

    return {
        "turns": turns,
        "total_turns": len(turns),
        "solved": solved,
        "final_hint_level": state.get("hint_level", 0),
        "total_time_s": round(sum(t["time_s"] for t in turns), 1),
    }


def call_deepseek(question: str, llm) -> Tuple[str, float]:
    """单轮调用裸 DeepSeek（无 RAG / 无 Socratic 策略）。"""
    SYSTEM = "你是计算机网络实验课助教，请清晰、完整地回答学生的问题。"
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
    t0 = time.time()
    try:
        resp = llm.invoke(messages)
        return resp.content, round(time.time() - t0, 1)
    except Exception as e:
        return f"[DeepSeek 调用失败: {e}]", 0.0


# ── 显示工具 ──────────────────────────────────────────────────────────────────

def divider(char="─", width=80, color=R):
    print(f"{color}{char * width}{R}")


def hint_bar(level: int) -> str:
    return f"[{'█' * level}{'░' * (3 - level)}] L{level}"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_all():
    print(f"\n{BOLD}{CYAN}{'=' * 80}{R}")
    print(f"{BOLD}{CYAN}   Agent vs 裸 DeepSeek — 问题驱动多轮交互对比{R}")
    print(f"{BOLD}{CYAN}{'=' * 80}{R}\n")

    print("初始化 LLM 客户端...", end=" ", flush=True)
    llm = build_chat_llm(temperature=0)
    print("完成。\n")

    all_results = []

    for q in QUESTIONS:
        divider("═", color=CYAN)
        print(f"\n{BOLD}题目 {q['id']} / {len(QUESTIONS)}  [{q['category']}]{R}")
        print(f"{BOLD}问题：{R}{q['question']}\n")

        # ── Agent 多轮 ──────────────────────────────────────────────────────
        print(f"{GREEN}► Agent 多轮引导（最多 {MAX_TURNS} 轮）...{R}")
        agent_result = run_agent_multiturn(q["question"], llm)

        solved_str = f"{MAGENTA}✅ 已解决{R}" if agent_result["solved"] else "⏱ 达到轮次上限"
        print(
            f"\n{BOLD}{GREEN}[Agent 路径]{R}  "
            f"{agent_result['total_turns']} 轮  |  "
            f"最终 Hint Level {hint_bar(agent_result['final_hint_level'])}  |  "
            f"{solved_str}  |  ⏱ {agent_result['total_time_s']}s 合计"
        )
        divider(color=DIM)
        prev_level = 0
        for t in agent_result["turns"]:
            level_change = (
                f"  {MAGENTA}↑ {prev_level}→{t['hint_level']}{R}"
                if t["hint_level"] > prev_level else ""
            )
            print(
                f"\n{BOLD}{YELLOW}  轮 {t['turn']}  {hint_bar(t['hint_level'])}"
                f"{level_change}  🔧 {t['tool_count']}次工具  ⏱ {t['time_s']}s{R}"
            )
            print(f"{BOLD}  学生：{R}{t['student_input']}")
            print(f"{BOLD}{GREEN}  Agent：{R}{t['agent_response'].strip()}")
            if t.get("student_followup"):
                followup = t["student_followup"].replace(
                    SOLVED_MARKER, f"{MAGENTA}【问题已解决】{R}"
                )
                print(f"{DIM}  ↳ 学生后续：{R}{followup}")
            prev_level = t["hint_level"]

        # ── DeepSeek 单轮 ───────────────────────────────────────────────────
        print(f"\n{BLUE}► 裸 DeepSeek 单轮回答...{R}", end=" ", flush=True)
        ds_ans, ds_t = call_deepseek(q["question"], llm)
        print(f"完成 ({ds_t}s)")
        print(f"\n{BOLD}{BLUE}[裸 DeepSeek 回答]  ⏱ {ds_t}s  （1 轮）{R}")
        print(ds_ans.strip())

        # ── 参考答案 ────────────────────────────────────────────────────────
        print(f"\n{BOLD}{YELLOW}[参考答案]{R}  {q['reference']}")
        print()

        all_results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "reference": q["reference"],
            "agent": agent_result,
            "deepseek": {"answer": ds_ans, "time_s": ds_t, "turns": 1},
        })

    _save_report(all_results)
    _print_summary(all_results)


def _print_summary(results: list):
    divider("═", color=CYAN)
    print(f"{BOLD}{CYAN}  运行完成 — 对比摘要{R}")
    divider("═", color=CYAN)
    print(f"  {'题目':<30} {'Agent 轮数':>10} {'Hint L':>8} {'状态':>10}  {'DeepSeek':>10}")
    divider(color=DIM)
    for r in results:
        a = r["agent"]
        solved_mark = "✅" if a["solved"] else "⏱"
        print(
            f"  {r['category']:<30} {a['total_turns']:>6} 轮    "
            f"L{a['final_hint_level']:>3}   {solved_mark}          "
            f"{r['deepseek']['turns']:>4} 轮"
        )
    print()


def _save_report(results: list):
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── JSON ────────────────────────────────────────────────────────────────
    json_path = out_dir / f"interactive_{ts}.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Markdown ────────────────────────────────────────────────────────────
    md_path = out_dir / f"interactive_{ts}.md"
    lines = [
        "# Agent vs 裸 DeepSeek — 问题驱动多轮交互对比",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 说明",
        "",
        "- **Agent**：RAG 检索 + Socratic 引导 + 状态跟踪，用 LLM 模拟学生持续响应直到问题解决",
        "- **裸 DeepSeek**：相同助教身份，单轮直接给出完整答案",
        "",
        "---",
        "",
    ]
    for r in results:
        a = r["agent"]
        d = r["deepseek"]
        bar = "█" * a["final_hint_level"] + "░" * (3 - a["final_hint_level"])
        lines += [
            f"## 题目 {r['id']}：{r['category']}",
            "",
            f"**问题：** {r['question']}",
            "",
            f"**参考答案：** {r['reference']}",
            "",
            f"### Agent 对话路径  "
            f"（{a['total_turns']} 轮 · 最终 Hint [{bar}] L{a['final_hint_level']} · "
            f"{'已解决' if a['solved'] else '达上限'} · ⏱ {a['total_time_s']}s）",
            "",
        ]
        prev = 0
        for t in a["turns"]:
            bar_t = "█" * t["hint_level"] + "░" * (3 - t["hint_level"])
            change = f" ↑{prev}→{t['hint_level']}" if t["hint_level"] > prev else ""
            lines += [
                f"#### 第 {t['turn']} 轮  `[{bar_t}] L{t['hint_level']}`{change}  "
                f"🔧 {t['tool_count']} 次工具  ⏱ {t['time_s']}s",
                "",
                f"**学生：** {t['student_input']}",
                "",
                "**Agent：**",
                "",
                t["agent_response"],
                "",
            ]
            if t.get("student_followup"):
                lines += [f"**学生后续：** {t['student_followup']}", ""]
            prev = t["hint_level"]
        lines += [
            f"### 裸 DeepSeek 回答  （1 轮 · ⏱ {d['time_s']}s）",
            "",
            d["answer"],
            "",
            "---",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{BOLD}{GREEN}Markdown 报告：{md_path}{R}")
    print(f"{BOLD}{GREEN}JSON 数据：    {json_path}{R}\n")


if __name__ == "__main__":
    run_all()
