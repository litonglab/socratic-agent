"""
Agent vs 裸 DeepSeek — 多轮 Socratic 对话对比脚本

模拟真实学生与 Agent 的多轮交互，对比：
  - Agent：RAG 检索 + 4 级 Socratic Hint 渐进引导 + 状态跟踪
  - 裸 DeepSeek：有对话历史，但直接给答案，无 Socratic 策略

保存完整对话记录，方便人工查看。

用法：
  python eval/socratic_comparison.py

输出：
  - 终端实时展示（含 Hint Level 变化）
  - eval/results/socratic_YYYYMMDD_HHMMSS.md（Markdown 对话记录）
  - eval/results/socratic_YYYYMMDD_HHMMSS.json（JSON 完整数据）
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# ── Socratic 对话场景 ─────────────────────────────────────────────────────────
# 每个场景模拟一名真实学生，从困惑→部分理解→解决的 3 轮交互。
# 设计原则：
#   Turn 1：宽泛提问，期待 Agent 引导而非直接给答案
#   Turn 2：学生给出错误/不完整的回答，看 Agent 是否仍然引导
#   Turn 3：学生接近答案，看 Agent 是否收敛并给出完整解法
SCENARIOS = [
    {
        "id": 1,
        "name": "实验排错 — PC 之间 ping 不通",
        "description": (
            "学生配置了 IP 但 ping 不通，逐步排查故障。"
            "重点对比：Agent 引导学生自己排查 vs DeepSeek 直接罗列所有原因。"
        ),
        "turns": [
            # Turn 1：宽泛报错
            "我在实验1里，两台 PC 的 IP 地址都配好了，但是 ping 不通，怎么回事？",
            # Turn 2：学生猜测错误方向（网线问题）
            "我检查了一下，网线应该没问题，插得很紧。",
            # Turn 3：学生提到防火墙但不知道怎么设置
            "你说的防火墙，我打开看了但不知道要改哪里，能告诉我具体步骤吗？",
        ],
    },
    {
        "id": 2,
        "name": "理论概念 — ARP 协议理解偏差",
        "description": (
            "学生把 ARP 误认为和 DNS 类似（名字→IP），"
            "Agent 通过引导性问题纠正认知，"
            "DeepSeek 直接纠正并给出完整解释。"
        ),
        "turns": [
            # Turn 1：带有错误前提的提问
            "在实验11里，ARP 协议是用来干什么的？我觉得它跟 DNS 一样，都是把名字转成地址。",
            # Turn 2：学生被引导后给出了部分正确答案
            "哦，那是把 IP 地址转成 MAC 地址吗？",
            # Turn 3：学生追问实验细节
            "那为什么实验开始前要先清空 ARP 缓存？不清空会怎样？",
        ],
    },
    {
        "id": 3,
        "name": "配置操作 — H3C 交换机 VLAN 配置",
        "description": (
            "学生第一次配置 H3C 交换机，不知道从哪里开始。"
            "重点对比：Agent 逐步引导命令顺序 vs DeepSeek 一次给出完整命令序列。"
        ),
        "turns": [
            # Turn 1：完全不知道怎么做
            "我要在 H3C 三层交换机上配置 VLAN 2，第一步应该输入什么命令？",
            # Turn 2：进入了系统视图，但下一步不确定
            "我输入了 sys 进入了系统视图，然后应该怎么创建 VLAN 2？",
            # Turn 3：创建了 VLAN，不知道怎么加端口
            "我已经用 vlan 2 创建好了，那怎么把端口 GigabitEthernet 1/0/1 加进去？",
        ],
    },
]

# 裸 DeepSeek 的系统提示：有助教身份，但无 Socratic 策略
BASELINE_SYSTEM = "你是一个计算机网络实验课程的助教，请清晰、完整地回答学生的问题。"


# ── 调用函数 ─────────────────────────────────────────────────────────────────

def call_agent_turn(
    question: str,
    history: list,
    state: dict,
) -> Tuple[str, list, dict, list]:
    """单轮调用 Agent，返回 (答案, 新history, 新state, tool_traces)。"""
    try:
        answer, new_history, tool_traces, new_state = query(
            question, history=history, state=state
        )
        return answer, new_history, new_state, tool_traces
    except Exception as e:
        err = f"[Agent 调用失败: {e}]"
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=err))
        return err, history, state, []


def call_deepseek_turn(
    question: str,
    history: list,
    llm,
) -> Tuple[str, list]:
    """单轮调用裸 DeepSeek（带历史，无 RAG / 无 Socratic 策略）。"""
    messages = [SystemMessage(content=BASELINE_SYSTEM)] + history + [
        HumanMessage(content=question)
    ]
    try:
        resp = llm.invoke(messages)
        new_history = history + [
            HumanMessage(content=question),
            AIMessage(content=resp.content),
        ]
        return resp.content, new_history
    except Exception as e:
        err = f"[DeepSeek 调用失败: {e}]"
        new_history = history + [
            HumanMessage(content=question),
            AIMessage(content=err),
        ]
        return err, new_history


# ── 格式化工具 ───────────────────────────────────────────────────────────────

def divider(char="─", width=80, color=R):
    print(f"{color}{char * width}{R}")


def hint_level_bar(level: int) -> str:
    """用方块可视化当前 Hint Level（0~3）。"""
    filled = "█" * level
    empty = "░" * (3 - level)
    return f"[{filled}{empty}] Level {level}/3"


# ── 场景执行 ─────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, llm) -> dict:
    divider("═", color=CYAN)
    print(f"\n{BOLD}{CYAN}  场景 {scenario['id']}：{scenario['name']}{R}")
    print(f"{DIM}  {scenario['description']}{R}\n")

    agent_history: list = []
    agent_state: dict = {}
    ds_history: list = []

    log = {
        "id": scenario["id"],
        "name": scenario["name"],
        "description": scenario["description"],
        "turns": [],
    }

    prev_hint_level = 0

    for i, student_msg in enumerate(scenario["turns"], 1):
        divider(color=DIM)
        print(f"\n{BOLD}{YELLOW}  ── 第 {i} 轮 / 共 {len(scenario['turns'])} 轮 ──{R}")
        print(f"{BOLD}  学生：{R}{student_msg}\n")

        # Agent
        print(f"  {GREEN}► Agent 思考中...{R}", end=" ", flush=True)
        t0 = time.time()
        agent_ans, agent_history, agent_state, tool_traces = call_agent_turn(
            student_msg, agent_history, agent_state
        )
        agent_t = round(time.time() - t0, 1)
        hint_level = agent_state.get("hint_level", 0)
        category = agent_state.get("question_category", "?")
        level_change = (
            f"{MAGENTA}↑ {prev_hint_level}→{hint_level}{R}"
            if hint_level > prev_hint_level
            else ""
        )
        print(f"完成 ({agent_t}s)")

        # DeepSeek
        print(f"  {BLUE}► 裸 DeepSeek 思考中...{R}", end=" ", flush=True)
        t0 = time.time()
        ds_ans, ds_history = call_deepseek_turn(student_msg, ds_history, llm)
        ds_t = round(time.time() - t0, 1)
        print(f"完成 ({ds_t}s)\n")

        # 打印 Agent 回答
        print(
            f"{BOLD}{GREEN}[Agent]{R} "
            f"{hint_level_bar(hint_level)} {level_change}  "
            f"分类={category}  工具={len(tool_traces)}次  ⏱ {agent_t}s"
        )
        print(agent_ans.strip())
        if tool_traces:
            calls = "、".join(
                f"{t['tool']}({t['input'][:30].strip()}...)" for t in tool_traces
            )
            print(f"{DIM}  工具调用：{calls}{R}")
        print()

        # 打印 DeepSeek 回答
        print(f"{BOLD}{BLUE}[裸 DeepSeek]{R}  ⏱ {ds_t}s")
        print(ds_ans.strip())
        print()

        # 记录
        log["turns"].append(
            {
                "turn": i,
                "student": student_msg,
                "agent": {
                    "answer": agent_ans,
                    "hint_level": hint_level,
                    "hint_level_prev": prev_hint_level,
                    "category": category,
                    "tool_traces": [
                        {"tool": t["tool"], "input": t["input"][:200]}
                        for t in tool_traces
                    ],
                    "time_s": agent_t,
                },
                "deepseek": {
                    "answer": ds_ans,
                    "time_s": ds_t,
                },
            }
        )

        prev_hint_level = hint_level

    return log


# ── 保存结果 ─────────────────────────────────────────────────────────────────

def _save_logs(all_logs: list):
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON（机器可读）
    json_path = out_dir / f"socratic_{ts}.json"
    json_path.write_text(
        json.dumps(all_logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown（人工查看）
    md_path = out_dir / f"socratic_{ts}.md"
    lines = [
        "# Agent vs 裸 DeepSeek — 多轮 Socratic 对话对比",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 说明",
        "",
        "- **Agent**：RAG 检索课程知识库 + 4 级 Socratic Hint 渐进引导 + 状态跟踪",
        "- **裸 DeepSeek**：相同助教身份，保留对话历史，但直接给答案，无 Socratic 策略",
        "",
        "---",
        "",
    ]

    for log in all_logs:
        lines += [
            f"## 场景 {log['id']}：{log['name']}",
            "",
            f"> {log['description']}",
            "",
        ]

        for t in log["turns"]:
            a = t["agent"]
            d = t["deepseek"]
            bar_filled = "█" * a["hint_level"] + "░" * (3 - a["hint_level"])

            lines += [
                f"### 第 {t['turn']} 轮",
                "",
                f"**学生：** {t['student']}",
                "",
                f"#### Agent 回答",
                f"| 指标 | 值 |",
                f"|---|---|",
                f"| Hint Level | `[{bar_filled}] {a['hint_level']}/3`（上轮：{a['hint_level_prev']}）|",
                f"| 问题分类 | `{a['category']}` |",
                f"| 工具调用 | {len(a['tool_traces'])} 次 |",
                f"| 耗时 | {a['time_s']}s |",
                "",
            ]

            if a["tool_traces"]:
                lines.append("**工具调用详情：**")
                for tr in a["tool_traces"]:
                    lines.append(f"- `{tr['tool']}`：{tr['input'][:80]}")
                lines.append("")

            lines += [
                a["answer"],
                "",
                f"#### 裸 DeepSeek 回答  ⏱ {d['time_s']}s",
                "",
                d["answer"],
                "",
                "---",
                "",
            ]

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{BOLD}{GREEN}Markdown 记录已保存：{md_path}{R}")
    print(f"{BOLD}{GREEN}JSON 数据已保存：  {json_path}{R}\n")


# ── 入口 ─────────────────────────────────────────────────────────────────────

def run_all():
    print(f"\n{BOLD}{CYAN}{'=' * 80}{R}")
    print(f"{BOLD}{CYAN}   Agent vs 裸 DeepSeek — 多轮 Socratic 对话对比{R}")
    print(f"{BOLD}{CYAN}{'=' * 80}{R}")
    print(
        f"\n共 {len(SCENARIOS)} 个场景，每场景 3 轮对话。\n"
        f"每轮同时调用 Agent 和裸 DeepSeek，记录完整对话与 Hint Level 变化。\n"
    )

    print("初始化裸 DeepSeek 客户端...", end=" ", flush=True)
    llm = build_chat_llm(temperature=0)
    print("完成。\n")

    all_logs = []
    for scenario in SCENARIOS:
        log = run_scenario(scenario, llm)
        all_logs.append(log)
        print()  # 场景间空行

    _save_logs(all_logs)

    # 终端摘要
    divider("═", color=CYAN)
    print(f"{BOLD}{CYAN}  运行完成 — Hint Level 变化摘要{R}")
    divider("═", color=CYAN)
    for log in all_logs:
        levels = [t["agent"]["hint_level"] for t in log["turns"]]
        bar = " → ".join(str(lv) for lv in levels)
        print(f"  场景 {log['id']} [{log['name']}]：Hint Level 路径 {bar}")
    print()


if __name__ == "__main__":
    run_all()
