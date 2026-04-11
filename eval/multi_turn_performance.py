"""
多轮对话工程性能基准测试

目标：
  - 测量 Agent 在高质量固定场景上的多轮对话时延
  - 统计平均单轮时延、首轮/后续轮时延、会话总时延
  - 分解各阶段耗时（前置处理 / 工具执行 / LLM 生成）
  - 分析不同题型下的会话成本与稳定性

默认设置：
  - 复用 10 个固定多轮场景
  - 每个场景重复运行 4 次，共 40 个会话样本

输出：
  eval/results/multi_turn_perf_YYYYMMDD_HHMMSS.csv
  eval/results/multi_turn_perf_sessions_YYYYMMDD_HHMMSS.csv
  eval/results/multi_turn_perf_YYYYMMDD_HHMMSS.json
  eval/results/multi_turn_perf_turn_latency_YYYYMMDD_HHMMSS.png
  eval/results/multi_turn_perf_session_cost_YYYYMMDD_HHMMSS.png
  eval/results/multi_turn_perf_stage_breakdown_YYYYMMDD_HHMMSS.png

环境变量：
  DEEPSEEK_API_KEY (必须)
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

try:
    from eval.socratic_evaluation import SCENARIOS
except Exception:
    from socratic_evaluation import SCENARIOS


RESULTS_DIR = ROOT / "eval" / "results"


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.array(values), q))


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.array(values)))


def _std(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.std(np.array(values)))


def _stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "min": None,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def warmup():
    print("预热中（加载模型与索引）...", end=" ", flush=True)
    try:
        from agentic_rag.agent import _get_client
        from agentic_rag.rag import RAGAgent

        _get_client()
        RAGAgent("测试查询", category="THEORY_CONCEPT", hint_level=0)
    except Exception:
        pass
    print("完成\n")


def benchmark_single_turn(
    question: str,
    history: List[BaseMessage],
    state: Dict[str, Any],
    *,
    enable_websearch: bool = False,
    max_agent_loops: int = 5,
) -> Dict[str, Any]:
    """对单个用户轮次进行阶段化性能测量。"""
    from agentic_rag.agent import (
        Agent,
        _execute_tool_action,
        _find_actions,
        _format_citations,
        _prepare_context,
    )

    q = (question or "").strip()
    history_messages_before = len(history)
    history_chars_before = sum(len(getattr(m, "content", "") or "") for m in history)

    t0 = time.perf_counter()
    ctx = _prepare_context(q, history, state, None, enable_websearch, False)
    classification_s = round(time.perf_counter() - t0, 3)

    if not enable_websearch:
        ctx.final_prompt += (
            "\n\n【工具可用性覆盖】\n"
            "本次性能评测未启用联网搜索工具。\n"
            "你当前只允许调用工具 `检索` 和 `拓扑`。\n"
            "如果需要工具，请使用 `<tool_calls>` JSON 数组协议，不要调用 `搜索`。"
        )

    route = "agent"
    tool_calls = 0
    internal_loops = 0
    retrieval_s = 0.0
    generation_s = 0.0
    tool_traces: List[Dict[str, Any]] = []
    last_citations: List[Dict[str, Any]] = []
    final_result = ""

    if ctx.early_reply is not None:
        route = "early_reply"
        final_result = ctx.early_reply
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=final_result))
        return {
            "status": "ok",
            "route": route,
            "answer": final_result,
            "answer_length": len(final_result),
            "classification_s": classification_s,
            "retrieval_s": 0.0,
            "generation_s": 0.0,
            "total_s": classification_s,
            "tool_calls": 0,
            "internal_loops": 0,
            "early_exit": True,
            "hint_level": state.get("hint_level"),
            "question_category": state.get("question_category"),
            "experiment_id": state.get("experiment_id"),
            "history_messages_before": history_messages_before,
            "history_chars_before": history_chars_before,
            "tool_traces": [],
            "error": "",
        }

    if ctx.use_general_llm:
        route = "general_llm"
        bot = Agent(ctx.final_prompt, history)
        t1 = time.perf_counter()
        final_result = bot(q)
        generation_s = round(time.perf_counter() - t1, 3)
        total_s = round(classification_s + generation_s, 3)
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=final_result))
        return {
            "status": "ok",
            "route": route,
            "answer": final_result,
            "answer_length": len(final_result),
            "classification_s": classification_s,
            "retrieval_s": 0.0,
            "generation_s": generation_s,
            "total_s": total_s,
            "tool_calls": 0,
            "internal_loops": 1,
            "early_exit": False,
            "hint_level": state.get("hint_level"),
            "question_category": state.get("question_category"),
            "experiment_id": state.get("experiment_id"),
            "history_messages_before": history_messages_before,
            "history_chars_before": history_chars_before,
            "tool_traces": [],
            "error": "",
        }

    bot = Agent(ctx.final_prompt, history)
    next_prompt = q

    for _ in range(max_agent_loops):
        internal_loops += 1

        t1 = time.perf_counter()
        result = bot(next_prompt)
        generation_s += time.perf_counter() - t1
        final_result = result

        action_matches = _find_actions(result)
        if action_matches:
            tool_calls += len(action_matches)
            t2 = time.perf_counter()
            observations = []
            for action_match in action_matches:
                action, action_input = action_match.groups()
                try:
                    observation = _execute_tool_action(
                        action_match,
                        ctx.contextual_actions,
                        tool_traces,
                        last_citations,
                    )
                except Exception:
                    if action == "搜索" and not enable_websearch:
                        fallback_observation = (
                            "工具：搜索："
                            f"{action_input}\n"
                            "检索结果：当前会话未启用联网搜索工具。"
                            "请仅使用 工具：检索 或 工具：拓扑，并基于已有课程材料继续回答。"
                        )
                        tool_traces.append({
                            "tool": action,
                            "input": action_input,
                            "output": fallback_observation,
                        })
                        observation = fallback_observation
                    else:
                        raise
                observations.append(observation)
            next_prompt = "\n\n".join(observations)
            retrieval_s += time.perf_counter() - t2
            continue
        break

    if last_citations and "引用：" not in (final_result or ""):
        final_result = (final_result or "").rstrip() + "\n\n" + _format_citations(last_citations)

    if final_result:
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=final_result))

    classification_s = round(classification_s, 3)
    retrieval_s = round(retrieval_s, 3)
    generation_s = round(generation_s, 3)
    total_s = round(classification_s + retrieval_s + generation_s, 3)

    return {
        "status": "ok",
        "route": route,
        "answer": final_result,
        "answer_length": len(final_result or ""),
        "classification_s": classification_s,
        "retrieval_s": retrieval_s,
        "generation_s": generation_s,
        "total_s": total_s,
        "tool_calls": tool_calls,
        "internal_loops": internal_loops,
        "early_exit": False,
        "hint_level": state.get("hint_level"),
        "question_category": state.get("question_category"),
        "experiment_id": state.get("experiment_id"),
        "history_messages_before": history_messages_before,
        "history_chars_before": history_chars_before,
        "tool_traces": tool_traces,
        "error": "",
    }


def run_session(
    scenario: Dict[str, Any],
    *,
    repeat_id: int,
    enable_websearch: bool,
    max_agent_loops: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    session_id = f"s{scenario['id']:02d}_r{repeat_id:02d}"
    history: List[BaseMessage] = []
    state: Dict[str, Any] = {}
    session_rows: List[Dict[str, Any]] = []
    cumulative_s = 0.0
    aborted = False
    error_msg = ""

    print(
        f"\n{'=' * 76}\n"
        f"会话 {session_id} | 场景 {scenario['id']} | {scenario['name']} [{scenario['category']}]\n"
        f"{'=' * 76}"
    )

    for turn_idx, student_msg in enumerate(scenario["turns"], 1):
        print(f"  [{turn_idx}/{len(scenario['turns'])}] {student_msg[:60]}...", end=" ", flush=True)
        try:
            result = benchmark_single_turn(
                student_msg,
                history,
                state,
                enable_websearch=enable_websearch,
                max_agent_loops=max_agent_loops,
            )
            cumulative_s += result["total_s"] or 0.0
            print(
                f"total={result['total_s']:.1f}s "
                f"(分类={result['classification_s']:.1f}s "
                f"检索={result['retrieval_s']:.1f}s "
                f"生成={result['generation_s']:.1f}s "
                f"工具={result['tool_calls']}次 "
                f"hint={result.get('hint_level', '?')})"
            )
        except Exception as exc:
            aborted = True
            error_msg = str(exc)
            result = {
                "status": "failed",
                "route": "error",
                "answer": "",
                "answer_length": 0,
                "classification_s": None,
                "retrieval_s": None,
                "generation_s": None,
                "total_s": None,
                "tool_calls": 0,
                "internal_loops": 0,
                "early_exit": False,
                "hint_level": state.get("hint_level"),
                "question_category": state.get("question_category"),
                "experiment_id": state.get("experiment_id"),
                "history_messages_before": len(history),
                "history_chars_before": sum(len(getattr(m, "content", "") or "") for m in history),
                "tool_traces": [],
                "error": error_msg,
            }
            print(f"[失败: {exc}]")

        session_rows.append({
            "session_id": session_id,
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "category": scenario["category"],
            "repeat_id": repeat_id,
            "turn": turn_idx,
            "question_length": len(student_msg),
            "answer_length": result["answer_length"],
            "history_messages_before": result["history_messages_before"],
            "history_chars_before": result["history_chars_before"],
            "hint_level": result.get("hint_level"),
            "question_category": result.get("question_category"),
            "experiment_id": result.get("experiment_id") or "",
            "route": result.get("route"),
            "classification_s": result.get("classification_s"),
            "retrieval_s": result.get("retrieval_s"),
            "generation_s": result.get("generation_s"),
            "total_s": result.get("total_s"),
            "tool_calls": result.get("tool_calls", 0),
            "internal_loops": result.get("internal_loops", 0),
            "early_exit": result.get("early_exit", False),
            "session_cumulative_s": round(cumulative_s, 3),
            "status": result.get("status", "ok"),
            "error": result.get("error", ""),
        })

        if result.get("status") != "ok":
            aborted = True
            if not error_msg:
                error_msg = result.get("error", "未知错误")
            break

    ok_rows = [r for r in session_rows if r["status"] == "ok" and r["total_s"] is not None]
    session_summary = {
        "session_id": session_id,
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "category": scenario["category"],
        "repeat_id": repeat_id,
        "turns_planned": len(scenario["turns"]),
        "turns_completed": len(ok_rows),
        "session_total_s": round(sum(r["total_s"] for r in ok_rows), 3) if ok_rows else None,
        "avg_turn_s": round(sum(r["total_s"] for r in ok_rows) / len(ok_rows), 3) if ok_rows else None,
        "tool_calls_total": sum(r["tool_calls"] for r in ok_rows),
        "internal_loops_total": sum(r["internal_loops"] for r in ok_rows),
        "final_hint_level": ok_rows[-1]["hint_level"] if ok_rows else None,
        "hint_trajectory": " → ".join(str(r["hint_level"]) for r in ok_rows) if ok_rows else "",
        "aborted": aborted,
        "error": error_msg,
    }
    return session_rows, session_summary


def _filter_scenarios(raw: Optional[str]) -> List[Dict[str, Any]]:
    scenarios = list(SCENARIOS)
    if not raw:
        return scenarios
    selected = {int(item.strip()) for item in raw.split(",") if item.strip()}
    return [s for s in scenarios if int(s["id"]) in selected]


def _serialize_stats(stats: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    serialized = {}
    for key, value in stats.items():
        serialized[key] = round(value, 3) if value is not None else None
    return serialized


def build_summary(
    turn_rows: List[Dict[str, Any]],
    session_rows: List[Dict[str, Any]],
    *,
    scenario_count: int,
    repeats: int,
    enable_websearch: bool,
    max_agent_loops: int,
) -> Dict[str, Any]:
    ok_turns = [r for r in turn_rows if r["status"] == "ok" and r["total_s"] is not None]
    ok_sessions = [r for r in session_rows if not r["aborted"] and r["session_total_s"] is not None]

    turn_totals = [r["total_s"] for r in ok_turns]
    first_turns = [r["total_s"] for r in ok_turns if r["turn"] == 1]
    followup_turns = [r["total_s"] for r in ok_turns if r["turn"] >= 2]
    session_totals = [r["session_total_s"] for r in ok_sessions]
    classification_vals = [r["classification_s"] for r in ok_turns if r["classification_s"] is not None]
    retrieval_vals = [r["retrieval_s"] for r in ok_turns if r["retrieval_s"] is not None]
    generation_vals = [r["generation_s"] for r in ok_turns if r["generation_s"] is not None]

    stage_sum = {
        "classification_s": float(sum(classification_vals)),
        "retrieval_s": float(sum(retrieval_vals)),
        "generation_s": float(sum(generation_vals)),
    }
    total_stage_time = sum(stage_sum.values()) or 1.0
    stage_share = {
        name: round(value / total_stage_time, 4)
        for name, value in stage_sum.items()
    }

    by_turn_index: Dict[int, Dict[str, Any]] = {}
    turn_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in ok_turns:
        turn_groups[int(row["turn"])].append(row)
    for turn_idx, rows in sorted(turn_groups.items()):
        vals = [r["total_s"] for r in rows]
        by_turn_index[turn_idx] = {
            "n": len(rows),
            "total_s": _serialize_stats(_stats(vals)),
            "classification_mean": round(_mean([r["classification_s"] for r in rows if r["classification_s"] is not None]) or 0.0, 3),
            "retrieval_mean": round(_mean([r["retrieval_s"] for r in rows if r["retrieval_s"] is not None]) or 0.0, 3),
            "generation_mean": round(_mean([r["generation_s"] for r in rows if r["generation_s"] is not None]) or 0.0, 3),
            "tool_calls_mean": round(_mean([r["tool_calls"] for r in rows]) or 0.0, 3),
            "internal_loops_mean": round(_mean([r["internal_loops"] for r in rows]) or 0.0, 3),
            "answer_length_mean": round(_mean([r["answer_length"] for r in rows]) or 0.0, 3),
            "history_chars_before_mean": round(_mean([r["history_chars_before"] for r in rows]) or 0.0, 3),
        }

    by_category: Dict[str, Dict[str, Any]] = {}
    category_turn_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    category_session_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in ok_turns:
        category_turn_groups[row["category"]].append(row)
    for row in session_rows:
        if row["session_total_s"] is not None:
            category_session_groups[row["category"]].append(row)
    for category in sorted(set(category_turn_groups) | set(category_session_groups)):
        turn_group = category_turn_groups.get(category, [])
        session_group = category_session_groups.get(category, [])
        by_category[category] = {
            "n_turns": len(turn_group),
            "n_sessions": len(session_group),
            "turn_latency": _serialize_stats(_stats([r["total_s"] for r in turn_group])),
            "session_latency": _serialize_stats(_stats([r["session_total_s"] for r in session_group if r["session_total_s"] is not None])),
            "tool_calls_mean": round(_mean([r["tool_calls"] for r in turn_group]) or 0.0, 3),
            "answer_length_mean": round(_mean([r["answer_length"] for r in turn_group]) or 0.0, 3),
        }

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_scenarios": scenario_count,
        "repeats": repeats,
        "enable_websearch": enable_websearch,
        "max_agent_loops": max_agent_loops,
        "n_sessions": len(session_rows),
        "n_turns": len(turn_rows),
        "n_ok_sessions": len(ok_sessions),
        "n_ok_turns": len(ok_turns),
        "failure_count": len([r for r in turn_rows if r["status"] != "ok"]),
        "failure_rate": round(len([r for r in turn_rows if r["status"] != "ok"]) / max(1, len(turn_rows)), 4),
        "avg_turn_latency": round(_mean(turn_totals) or 0.0, 3),
        "avg_first_turn_latency": round(_mean(first_turns) or 0.0, 3),
        "avg_followup_turn_latency": round(_mean(followup_turns) or 0.0, 3),
        "avg_session_latency": round(_mean(session_totals) or 0.0, 3),
        "p50_turn_latency": round(_percentile(turn_totals, 50) or 0.0, 3),
        "p95_turn_latency": round(_percentile(turn_totals, 95) or 0.0, 3),
        "p99_turn_latency": round(_percentile(turn_totals, 99) or 0.0, 3),
        "p50_session_latency": round(_percentile(session_totals, 50) or 0.0, 3),
        "p95_session_latency": round(_percentile(session_totals, 95) or 0.0, 3),
        "avg_tool_calls_per_turn": round(_mean([r["tool_calls"] for r in ok_turns]) or 0.0, 3),
        "avg_tool_calls_per_session": round(_mean([r["tool_calls_total"] for r in ok_sessions]) or 0.0, 3),
        "avg_internal_loops_per_turn": round(_mean([r["internal_loops"] for r in ok_turns]) or 0.0, 3),
        "stage_share": stage_share,
        "turn_latency": _serialize_stats(_stats(turn_totals)),
        "session_latency": _serialize_stats(_stats(session_totals)),
        "by_turn_index": by_turn_index,
        "by_category": by_category,
    }
    return summary


def save_turn_csv(path: Path, rows: List[Dict[str, Any]]):
    fieldnames = [
        "session_id",
        "scenario_id",
        "scenario_name",
        "category",
        "repeat_id",
        "turn",
        "question_length",
        "answer_length",
        "history_messages_before",
        "history_chars_before",
        "hint_level",
        "question_category",
        "experiment_id",
        "route",
        "classification_s",
        "retrieval_s",
        "generation_s",
        "total_s",
        "tool_calls",
        "internal_loops",
        "early_exit",
        "session_cumulative_s",
        "status",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_session_csv(path: Path, rows: List[Dict[str, Any]]):
    fieldnames = [
        "session_id",
        "scenario_id",
        "scenario_name",
        "category",
        "repeat_id",
        "turns_planned",
        "turns_completed",
        "session_total_s",
        "avg_turn_s",
        "tool_calls_total",
        "internal_loops_total",
        "final_hint_level",
        "hint_trajectory",
        "aborted",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(
    turn_rows: List[Dict[str, Any]],
    session_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    timestamp: str,
) -> List[str]:
    ok_turns = [r for r in turn_rows if r["status"] == "ok" and r["total_s"] is not None]
    ok_sessions = [r for r in session_rows if r["session_total_s"] is not None]
    if not ok_turns or not ok_sessions:
        return []

    plt = _configure_matplotlib()
    paths: List[str] = []

    # 图 1：按轮次平均时延
    turn_stats = summary["by_turn_index"]
    turns = sorted(int(k) for k in turn_stats.keys())
    means = [turn_stats[t]["total_s"]["mean"] for t in turns]
    stds = [turn_stats[t]["total_s"]["std"] for t in turns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(turns, means, yerr=stds, fmt="o-", color="#3498db", linewidth=2, capsize=4)
    for x, y in zip(turns, means):
        ax.text(x, y + 1.0, f"{y:.1f}s", ha="center", fontsize=9)
    ax.set_xlabel("对话轮次")
    ax.set_ylabel("平均单轮时延 (s)")
    ax.set_title("多轮对话各轮次平均时延")
    ax.set_xticks(turns)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    turn_path = RESULTS_DIR / f"multi_turn_perf_turn_latency_{timestamp}.png"
    fig.savefig(turn_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(turn_path))

    # 图 2：按题型会话总时延箱线图
    grouped_sessions: Dict[str, List[float]] = defaultdict(list)
    label_map = {
        "LAB_TROUBLESHOOTING": "实验排错",
        "THEORY_CONCEPT": "理论概念",
        "CONFIG_REVIEW": "配置审查",
        "CALCULATION": "计算分析",
    }
    for row in ok_sessions:
        grouped_sessions[row["category"]].append(row["session_total_s"])

    labels = [label_map.get(cat, cat) for cat in grouped_sessions.keys()]
    data = [grouped_sessions[cat] for cat in grouped_sessions.keys()]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    box_colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for idx, vals in enumerate(data, start=1):
        avg = float(np.mean(vals))
        ax.plot(idx, avg, "D", color="red", markersize=6)
        ax.text(idx + 0.12, avg, f"{avg:.1f}s", fontsize=8, va="center")
    ax.set_ylabel("会话总时延 (s)")
    ax.set_title("不同题型的多轮会话成本")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    session_path = RESULTS_DIR / f"multi_turn_perf_session_cost_{timestamp}.png"
    fig.savefig(session_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(session_path))

    # 图 3：按轮次阶段耗时堆叠柱状图
    cls_means = [turn_stats[t]["classification_mean"] for t in turns]
    ret_means = [turn_stats[t]["retrieval_mean"] for t in turns]
    gen_means = [turn_stats[t]["generation_mean"] for t in turns]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(turns, cls_means, label="分类", color="#4C72B0", alpha=0.85)
    ax.bar(turns, ret_means, bottom=cls_means, label="检索+工具", color="#55A868", alpha=0.85)
    bottoms = [c + r for c, r in zip(cls_means, ret_means)]
    ax.bar(turns, gen_means, bottom=bottoms, label="生成", color="#C44E52", alpha=0.85)
    for x, total in zip(turns, [c + r + g for c, r, g in zip(cls_means, ret_means, gen_means)]):
        ax.text(x, total + 1.0, f"{total:.1f}s", ha="center", fontsize=9)
    ax.set_xlabel("对话轮次")
    ax.set_ylabel("平均时延 (s)")
    ax.set_title("多轮对话各阶段平均耗时分解")
    ax.set_xticks(turns)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    stage_path = RESULTS_DIR / f"multi_turn_perf_stage_breakdown_{timestamp}.png"
    fig.savefig(stage_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(stage_path))

    return paths


def main():
    parser = argparse.ArgumentParser(description="多轮对话工程性能基准测试")
    parser.add_argument("--scenarios", type=str, default=None, help="要运行的场景 ID（逗号分隔，默认全部）")
    parser.add_argument("--repeats", type=int, default=4, help="每个场景重复运行次数（默认 4）")
    parser.add_argument("--enable-websearch", action="store_true", help="启用搜索工具（默认关闭）")
    parser.add_argument("--max-agent-loops", type=int, default=5, help="单轮内部 Agent 最大循环数（默认 5）")
    args = parser.parse_args()

    scenarios = _filter_scenarios(args.scenarios)
    if not scenarios:
        print("错误：未匹配到任何场景")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("多轮对话工程性能基准测试")
    print(f"  场景数：{len(scenarios)}")
    print(f"  每场景重复次数：{args.repeats}")
    print(f"  预计会话样本数：{len(scenarios) * args.repeats}")
    print(f"  搜索工具：{'开启' if args.enable_websearch else '关闭'}")
    print(f"  Agent 最大内部循环：{args.max_agent_loops}")
    print()

    warmup()

    all_turn_rows: List[Dict[str, Any]] = []
    all_session_rows: List[Dict[str, Any]] = []

    total_sessions = len(scenarios) * args.repeats
    session_counter = 0
    for repeat_id in range(1, args.repeats + 1):
        for scenario in scenarios:
            session_counter += 1
            print(f"[{session_counter}/{total_sessions}] 开始运行场景 {scenario['id']} 第 {repeat_id} 次")
            turn_rows, session_row = run_session(
                scenario,
                repeat_id=repeat_id,
                enable_websearch=args.enable_websearch,
                max_agent_loops=args.max_agent_loops,
            )
            all_turn_rows.extend(turn_rows)
            all_session_rows.append(session_row)

    turn_csv = RESULTS_DIR / f"multi_turn_perf_{timestamp}.csv"
    session_csv = RESULTS_DIR / f"multi_turn_perf_sessions_{timestamp}.csv"
    save_turn_csv(turn_csv, all_turn_rows)
    save_session_csv(session_csv, all_session_rows)
    print(f"\n逐轮明细已保存：{turn_csv}")
    print(f"会话汇总已保存：{session_csv}")

    summary = build_summary(
        all_turn_rows,
        all_session_rows,
        scenario_count=len(scenarios),
        repeats=args.repeats,
        enable_websearch=args.enable_websearch,
        max_agent_loops=args.max_agent_loops,
    )
    summary["timestamp"] = timestamp

    print(f"\n{'=' * 78}")
    print("多轮对话工程性能汇总")
    print(f"{'=' * 78}")
    print(f"会话样本数：{summary['n_sessions']}，有效会话：{summary['n_ok_sessions']}")
    print(f"轮次样本数：{summary['n_turns']}，有效轮次：{summary['n_ok_turns']}")
    print(f"失败轮次：{summary['failure_count']}，失败率：{summary['failure_rate']:.2%}")
    print()
    print(f"平均单轮时延：{summary['avg_turn_latency']:.2f}s")
    print(f"首轮平均时延：{summary['avg_first_turn_latency']:.2f}s")
    print(f"后续轮平均时延：{summary['avg_followup_turn_latency']:.2f}s")
    print(f"平均会话总时延：{summary['avg_session_latency']:.2f}s")
    print(f"P50 / P95 单轮时延：{summary['p50_turn_latency']:.2f}s / {summary['p95_turn_latency']:.2f}s")
    print(f"平均每轮工具调用：{summary['avg_tool_calls_per_turn']:.2f}")
    print(f"平均每会话工具调用：{summary['avg_tool_calls_per_session']:.2f}")
    print()
    print("按轮次平均时延：")
    for turn_idx, info in summary["by_turn_index"].items():
        print(
            f"  第 {turn_idx} 轮: n={info['n']:<3} "
            f"avg={info['total_s']['mean']:.2f}s "
            f"(分类={info['classification_mean']:.2f}s "
            f"检索={info['retrieval_mean']:.2f}s "
            f"生成={info['generation_mean']:.2f}s)"
        )
    print()
    print("按题型会话时延：")
    for category, info in summary["by_category"].items():
        session_mean = info["session_latency"]["mean"]
        turn_mean = info["turn_latency"]["mean"]
        print(
            f"  {category:<20} "
            f"session_avg={session_mean:.2f}s "
            f"turn_avg={turn_mean:.2f}s "
            f"n_sessions={info['n_sessions']}"
        )

    summary_json = RESULTS_DIR / f"multi_turn_perf_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    figure_paths = []
    try:
        figure_paths = generate_figures(
            all_turn_rows,
            all_session_rows,
            summary,
            timestamp=timestamp,
        )
        for path in figure_paths:
            print(f"图表已保存：{path}")
    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")

    if figure_paths:
        summary["figure_paths"] = figure_paths
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
