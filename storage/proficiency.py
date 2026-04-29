"""学生水平评分引擎

基于交互信号的 EMA（指数移动平均）+ 时间衰减算法，
零 LLM 开销，纯算术运算。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .user_store import get_proficiency_scores, upsert_proficiency_score

# ---------- 常量 ----------

HALF_LIFE_DAYS = 14          # 旧交互影响半衰期（天）
CONFIDENCE_SATURATION = 30   # 交互次数达到此值时置信度 ≈ 1.0
LEARNING_RATE_BASE = 0.15    # EMA 基础学习率
INITIAL_SCORE = 0.5          # 新用户初始分数（中等水平）

_CATEGORIES = [
    "LAB_TROUBLESHOOTING",
    "THEORY_CONCEPT",
    "CONFIG_REVIEW",
    "CALCULATION",
]

# ---------- 信号提取 ----------


def compute_turn_signal(state: Dict[str, Any]) -> float:
    """从单轮交互 state 中提取水平信号。

    返回 [-1.0, +1.0]，正值 = 表现好，负值 = 表现差。
    """
    hint_start = state.get("_hint_level_start", 0)
    hint_end = state.get("hint_level", 0)
    hint_decision = state.get("_hint_decision", "MAINTAIN")
    was_failsafe = state.get("_was_failsafe", False)
    transition_reason = state.get("_hint_transition_reason", "")
    phase = state.get("_hint_phase", "")
    evidence_score = int(state.get("_hint_evidence_score", 0) or 0)
    stagnation_turns = int(state.get("_hint_stagnation_turns", 0) or 0)

    signal = 0.0

    # 因素 1：Hint 升级惩罚
    level_delta = hint_end - hint_start
    if level_delta > 0:
        signal -= 0.3 * level_delta
        if was_failsafe:
            signal -= 0.1

    # 因素 2：状态机原因修正
    if transition_reason in {"direct_answer_request", "lab_user_requested_direct_answer"}:
        signal += 0.1  # 用户选择直接收敛，不等同于能力不足
    elif transition_reason in {"evidence_complete_ready_to_converge", "resolved"}:
        signal += 0.15
    elif transition_reason in {"stalled_without_evidence", "evidence_collection_stalled", "non_lab_stagnation_failsafe"}:
        signal -= 0.15

    # 因素 3：低 Level 保持 = 正信号
    if hint_decision == "MAINTAIN" and hint_start <= 1 and level_delta == 0:
        signal += 0.2

    # 因素 4：在 Level 0 保持 = 额外加分
    if hint_start == 0 and level_delta == 0:
        signal += 0.1

    # 因素 5：LAB 证据推进是正信号，长时间停滞是负信号
    if phase in {"narrowing_root_cause", "proposing_fix"} and evidence_score >= 3:
        signal += 0.1
    if stagnation_turns >= 3:
        signal -= 0.1

    return max(-1.0, min(1.0, signal))


# ---------- EMA 更新 ----------


def _time_decay(last_updated: str, now: str) -> float:
    """计算时间衰减因子 (0, 1]。"""
    try:
        dt_last = datetime.fromisoformat(last_updated)
        dt_now = datetime.fromisoformat(now)
        days_elapsed = max(0.0, (dt_now - dt_last).total_seconds() / 86400)
    except (ValueError, TypeError):
        days_elapsed = 0.0
    return math.exp(-0.693 * days_elapsed / HALF_LIFE_DAYS)


def _update_score(
    current_score: float,
    current_count: int,
    new_signal: float,
    decay: float,
) -> float:
    """EMA 更新：融合时间衰减 + 自适应学习率。"""
    # 自适应学习率：新用户高，老用户低
    alpha = LEARNING_RATE_BASE / (1 + 0.02 * current_count)

    # 衰减当前分数（向 0.5 中性回归）
    decayed_score = INITIAL_SCORE + (current_score - INITIAL_SCORE) * decay

    # 将 signal [-1, 1] 映射到 [0, 1]
    target = new_signal * 0.5 + 0.5

    # EMA
    new_score = decayed_score + alpha * (target - decayed_score)
    return max(0.05, min(0.95, new_score))


def _compute_confidence(count: int, decay: float) -> float:
    """基于交互次数的 sigmoid 置信度，乘以时间衰减。"""
    raw = 1.0 - math.exp(-count / (CONFIDENCE_SATURATION * 0.5))
    recency = min(1.0, decay + 0.3)
    return raw * recency


# ---------- 对外接口 ----------


def update_proficiency_from_metric(user_id: str, state: Dict[str, Any]) -> None:
    """在每次交互后调用，更新该用户的水平分数。

    纯算术，< 1ms，在 _persist 后台线程中调用。
    """
    category = state.get("question_category", "UNKNOWN")
    if category not in _CATEGORIES:
        return

    signal = compute_turn_signal(state)
    now = datetime.now(timezone.utc).isoformat()

    # 读取当前分数
    scores = get_proficiency_scores(user_id)
    entry = scores.get(category, {
        "score": INITIAL_SCORE,
        "confidence": 0.0,
        "interaction_count": 0,
        "last_updated": now,
    })

    decay = _time_decay(entry["last_updated"], now)
    new_count = entry["interaction_count"] + 1
    new_score = _update_score(entry["score"], entry["interaction_count"], signal, decay)
    new_confidence = _compute_confidence(new_count, decay)

    # 写入分类分数
    upsert_proficiency_score(user_id, category, new_score, new_confidence, new_count)

    # 更新 OVERALL
    _update_overall(user_id, now)


def _update_overall(user_id: str, now: str) -> None:
    """按置信度加权计算总体分数。"""
    scores = get_proficiency_scores(user_id)

    total_weight = 0.0
    weighted_sum = 0.0
    total_count = 0
    active_cats = 0

    for cat in _CATEGORIES:
        entry = scores.get(cat)
        if not entry:
            continue
        w = entry["confidence"]
        total_weight += w
        weighted_sum += entry["score"] * w
        total_count += entry["interaction_count"]
        active_cats += 1

    if total_weight < 0.01:
        return  # 无数据

    overall_score = weighted_sum / total_weight
    # 置信度：活跃分类的平均置信度，再乘以覆盖率折扣
    avg_confidence = total_weight / active_cats
    coverage = active_cats / len(_CATEGORIES)
    overall_confidence = min(1.0, avg_confidence * (0.5 + 0.5 * coverage))

    upsert_proficiency_score(user_id, "OVERALL", overall_score, overall_confidence, total_count)


def get_initial_hint_level(user_id: str, category: Optional[str] = None) -> int:
    """根据水平分数决定新会话初始 Hint Level。

    返回 0、1 或 2（永远不返回 3，Level 3 只能通过对话升级获得）。
    """
    scores = get_proficiency_scores(user_id)
    if not scores:
        return 0

    # 优先使用分类分数，其次用 OVERALL
    entry = None
    if category and category in scores:
        entry = scores[category]
    if not entry or entry.get("confidence", 0) < 0.3:
        entry = scores.get("OVERALL")

    if not entry or entry.get("confidence", 0) < 0.3:
        return 0  # 数据不足，默认从 0 开始

    score = entry["score"]

    if score >= 0.5:
        return 0   # 正常 Socratic
    elif score >= 0.35:
        return 1   # 跳过纯探索，给现象提示
    else:
        return 2   # 直接给原理解释


def get_proficiency_summary(user_id: str) -> str:
    """生成自然语言水平描述，用于注入系统 Prompt。

    返回空字符串表示数据不足。
    """
    scores = get_proficiency_scores(user_id)
    if not scores:
        return ""

    overall = scores.get("OVERALL")
    if not overall or overall.get("confidence", 0) < 0.2:
        return ""

    _LABELS = {
        "LAB_TROUBLESHOOTING": "实验排障",
        "THEORY_CONCEPT": "理论概念",
        "CONFIG_REVIEW": "配置审查",
        "CALCULATION": "子网计算",
    }

    parts = []
    for cat in _CATEGORIES:
        entry = scores.get(cat)
        if entry and entry.get("confidence", 0) >= 0.2:
            label = _LABELS.get(cat, cat)
            s = entry["score"]
            if s >= 0.6:
                level = "较好"
            elif s >= 0.5:
                level = "一般"
            else:
                level = "较弱"
            parts.append(f"{label}{level}({s:.2f})")

    if not parts:
        return ""

    overall_s = overall["score"]
    if overall_s >= 0.6:
        overall_label = "进阶水平"
    elif overall_s >= 0.5:
        overall_label = "中等水平"
    else:
        overall_label = "基础水平"

    return f"该学生整体{overall_label}({overall_s:.2f})。分项：{'，'.join(parts)}。请适当调整解释深度和引导方式。"
