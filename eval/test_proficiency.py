"""
学生水平识别模型 - 模拟量化测试

用法：python eval/test_proficiency.py

功能：
  模拟三种典型学生（强/中/弱）的多轮对话交互，
  验证水平评分系统能否正确区分并个性化。

不调用 LLM，零 API 成本，纯算法模拟。

输出：
  1. 每轮交互后的分数变化曲线（CSV）
  2. 控制台汇总：最终分数、初始 Hint Level、Prompt 摘要
"""

import os
import sys
import json
import random
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from storage.user_store import (
    DB_FILE,
    get_proficiency_scores,
    upsert_proficiency_score,
    _connect,
    _utc_now,
)
from storage.proficiency import (
    compute_turn_signal,
    update_proficiency_from_metric,
    get_initial_hint_level,
    get_proficiency_summary,
    _CATEGORIES,
)


# ── 类型映射 ──────────────────────────────────────────────
_TYPE_TO_CATEGORY = {
    "concept": "THEORY_CONCEPT",
    "troubleshooting": "LAB_TROUBLESHOOTING",
    "config": "CONFIG_REVIEW",
    "calculation": "CALCULATION",
}

# ── 学生行为模板 ──────────────────────────────────────────
# 定义不同水平学生在各 hint level 下的行为概率
# escalation_prob: 在当前 level 被判定为 INCREASE 的概率
# failsafe_prob: 在不 INCREASE 时触发 failsafe（3 轮卡住）的概率

STUDENT_PROFILES = {
    "强学生": {
        "description": "理论扎实，实验熟练，很少需要提示升级",
        "escalation_prob": {0: 0.10, 1: 0.15, 2: 0.20, 3: 0.0},
        "failsafe_prob": 0.05,
        "avg_turns_per_session": 3,
    },
    "中等学生": {
        "description": "基础知识可以，但复杂问题需要引导",
        "escalation_prob": {0: 0.35, 1: 0.30, 2: 0.25, 3: 0.0},
        "failsafe_prob": 0.15,
        "avg_turns_per_session": 4,
    },
    "弱学生": {
        "description": "基础薄弱，经常需要直接答案",
        "escalation_prob": {0: 0.65, 1: 0.55, 2: 0.45, 3: 0.0},
        "failsafe_prob": 0.30,
        "avg_turns_per_session": 5,
    },
}


def _create_test_user(conn: sqlite3.Connection, username: str) -> str:
    """在数据库中创建临时测试用户，返回 user_id。"""
    user_id = f"u_test_{username}"
    conn.execute(
        """
        INSERT OR IGNORE INTO users
        (id, username, password_salt, password_hash, profile_json, preferences_json, created_at)
        VALUES (?, ?, 'test', 'test', '{}', '{}', ?)
        """,
        (user_id, f"__test_{username}", _utc_now()),
    )
    conn.commit()
    return user_id


def _cleanup_test_users(conn: sqlite3.Connection):
    """清理所有测试用户数据。"""
    conn.execute("DELETE FROM proficiency_scores WHERE user_id LIKE 'u_test_%'")
    conn.execute("DELETE FROM interaction_metrics WHERE user_id LIKE 'u_test_%'")
    conn.execute("DELETE FROM users WHERE id LIKE 'u_test_%'")
    conn.commit()


def simulate_turn(
    hint_level: int,
    profile: dict,
) -> dict:
    """模拟一轮交互，返回 state 信号。

    根据学生 profile 的概率决定是否触发 hint 升级。
    """
    esc_prob = profile["escalation_prob"].get(hint_level, 0)
    do_escalate = random.random() < esc_prob

    if do_escalate:
        new_level = min(hint_level + 1, 3)
        return {
            "_hint_level_start": hint_level,
            "hint_level": new_level,
            "_hint_decision": "INCREASE",
            "_was_failsafe": False,
        }
    else:
        # 不升级，但可能触发 failsafe
        do_failsafe = random.random() < profile["failsafe_prob"]
        if do_failsafe and hint_level < 3:
            new_level = min(hint_level + 1, 3)
            return {
                "_hint_level_start": hint_level,
                "hint_level": new_level,
                "_hint_decision": "MAINTAIN",
                "_was_failsafe": True,
            }
        else:
            return {
                "_hint_level_start": hint_level,
                "hint_level": hint_level,
                "_hint_decision": "MAINTAIN",
                "_was_failsafe": False,
            }


def simulate_session(
    user_id: str,
    category: str,
    profile: dict,
    session_idx: int,
) -> list:
    """模拟一个完整会话（多轮），返回每轮的 (turn, signal, score) 记录。"""
    n_turns = max(2, profile["avg_turns_per_session"] + random.randint(-1, 1))
    hint_level = get_initial_hint_level(user_id, category)
    records = []

    for turn in range(1, n_turns + 1):
        turn_state = simulate_turn(hint_level, profile)
        turn_state["question_category"] = category
        turn_state["user_turn_count"] = turn

        signal = compute_turn_signal(turn_state)
        update_proficiency_from_metric(user_id, turn_state)

        scores = get_proficiency_scores(user_id)
        cat_score = scores.get(category, {}).get("score", 0.5)
        overall_score = scores.get("OVERALL", {}).get("score", 0.5)

        records.append({
            "session": session_idx,
            "turn": turn,
            "category": category,
            "hint_start": turn_state["_hint_level_start"],
            "hint_end": turn_state["hint_level"],
            "decision": turn_state["_hint_decision"],
            "failsafe": turn_state["_was_failsafe"],
            "signal": round(signal, 3),
            "cat_score": round(cat_score, 3),
            "overall_score": round(overall_score, 3),
        })

        hint_level = turn_state["hint_level"]

    return records


def load_questions_by_type() -> dict:
    """从 qa_dataset.json 加载题目按类型分组。"""
    qa_path = ROOT / "eval" / "qa_dataset.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    by_type = {}
    for q in questions:
        cat = _TYPE_TO_CATEGORY.get(q.get("type", ""), "THEORY_CONCEPT")
        by_type.setdefault(cat, []).append(q)
    return by_type


def run_simulation():
    """运行完整模拟测试。"""
    random.seed(42)

    questions_by_type = load_questions_by_type()

    conn = _connect()
    _cleanup_test_users(conn)

    print("=" * 70)
    print("学生水平识别模型 - 模拟量化测试")
    print("=" * 70)

    all_records = []
    results_summary = []

    for profile_name, profile in STUDENT_PROFILES.items():
        user_id = _create_test_user(conn, profile_name)

        print(f"\n{'─' * 60}")
        print(f"模拟 [{profile_name}]: {profile['description']}")
        print(f"{'─' * 60}")

        profile_records = []

        # 每个类别模拟 5 个 session
        for cat in _CATEGORIES:
            for s in range(5):
                recs = simulate_session(user_id, cat, profile, s + 1)
                profile_records.extend(recs)

        # 最终水平
        scores = get_proficiency_scores(user_id)
        initial_levels = {}

        print(f"\n  最终水平分数：")
        for cat in _CATEGORIES:
            entry = scores.get(cat, {})
            score = entry.get("score", 0.5)
            conf = entry.get("confidence", 0.0)
            count = entry.get("interaction_count", 0)
            level = get_initial_hint_level(user_id, cat)
            initial_levels[cat] = level
            print(f"    {cat:25s}: score={score:.3f}  conf={conf:.3f}  count={count:3d}  → 初始Level={level}")

        overall = scores.get("OVERALL", {})
        print(f"    {'OVERALL':25s}: score={overall.get('score', 0.5):.3f}  conf={overall.get('confidence', 0.0):.3f}")

        summary = get_proficiency_summary(user_id)
        print(f"\n  Prompt 摘要: {summary or '（数据不足）'}")

        # 记录结果
        for r in profile_records:
            r["student_type"] = profile_name
        all_records.extend(profile_records)

        results_summary.append({
            "profile": profile_name,
            "scores": {k: round(v.get("score", 0.5), 3) for k, v in scores.items()},
            "initial_levels": initial_levels,
            "summary": summary,
        })

    conn.close()

    # ── 输出 CSV ──
    csv_path = ROOT / "eval" / "proficiency_simulation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "student_type", "session", "turn", "category",
            "hint_start", "hint_end", "decision", "failsafe",
            "signal", "cat_score", "overall_score",
        ])
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n\n详细记录已保存: {csv_path}")

    # ── 验证断言 ──
    print("\n" + "=" * 70)
    print("验证结果")
    print("=" * 70)

    strong = results_summary[0]  # 强学生
    weak = results_summary[2]    # 弱学生

    checks = []

    # 检查 1：强学生 OVERALL 分数 > 弱学生
    s_strong = strong["scores"].get("OVERALL", 0.5)
    s_weak = weak["scores"].get("OVERALL", 0.5)
    ok1 = s_strong > s_weak
    checks.append(("强学生 OVERALL > 弱学生 OVERALL", ok1, f"{s_strong:.3f} vs {s_weak:.3f}"))

    # 检查 2：弱学生至少有一个类别初始 Level > 0
    weak_levels = list(weak["initial_levels"].values())
    ok2 = any(l > 0 for l in weak_levels)
    checks.append(("弱学生至少有一个分类初始Level>0", ok2, f"levels={weak_levels}"))

    # 检查 3：强学生所有类别初始 Level == 0
    strong_levels = list(strong["initial_levels"].values())
    ok3 = all(l == 0 for l in strong_levels)
    checks.append(("强学生所有分类初始Level=0", ok3, f"levels={strong_levels}"))

    # 检查 4：每个类别强学生 > 弱学生
    for cat in _CATEGORIES:
        s_s = strong["scores"].get(cat, 0.5)
        s_w = weak["scores"].get(cat, 0.5)
        ok = s_s > s_w
        checks.append((f"{cat}: 强>弱", ok, f"{s_s:.3f} vs {s_w:.3f}"))

    # 检查 5：弱学生有 Prompt 摘要
    ok5 = bool(weak["summary"])
    checks.append(("弱学生生成了Prompt摘要", ok5, weak["summary"][:50] if weak["summary"] else "空"))

    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {name}  ({detail})")

    # ── 清理 ──
    conn = _connect()
    _cleanup_test_users(conn)
    conn.close()

    print(f"\n{'=' * 70}")
    if all_pass:
        print("所有检查通过！水平识别模型工作正常。")
    else:
        print("部分检查未通过，请检查评分参数。")
    print(f"{'=' * 70}")

    return all_pass


if __name__ == "__main__":
    import csv
    success = run_simulation()
    sys.exit(0 if success else 1)
