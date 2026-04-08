"""
消融实验：量化各核心模块的独立贡献

通过系统地移除各模块，对比完整系统与消融变体的回答质量，
证明每个创新模块的必要性。

变体设计：
  A: 完整系统（全部开启）
  B: -Reranker（禁用重排器）
  C: -Socratic（强制最高 hint_level，直接给完整答案）
  D: -拓扑（移除拓扑工具）
  E: -水平自适应（不传 user_id，无个性化初始 hint level）
  F: 纯 LLM 基线（裸 DeepSeek，无 RAG / 无工具）

用法：
  python eval/ablation_study.py                         # 运行全部变体
  python eval/ablation_study.py --variants A,B,F        # 只跑指定变体
  python eval/ablation_study.py --n-samples 30          # 指定抽样题数
  python eval/ablation_study.py --dataset eval/qa_dataset_topo_balanced.json
  python eval/ablation_study.py --resume                # 从断点续跑

输出：
  eval/results/ablation_YYYYMMDD_HHMMSS.csv             — 逐题逐变体打分
  eval/results/ablation_YYYYMMDD_HHMMSS.json            — 汇总 + 统计检验
  eval/results/ablation_YYYYMMDD_HHMMSS.png             — 消融贡献柱状图

环境变量：
  OPENAI_API_KEY   (必须，用于 GPT-4o 打分)
  DEEPSEEK_API_KEY (必须，用于 DeepSeek 生成回答)
"""

import os
import sys
import json
import csv
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

os.environ.setdefault("RAG_REBUILD_INDEX", "0")

from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.agent import query as agent_query
from agentic_rag.llm_config import build_chat_llm

# ── 配置 ─────────────────────────────────────────────
QA_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

SCORE_KEYS = [
    "relevance", "faithfulness", "completeness", "technical_accuracy",
    "pedagogical_guidance", "progressive_disclosure", "overall",
]
SCORE_NAMES_ZH = {
    "relevance": "相关性", "faithfulness": "忠实性",
    "completeness": "完整性", "technical_accuracy": "技术准确性",
    "pedagogical_guidance": "引导性", "progressive_disclosure": "递进性",
    "overall": "综合",
}

# 各分类的 hint_level 上限
MAX_HINT_LEVEL = {
    "LAB_TROUBLESHOOTING": 3,
    "THEORY_CONCEPT": 1,
    "CONFIG_REVIEW": 2,
    "CALCULATION": 1,
}

# ── 打分 Prompt（复用 run_experiments.py 的标准）────────
JUDGE_SYSTEM = """\
你是计算机网络课程的专家评审，负责客观评估 AI 助教回答的质量。
你的打分必须严格、一致，不偏袒任何方案。
请严格对照每个维度的 5 级锚点描述打分，不要凭主观印象。"""

JUDGE_USER_TEMPLATE = """\
请对以下 AI 助教的回答从 6 个维度打分（每项 1-5 分整数）。
注意：AI 回答可能包含多轮对话的累积内容（用分隔线隔开），请综合所有轮次的信息进行评估。

【学生问题】
{question}

【参考答案要点】（用于判断完整性的客观基准）
{reference}

【AI 回答】
{answer}

─── 评分标准（请严格对照打分）──────────────────────
1. 相关性（relevance）
   5=完全紧扣问题，无跑题内容
   4=主要内容相关，有少量无关内容
   3=部分相关，但有明显偏题
   2=仅少量内容与问题相关
   1=回答与问题几乎无关

2. 忠实性（faithfulness）
   5=所有结论均有合理依据，无捏造
   4=绝大部分有据可查，有 1 处无法确认来源
   3=有 2-3 处无据内容或轻微矛盾
   2=多处内容缺乏支撑或与事实矛盾
   1=大量编造，与事实严重不符

3. 完整性（completeness）
   5=参考答案要点全部被覆盖（允许跨多轮累积覆盖）
   4=覆盖 80% 以上要点，缺少次要细节
   3=覆盖约 60% 要点，有明显遗漏
   2=仅覆盖少数要点，核心内容缺失
   1=几乎未覆盖任何要点

4. 技术准确性（technical_accuracy）
   5=IP地址/子网掩码/命令语法/协议行为等技术事实完全正确
   4=技术内容基本正确，有 1 处小瑕疵
   3=有 1-2 处明显技术错误
   2=多处技术错误，影响理解
   1=技术内容严重错误

5. 教学引导性（pedagogical_guidance）
   5=通过提问、类比、反问等方式有效引导学生独立思考，而非直接给出答案
   4=有一定引导，但部分内容仍是直接告知
   3=引导与直接告知各占一半
   2=基本是直接给答案，偶尔有引导
   1=完全是直接给出完整答案，无任何引导

6. 信息递进性（progressive_disclosure）
   5=信息从简到繁分层递进，符合学生认知规律，逐步深入
   4=有明显的递进结构，个别地方跳跃
   3=部分递进，但整体结构不够清晰
   2=信息堆砌，缺乏层次感
   1=一次性倾倒所有信息，无递进结构

综合（overall）= 加权平均，权重：
  相关性 0.15 + 忠实性 0.20 + 完整性 0.15 + 技术准确 0.15 + 引导性 0.20 + 递进性 0.15
  四舍五入到整数。
────────────────────────────────────────────────────

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "pedagogical_guidance": <1-5>,
  "progressive_disclosure": <1-5>,
  "overall": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""

BASELINE_SYSTEM = "你是一个计算机网络实验课程的助教，请直接、完整地回答学生的问题。"

# ── 多轮对话模拟 ─────────────────────────────────────
# 模拟学生在苏格拉底引导下的自然跟进
STUDENT_FOLLOWUPS = [
    "我不太理解，能再详细解释一下吗？",
    "还有其他需要注意的地方吗？能给我更完整的解答吗？",
]

MULTI_TURN_ROUNDS = 3  # 总轮数（1 轮初始 + 2 轮跟进），可通过 --rounds 覆盖


def _multi_turn_query(question: str, state: dict, n_rounds: int = 3) -> str:
    """多轮对话：模拟学生跟进，返回所有轮次答案的拼接。"""
    history = []
    all_answers = []

    # 第 1 轮：原始问题
    answer, history, _, state = agent_query(question, history=history, state=state)
    all_answers.append(answer)

    # 后续轮次：学生跟进
    for i in range(min(n_rounds - 1, len(STUDENT_FOLLOWUPS))):
        followup = STUDENT_FOLLOWUPS[i]
        answer, history, _, state = agent_query(followup, history=history, state=state)
        all_answers.append(answer)

    return "\n\n".join(all_answers)


# ── 变体定义 ──────────────────────────────────────────

VARIANT_DESCRIPTIONS = {
    "A": "完整系统（多轮）",
    "B": "-Reranker（多轮）",
    "C": "-Socratic（直接回答）",
    "D": "-拓扑（多轮）",
    "E": "-水平自适应（多轮）",
    "F": "纯 LLM 基线",
}


def stratified_sample(questions: list, n: int, seed: int = 42) -> list:
    """按 question type 分层抽样。"""
    rng = random.Random(seed)
    by_type: Dict[str, list] = {}
    for q in questions:
        t = q.get("type", "unknown")
        by_type.setdefault(t, []).append(q)

    per_type = max(1, n // len(by_type))
    sampled = []
    remaining = []
    for t, qs in by_type.items():
        rng.shuffle(qs)
        sampled.extend(qs[:per_type])
        remaining.extend(qs[per_type:])

    need = n - len(sampled)
    if need > 0:
        rng.shuffle(remaining)
        sampled.extend(remaining[:need])

    return sampled[:n]


def call_variant_A(question: str, n_rounds: int = 3) -> str:
    """完整系统（多轮对话）：模拟学生跟进 2-3 轮，评估累积回答。"""
    return _multi_turn_query(question, state={}, n_rounds=n_rounds)


def call_variant_B(question: str, n_rounds: int = 3) -> str:
    """-Reranker：设环境变量禁用重排器（多轮）。"""
    old = os.environ.get("DISABLE_RERANKER")
    os.environ["DISABLE_RERANKER"] = "1"
    try:
        from agentic_rag import rag
        rag._retriever_cache.clear() if hasattr(rag, '_retriever_cache') else None
        return _multi_turn_query(question, state={}, n_rounds=n_rounds)
    finally:
        if old is None:
            os.environ.pop("DISABLE_RERANKER", None)
        else:
            os.environ["DISABLE_RERANKER"] = old


def call_variant_C(question: str) -> str:
    """-Socratic：强制最高 hint_level，直接给完整答案。"""
    state = {"hint_level": 99}  # 设超大值，会被 min(level, max_level) 截断为最大级别
    answer, _, _, _ = agent_query(question, history=[], state=state)
    return answer


def call_variant_D(question: str, n_rounds: int = 3) -> str:
    """-拓扑：通过 monkey-patch 移除拓扑工具（多轮）。"""
    from agentic_rag import agent as agent_mod
    original_prepare = agent_mod._prepare_context

    def patched_prepare(*args, **kwargs):
        ctx = original_prepare(*args, **kwargs)
        if ctx.contextual_actions and "拓扑" in ctx.contextual_actions:
            ctx.contextual_actions["拓扑"] = lambda _input: "拓扑工具当前不可用，请根据已有知识回答。"
        return ctx

    agent_mod._prepare_context = patched_prepare
    try:
        return _multi_turn_query(question, state={}, n_rounds=n_rounds)
    finally:
        agent_mod._prepare_context = original_prepare


def call_variant_E(question: str, n_rounds: int = 3) -> str:
    """-水平自适应：不传 user_id，初始 hint_level=0（多轮）。"""
    return _multi_turn_query(question, state={}, n_rounds=n_rounds)


def call_variant_F(question: str, llm) -> str:
    """纯 LLM 基线：裸 DeepSeek，无 RAG。"""
    messages = [
        SystemMessage(content=BASELINE_SYSTEM),
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    return resp.content


VARIANT_CALLERS = {
    "A": lambda q, llm, **kw: call_variant_A(q, n_rounds=kw.get("rounds", 3)),
    "B": lambda q, llm, **kw: call_variant_B(q, n_rounds=kw.get("rounds", 3)),
    "C": lambda q, llm, **kw: call_variant_C(q),
    "D": lambda q, llm, **kw: call_variant_D(q, n_rounds=kw.get("rounds", 3)),
    "E": lambda q, llm, **kw: call_variant_E(q, n_rounds=kw.get("rounds", 3)),
    "F": lambda q, llm, **kw: call_variant_F(q, llm),
}


def judge_answer(client: OpenAI, question: str, answer: str, reference: str) -> Dict[str, Any]:
    """GPT-4o 打分（端到端评估，不需要检索上下文）。"""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference,
        answer=answer[:1500],
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
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(QA_DATASET),
        help="题库路径（默认 eval/qa_dataset.json）",
    )
    parser.add_argument("--n-samples", type=int, default=35, help="抽样题数 (默认 35)")
    parser.add_argument("--variants", type=str, default="A,B,C,D,E,F",
                        help="要运行的变体 (逗号分隔)")
    parser.add_argument("--resume", action="store_true", help="从断点续跑")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3,
                        help="变体A多轮对话轮数 (默认 3)")
    args = parser.parse_args()

    variants = [v.strip().upper() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANT_CALLERS:
            print(f"错误：未知变体 {v}，可用：{list(VARIANT_CALLERS.keys())}")
            sys.exit(1)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("错误：请设置 OPENAI_API_KEY")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        print(f"错误：找不到题库文件 {dataset_path}")
        sys.exit(1)

    # 加载数据集
    with open(dataset_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    # 只保留有参考答案的题目
    questions_with_ref = [q for q in all_questions if q.get("reference")]
    sampled = stratified_sample(questions_with_ref, args.n_samples, seed=args.seed)
    print(f"题库：{dataset_path}")
    print(f"从 {len(questions_with_ref)} 题（有参考答案）中抽样 {len(sampled)} 题")

    type_dist = {}
    for q in sampled:
        t = q.get("type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"  类型分布：{type_dist}")
    print(f"  运行变体：{variants}")

    # 初始化
    llm = build_chat_llm(temperature=0)
    openai_client = OpenAI(api_key=openai_key)

    # 断点处理
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "ablation_checkpoint.jsonl"

    completed = set()
    all_rows = []
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    all_rows.append(row)
                    completed.add((str(row["question_id"]), row["variant"]))
        print(f"  [断点续跑] 已加载 {len(all_rows)} 条历史结果")

    # 逐题逐变体评测
    total = len(sampled) * len(variants)
    done = len(completed)

    for qi, q_item in enumerate(sampled):
        qid = q_item.get("id", "?")
        question = q_item.get("question", "")
        qtype = q_item.get("type", "unknown")
        reference = q_item.get("reference", "")

        print(f"\n── 问题 {qi+1}/{len(sampled)} (ID={qid}): {question[:50]}...")

        for variant in variants:
            key = (str(qid), variant)
            if key in completed:
                print(f"  变体 {variant} ({VARIANT_DESCRIPTIONS[variant]}): 已完成，跳过")
                continue

            done += 1
            desc = VARIANT_DESCRIPTIONS[variant]
            print(f"  [{done}/{total}] 变体 {variant} ({desc})...", end=" ", flush=True)

            try:
                t0 = time.time()
                answer = VARIANT_CALLERS[variant](question, llm, rounds=args.rounds)
                elapsed = round(time.time() - t0, 1)
                print(f"({elapsed}s)", end=" ", flush=True)
            except Exception as e:
                answer = f"[调用失败: {e}]"
                elapsed = 0
                print(f"[失败: {e}]", end=" ", flush=True)

            # GPT-4o 打分
            scores = judge_answer(openai_client, question, answer, reference)
            print(f"overall={scores.get('overall', '?')}")

            row = {
                "question_id": qid,
                "question_type": qtype,
                "variant": variant,
                "variant_desc": desc,
                "answer_length": len(answer),
                "time_s": elapsed,
                **{k: scores.get(k) for k in SCORE_KEYS},
                "comment": scores.get("comment", ""),
            }
            all_rows.append(row)
            completed.add(key)

            # 保存断点
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            time.sleep(0.5)

    # 保存结果
    output_csv = RESULTS_DIR / f"ablation_{timestamp}.csv"
    fieldnames = [
        "question_id", "question_type", "variant", "variant_desc",
        "answer_length", "time_s", *SCORE_KEYS, "comment",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n结果已保存：{output_csv}")

    # 汇总统计
    import numpy as np

    summary = {"timestamp": timestamp, "n_samples": len(sampled), "variants": {}}

    print(f"\n{'='*75}")
    print("消融实验汇总（各变体 overall 平均分）")
    print(f"{'='*75}")
    print(f"{'变体':<6} {'描述':<24} {'题数':>4} {'相关性':>6} {'忠实性':>6} {'完整性':>6} {'技术准确':>8} {'引导性':>6} {'递进性':>6} {'综合':>6}")
    print(f"{'-'*95}")

    variant_overall = {}
    for v in variants:
        v_rows = [r for r in all_rows if r["variant"] == v]
        desc = VARIANT_DESCRIPTIONS[v]

        scores_by_dim = {}
        for dim in SCORE_KEYS:
            vals = [r[dim] for r in v_rows if r.get(dim) is not None]
            scores_by_dim[dim] = round(np.mean(vals), 3) if vals else None

        variant_overall[v] = scores_by_dim.get("overall", 0)
        summary["variants"][v] = {
            "description": desc,
            "n": len(v_rows),
            "scores": scores_by_dim,
        }

        def fmt(dim):
            v_val = scores_by_dim.get(dim)
            return f"{v_val:.2f}" if v_val is not None else "N/A"

        print(
            f"  {v:<4} {desc:<24} {len(v_rows):>4} "
            f"{fmt('relevance'):>6} {fmt('faithfulness'):>6} "
            f"{fmt('completeness'):>6} {fmt('technical_accuracy'):>8} "
            f"{fmt('pedagogical_guidance'):>6} {fmt('progressive_disclosure'):>6} {fmt('overall'):>6}"
        )

    # 计算各模块贡献
    if "A" in variant_overall:
        full_score = variant_overall["A"]
        print(f"\n{'='*75}")
        print("各模块贡献（完整系统 - 消融变体 = 模块贡献）")
        print(f"{'='*75}")
        contributions = {}
        for v in variants:
            if v == "A":
                continue
            ablated_score = variant_overall.get(v, 0)
            contrib = full_score - ablated_score if (full_score and ablated_score) else None
            contributions[v] = contrib
            desc = VARIANT_DESCRIPTIONS[v]
            c_str = f"{contrib:+.3f}" if contrib is not None else "N/A"
            print(f"  {v} ({desc}): Δ = {c_str}")

        summary["contributions"] = {
            v: round(c, 4) if c is not None else None
            for v, c in contributions.items()
        }

    # 配对 Wilcoxon 检验（A vs 其他）
    if "A" in variants:
        try:
            from scipy.stats import wilcoxon

            print(f"\n{'='*75}")
            print("配对 Wilcoxon 检验 (A vs 消融变体)")
            print(f"{'='*75}")

            a_by_qid = {str(r["question_id"]): r.get("overall")
                        for r in all_rows if r["variant"] == "A" and r.get("overall") is not None}

            for v in variants:
                if v == "A":
                    continue
                v_by_qid = {str(r["question_id"]): r.get("overall")
                            for r in all_rows if r["variant"] == v and r.get("overall") is not None}

                common = sorted(set(a_by_qid.keys()) & set(v_by_qid.keys()))
                if len(common) < 5:
                    print(f"  A vs {v}: 配对数不足 ({len(common)})")
                    continue

                sa = [a_by_qid[q] for q in common]
                sv = [v_by_qid[q] for q in common]
                nonzero = [(a, b) for a, b in zip(sa, sv) if a != b]
                if len(nonzero) < 5:
                    print(f"  A vs {v}: 差异样本不足")
                    continue

                stat, p = wilcoxon([a for a, _ in nonzero], [b for _, b in nonzero])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  A vs {v} ({VARIANT_DESCRIPTIONS[v]}): p={p:.6f} {sig}  (n={len(common)})")

                summary["variants"][v]["wilcoxon_p"] = round(p, 6)
                summary["variants"][v]["significance"] = sig

        except ImportError:
            print("\n[提示] 未安装 scipy，跳过统计检验")

    # 保存汇总 JSON
    summary_json = RESULTS_DIR / f"ablation_{timestamp}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存：{summary_json}")

    # 可视化：消融贡献柱状图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：各变体综合分数
        ax1 = axes[0]
        v_names = [f"{v}\n{VARIANT_DESCRIPTIONS[v]}" for v in variants]
        v_scores = [variant_overall.get(v, 0) or 0 for v in variants]
        colors = ["#2ecc71" if v == "A" else "#e74c3c" if v == "F" else "#3498db" for v in variants]
        bars = ax1.bar(range(len(variants)), v_scores, color=colors, alpha=0.8, edgecolor="white")
        ax1.set_xticks(range(len(variants)))
        ax1.set_xticklabels(v_names, fontsize=8)
        ax1.set_ylabel("综合分数 (1-5)")
        ax1.set_title("各消融变体综合分数")
        ax1.set_ylim(0, 5.5)
        ax1.grid(axis="y", alpha=0.3)
        for bar, score in zip(bars, v_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{score:.2f}", ha="center", va="bottom", fontsize=9)

        # 右图：各模块贡献（堆叠柱状图）
        ax2 = axes[1]
        if "A" in variant_overall and contributions:
            contrib_variants = [v for v in variants if v != "A" and v != "F"]
            contrib_values = [contributions.get(v, 0) or 0 for v in contrib_variants]
            contrib_labels = [f"{v}\n{VARIANT_DESCRIPTIONS[v]}" for v in contrib_variants]

            bar_colors = ["#e74c3c" if c > 0 else "#3498db" for c in contrib_values]
            ax2.bar(range(len(contrib_variants)), contrib_values, color=bar_colors, alpha=0.8)
            ax2.set_xticks(range(len(contrib_variants)))
            ax2.set_xticklabels(contrib_labels, fontsize=8)
            ax2.set_ylabel("贡献量 (Δ overall)")
            ax2.set_title("各模块对综合分数的贡献\n(正值 = 移除后分数下降)")
            ax2.axhline(y=0, color="black", linewidth=0.5)
            ax2.grid(axis="y", alpha=0.3)
            for i, (c, v) in enumerate(zip(contrib_values, contrib_variants)):
                ax2.text(i, c + 0.02 if c >= 0 else c - 0.05,
                         f"{c:+.2f}", ha="center", va="bottom" if c >= 0 else "top", fontsize=9)

        fig.suptitle("消融实验结果", fontsize=14, y=1.02)
        fig.tight_layout()
        png_path = RESULTS_DIR / f"ablation_{timestamp}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"图表已保存：{png_path}")

    except ImportError:
        print("[提示] 未安装 matplotlib，跳过图表生成")


if __name__ == "__main__":
    main()
