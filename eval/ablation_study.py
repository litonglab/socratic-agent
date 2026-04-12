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
  G: -RAG检索（禁用检索工具，保留苏格拉底策略与拓扑）

分析方法：
  单模块消融 B/C/D/E/G 各只移除一个模块，Δ_X = A − X 为该模块的独立移除效应。
  F 同时移除所有模块，(A − F) 为联合效应。
  交互效应 = (A − F) − Σ Δ_X，反映模块间协同或冗余。

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

# 必须在所有 ML 库（torch / tokenizers / sentence-transformers）导入前设置，
# 防止 loky/joblib 在多线程环境里 fork 子进程（macOS segfault 根因）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import csv
import time
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from storage.user_store import create_user
from storage.proficiency import upsert_proficiency_score

# ── 配置 ─────────────────────────────────────────────
QA_DATASET = ROOT / "eval" / "qa_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

SCORE_KEYS = [
    "relevance", "faithfulness", "completeness", "technical_accuracy",
    "pedagogical_guidance", "progressive_disclosure", "overall",
]

OVERALL_WEIGHTS = {
    "relevance": 0.15,
    "faithfulness": 0.20,
    "completeness": 0.15,
    "technical_accuracy": 0.15,
    "pedagogical_guidance": 0.20,
    "progressive_disclosure": 0.15,
}


def _compute_overall(scores: Dict[str, Any]) -> Optional[float]:
    """按 OVERALL_WEIGHTS 加权计算 overall（浮点，不做整数舍入）。

    保留浮点可保证 avg(overall_i) == sum(w_j * avg(dim_j))，
    使汇总表格与公式严格自洽。
    """
    vals = [scores.get(k) for k in OVERALL_WEIGHTS]
    if any(v is None for v in vals):
        return None
    return round(sum(v * w for v, w in zip(vals, OVERALL_WEIGHTS.values())), 3)


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

────────────────────────────────────────────────────

请严格按以下 JSON 格式输出，不要有任何其他文字：
{{
  "relevance": <1-5>,
  "faithfulness": <1-5>,
  "completeness": <1-5>,
  "technical_accuracy": <1-5>,
  "pedagogical_guidance": <1-5>,
  "progressive_disclosure": <1-5>,
  "comment": "<评分简要理由，不超过 60 字>"
}}"""

BASELINE_SYSTEM = "你是一个计算机网络实验课程的助教，请直接、完整地回答学生的问题。"

# ── 评测用模拟学生 ────────────────────────────────────
EVAL_USER_ID = "__ablation_eval_student__"
EVAL_USER_PROFILE = {
    "scores": {
        "LAB_TROUBLESHOOTING": 0.45,
        "THEORY_CONCEPT": 0.40,
        "CONFIG_REVIEW": 0.42,
        "CALCULATION": 0.38,
        "OVERALL": 0.42,
    },
    "confidence": 0.70,
    "interaction_count": 18,
}


def _ensure_eval_student():
    """在 SQLite 中确保评测用模拟学生存在。"""
    try:
        create_user({
            "id": EVAL_USER_ID,
            "username": EVAL_USER_ID,
            "password_salt": "eval",
            "password_hash": "eval",
        })
    except Exception:
        pass
    for cat, score in EVAL_USER_PROFILE["scores"].items():
        upsert_proficiency_score(
            user_id=EVAL_USER_ID,
            category=cat,
            score=score,
            confidence=EVAL_USER_PROFILE["confidence"],
            interaction_count=EVAL_USER_PROFILE["interaction_count"],
        )


# ── 多轮对话模拟 ─────────────────────────────────────

STUDENT_FOLLOWUP_PROMPT = """\
你正在扮演一个计算机网络实验课的学生，基础较薄弱。
老师刚刚回答了你的问题，请根据老师的回答，用一句简短的中文提出一个自然的跟进问题。
要求：
- 如果老师的回答里有你不太懂的术语或步骤，就追问那个点
- 如果老师给了引导性问题，就尝试回答并请求确认
- 不要重复老师已经回答清楚的内容
- 只输出学生的一句话，不要加任何前缀或解释

老师的回答：
{answer}"""

MULTI_TURN_ROUNDS = 3


def _generate_student_followup(llm, teacher_answer: str) -> str:
    """用 DeepSeek 根据上一轮回答动态生成学生跟进。"""
    prompt = STUDENT_FOLLOWUP_PROMPT.format(answer=teacher_answer[:2000])
    messages = [HumanMessage(content=prompt)]
    resp = llm.invoke(messages)
    return (resp.content or "").strip() or "我还是不太明白，能再解释一下吗？"


def _multi_turn_query(
    question: str,
    state: dict,
    n_rounds: int = 3,
    user_id: Optional[str] = None,
    llm=None,
) -> str:
    """多轮对话：动态生成学生跟进，返回所有轮次答案的拼接。"""
    history = []
    all_answers = []

    answer, history, _, state = agent_query(
        question, history=history, state=state, user_id=user_id,
    )
    all_answers.append(answer)

    for _ in range(n_rounds - 1):
        followup = _generate_student_followup(llm, answer)
        answer, history, _, state = agent_query(
            followup, history=history, state=state, user_id=user_id,
        )
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
    "G": "-RAG检索（多轮）",
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


def call_variant_A(question: str, llm, n_rounds: int = 3) -> str:
    """完整系统（多轮对话）：传 user_id 启用水平自适应。"""
    return _multi_turn_query(
        question, state={}, n_rounds=n_rounds,
        user_id=EVAL_USER_ID, llm=llm,
    )


def call_variant_B(question: str, llm, n_rounds: int = 3) -> str:
    """-Reranker（多轮）。全局开关由批级别管理，此处直接调用。"""
    return _multi_turn_query(
        question, state={}, n_rounds=n_rounds,
        user_id=EVAL_USER_ID, llm=llm,
    )


def call_variant_C(question: str) -> str:
    """-Socratic：强制最高 hint_level，直接给完整答案（单轮）。"""
    state = {"hint_level": 99}
    answer, _, _, _ = agent_query(question, history=[], state=state)
    return answer


def call_variant_D(question: str, llm, n_rounds: int = 3) -> str:
    """-拓扑（多轮）。monkey-patch 由批级别管理，此处直接调用。"""
    return _multi_turn_query(
        question, state={}, n_rounds=n_rounds,
        user_id=EVAL_USER_ID, llm=llm,
    )


def call_variant_E(question: str, llm, n_rounds: int = 3) -> str:
    """-水平自适应：不传 user_id，无个性化初始 hint level（多轮）。"""
    return _multi_turn_query(
        question, state={}, n_rounds=n_rounds,
        user_id=None, llm=llm,
    )


def call_variant_F(question: str, llm) -> str:
    """纯 LLM 基线：裸 DeepSeek，无 RAG。"""
    messages = [
        SystemMessage(content=BASELINE_SYSTEM),
        HumanMessage(content=question),
    ]
    resp = llm.invoke(messages)
    return resp.content


def call_variant_G(question: str, llm, n_rounds: int = 3) -> str:
    """-RAG检索（多轮）。禁用检索工具但保留苏格拉底策略与拓扑，monkey-patch 由批级别管理。"""
    return _multi_turn_query(
        question, state={}, n_rounds=n_rounds,
        user_id=EVAL_USER_ID, llm=llm,
    )


VARIANT_CALLERS = {
    "A": lambda q, llm, **kw: call_variant_A(q, llm, n_rounds=kw.get("rounds", 3)),
    "B": lambda q, llm, **kw: call_variant_B(q, llm, n_rounds=kw.get("rounds", 3)),
    "C": lambda q, llm, **kw: call_variant_C(q),
    "D": lambda q, llm, **kw: call_variant_D(q, llm, n_rounds=kw.get("rounds", 3)),
    "E": lambda q, llm, **kw: call_variant_E(q, llm, n_rounds=kw.get("rounds", 3)),
    "F": lambda q, llm, **kw: call_variant_F(q, llm),
    "G": lambda q, llm, **kw: call_variant_G(q, llm, n_rounds=kw.get("rounds", 3)),
}


def judge_answer(client: OpenAI, question: str, answer: str, reference: str) -> Dict[str, Any]:
    """GPT-4o 打分（端到端评估，不需要检索上下文）。overall 由代码按权重计算。"""
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference=reference,
        answer=answer[:12000],
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            timeout=60,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        scores["overall"] = _compute_overall(scores)
        return scores
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
    parser.add_argument("--variants", type=str, default="A,B,C,D,E,F,G",
                        help="要运行的变体 (逗号分隔)")
    parser.add_argument("--resume", action="store_true", help="从断点续跑")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3,
                        help="变体A多轮对话轮数 (默认 3)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="每个变体的并行线程数 (默认 4)")
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

    # 写入评测用模拟学生档案（水平自适应对照需要）
    _ensure_eval_student()
    print(f"  评测学生档案: {EVAL_USER_ID} (OVERALL={EVAL_USER_PROFILE['scores']['OVERALL']:.2f})")

    # 主线程预热：在线程池启动前完整初始化 RAG + Reranker，
    # 避免 loky/joblib 在多线程环境里 fork 子进程（macOS segfault）
    print("  [预热] 触发 RAG 初始化...", flush=True)
    from agentic_rag.rag import _ensure_rag_initialized, RAGAgent
    _ensure_rag_initialized()
    try:
        RAGAgent("网络实验", category="LAB_TROUBLESHOOTING", hint_level=0)
    except Exception:
        pass
    print("  [预热] RAG 就绪", flush=True)

    # 断点处理
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "ablation_checkpoint.jsonl"

    completed = set()
    all_rows = []
    if args.resume and checkpoint_path.exists():
        _skipped = 0
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "api_error":
                        _skipped += 1
                        continue
                    all_rows.append(row)
                    completed.add((str(row["question_id"]), row["variant"]))
        print(f"  [断点续跑] 已加载 {len(all_rows)} 条有效结果，跳过 {_skipped} 条失败记录（将重试）")

    # ── 按变体分批、批内并行评测 ─────────────────────────
    total = len(sampled) * len(variants)
    done_counter = {"n": len(completed)}
    print_lock = threading.Lock()
    checkpoint_lock = threading.Lock()

    # 每个线程独立持有一个 DeepSeekChatClient（requests.Session 非线程安全）
    _thread_local = threading.local()

    def _get_thread_llm():
        if not hasattr(_thread_local, "llm"):
            _thread_local.llm = build_chat_llm(temperature=0)
        return _thread_local.llm

    def _run_one(q_item, variant):
        """单个 (题目, 变体) 任务，线程安全。"""
        qid = q_item.get("id", "?")
        question = q_item.get("question", "")
        qtype = q_item.get("type", "unknown")
        reference = q_item.get("reference", "")
        desc = VARIANT_DESCRIPTIONS[variant]
        thread_llm = _get_thread_llm()

        answer = None
        elapsed = 0
        _max_retries = 4
        for _attempt in range(_max_retries + 1):
            try:
                t0 = time.time()
                answer = VARIANT_CALLERS[variant](question, thread_llm, rounds=args.rounds)
                elapsed = round(time.time() - t0, 1)
                break
            except Exception as e:
                if _attempt < _max_retries:
                    wait = min(2 ** _attempt * 5, 60) + random.uniform(0, 3)
                    with print_lock:
                        print(f"\n    [{variant} Q{qid} 重试 {_attempt+1}/{_max_retries}] {e}，"
                              f"等待 {wait:.0f}s...", flush=True)
                    time.sleep(wait)
                else:
                    with print_lock:
                        print(f"\n    [{variant} Q{qid}] 所有重试耗尽: {e}", flush=True)

        if answer is None:
            fail_row = {
                "question_id": qid, "question_type": qtype,
                "variant": variant, "variant_desc": desc,
                "status": "api_error",
            }
            with checkpoint_lock:
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(fail_row, ensure_ascii=False) + "\n")
            with print_lock:
                done_counter["n"] += 1
                print(f"  [{done_counter['n']}/{total}] {variant} Q{qid}: NA (api_error)", flush=True)
            return None

        scores = judge_answer(openai_client, question, answer, reference)

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
        with checkpoint_lock:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with print_lock:
            done_counter["n"] += 1
            print(f"  [{done_counter['n']}/{total}] {variant} Q{qid}: "
                  f"overall={scores.get('overall', '?')} ({elapsed}s)", flush=True)
        return row

    max_workers = max(1, args.max_workers)

    for variant in variants:
        pending = [q for q in sampled if (str(q.get("id", "?")), variant) not in completed]
        if not pending:
            print(f"\n── 变体 {variant} ({VARIANT_DESCRIPTIONS[variant]}): 全部已完成，跳过")
            continue

        desc = VARIANT_DESCRIPTIONS[variant]
        print(f"\n{'='*60}")
        print(f"变体 {variant} ({desc}): {len(pending)} 题待评测，并行={min(max_workers, len(pending))}")
        print(f"{'='*60}")

        # B / D / G 需要在批级别设置全局状态
        _saved_reranker = None
        _saved_prepare = None
        if variant == "B":
            from agentic_rag import rag as rag_mod
            _saved_reranker = rag_mod.DISABLE_RERANKER
            rag_mod.DISABLE_RERANKER = True
            rag_mod._retriever_cache.clear()
        elif variant == "D":
            from agentic_rag import agent as agent_mod
            _saved_prepare = agent_mod._prepare_context
            _original_prepare_d = agent_mod._prepare_context

            def _patched_prepare_d(*args, **kwargs):
                ctx = _original_prepare_d(*args, **kwargs)
                if ctx.contextual_actions and "拓扑" in ctx.contextual_actions:
                    ctx.contextual_actions["拓扑"] = lambda _input: "拓扑工具当前不可用，请根据已有知识回答。"
                return ctx
            agent_mod._prepare_context = _patched_prepare_d
        elif variant == "G":
            from agentic_rag import agent as agent_mod
            _saved_prepare = agent_mod._prepare_context
            _original_prepare_g = agent_mod._prepare_context

            def _patched_prepare_g(*args, **kwargs):
                ctx = _original_prepare_g(*args, **kwargs)
                if ctx.contextual_actions and "检索" in ctx.contextual_actions:
                    ctx.contextual_actions["检索"] = lambda _input: "检索工具当前不可用，请根据已有知识回答。"
                return ctx
            agent_mod._prepare_context = _patched_prepare_g

        try:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as executor:
                futures = {executor.submit(_run_one, q, variant): q for q in pending}
                for future in as_completed(futures):
                    row = future.result()
                    if row is not None:
                        all_rows.append(row)
                        completed.add((str(row["question_id"]), variant))
        finally:
            if variant == "B" and _saved_reranker is not None:
                from agentic_rag import rag as rag_mod
                rag_mod.DISABLE_RERANKER = _saved_reranker
                rag_mod._retriever_cache.clear()
            elif variant in ("D", "G") and _saved_prepare is not None:
                from agentic_rag import agent as agent_mod
                agent_mod._prepare_context = _saved_prepare

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

    # ── 正交消融分解：单模块移除效应 + 交互效应 ───────────
    # 每个单模块消融变体（B/C/D/E/G）只移除一个模块，A − X 即为该模块的
    # 独立移除效应 Δ_X；而 F 同时移除所有模块，(A − F) 是联合效应。
    # 交互效应 = (A − F) − Σ Δ_X，反映模块间的协同或冗余。
    SINGLE_REMOVAL_VARIANTS = {
        "B": "Reranker",
        "C": "苏格拉底策略",
        "D": "拓扑模块",
        "E": "水平自适应",
        "G": "RAG检索",
    }

    contributions = {}   # 保持向后兼容的 key
    single_effects = {}  # variant -> delta

    if "A" in variant_overall:
        full_score = variant_overall["A"]
        print(f"\n{'='*75}")
        print("单模块移除效应 Δ（完整系统 − 消融变体 = 移除该模块后的综合分下降）")
        print(f"{'='*75}")

        for v, module_name in SINGLE_REMOVAL_VARIANTS.items():
            if v not in variant_overall:
                continue
            ablated = variant_overall[v]
            delta = round(full_score - ablated, 4) if (full_score and ablated) else None
            single_effects[v] = delta
            contributions[v] = delta
            d_str = f"{delta:+.3f}" if delta is not None else "N/A"
            print(f"  Δ_{module_name:<8s} (A − {v}) = {d_str}")

        # 交互效应分析
        if "F" in variant_overall and variant_overall["F"] is not None:
            f_score = variant_overall["F"]
            total_gap = round(full_score - f_score, 4) if full_score else None
            valid_deltas = [d for d in single_effects.values() if d is not None]
            sum_individual = round(sum(valid_deltas), 4) if valid_deltas else None

            print(f"\n  {'─'*60}")
            if total_gap is not None:
                print(f"  联合效应 (A − F)       = {total_gap:+.3f}")
            if sum_individual is not None:
                print(f"  单模块效应之和 Σ Δ_X   = {sum_individual:+.3f}")
            if total_gap is not None and sum_individual is not None:
                interaction = round(total_gap - sum_individual, 4)
                print(f"  交互效应 (协同/冗余)   = {interaction:+.3f}")
                if abs(interaction) > 0.01:
                    direction = "正向协同（整体 > 各部分之和）" if interaction > 0 else "负向冗余（整体 < 各部分之和）"
                    print(f"  → 模块间存在{direction}")
                summary["interaction_effect"] = interaction
            contributions["F"] = total_gap

        summary["single_removal_effects"] = {
            v: round(d, 4) if d is not None else None
            for v, d in single_effects.items()
        }
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

        # 右图：单模块移除效应 + 交互效应（正交消融分解）
        ax2 = axes[1]
        if "A" in variant_overall and single_effects:
            bar_labels = []
            bar_values = []
            bar_colors_r = []

            for v, module_name in SINGLE_REMOVAL_VARIANTS.items():
                if v not in single_effects or single_effects[v] is None:
                    continue
                bar_labels.append(f"Δ_{module_name}")
                bar_values.append(single_effects[v])
                bar_colors_r.append("#e74c3c" if single_effects[v] > 0 else "#3498db")

            if summary.get("interaction_effect") is not None:
                bar_labels.append("交互效应")
                bar_values.append(summary["interaction_effect"])
                bar_colors_r.append("#9b59b6")

            ax2.bar(range(len(bar_labels)), bar_values, color=bar_colors_r, alpha=0.8, edgecolor="white")
            ax2.set_xticks(range(len(bar_labels)))
            ax2.set_xticklabels(bar_labels, fontsize=7, rotation=15, ha="right")
            ax2.set_ylabel("Δ overall")
            ax2.set_title("正交消融分解\n(单模块移除效应 + 交互效应)")
            ax2.axhline(y=0, color="black", linewidth=0.5)
            ax2.grid(axis="y", alpha=0.3)
            for i, val in enumerate(bar_values):
                ax2.text(i, val + 0.02 if val >= 0 else val - 0.05,
                         f"{val:+.2f}", ha="center",
                         va="bottom" if val >= 0 else "top", fontsize=9)

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
