import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agentic_rag.rag import RAGAgent
from agentic_rag.web_search import WebSearch
from dataclasses import dataclass
from agentic_rag.utils import _coerce_to_text
from agentic_rag.llm_config import build_chat_llm
from .prompts import (
    BASE_PROMPT_LAB,
    BASE_PROMPT_THEORY,
    BASE_PROMPT_REVIEW,
    BASE_PROMPT_CALC,
    STRATEGY_LAB,
    STRATEGY_THEORY,
    STRATEGY_REVIEW,
    STRATEGY_CALC,
    UNIFIED_CLASSIFICATION_PROMPT,
    BASE_PROMPT_GENERAL,
)

# topo_rag 依赖 PIL / python-docx / OpenAI 等组件，延迟导入避免拖慢启动
_topo_retriever = None
_topo_lock = threading.Lock()


def _get_topo_retriever():
    global _topo_retriever
    if _topo_retriever is None:
        with _topo_lock:
            if _topo_retriever is None:
                from agentic_rag.topo_rag import TopoRetriever
                _topo_retriever = TopoRetriever
    return _topo_retriever


# 各分类的 hint_level 上限
_MAX_HINT_LEVEL = {
    "LAB_TROUBLESHOOTING": 3,   # 4 级：诊断→根因→修复指导→完整解决
    "THEORY_CONCEPT": 1,        # 2 级：简明解释→全面讲透
    "CONFIG_REVIEW": 2,         # 3 级：定位错误→原因+修正→完整方案
    "CALCULATION": 1,           # 2 级：公式+思路→完整演算
}

_BASE_PROMPTS = {
    "LAB_TROUBLESHOOTING": BASE_PROMPT_LAB,
    "THEORY_CONCEPT": BASE_PROMPT_THEORY,
    "CONFIG_REVIEW": BASE_PROMPT_REVIEW,
    "CALCULATION": BASE_PROMPT_CALC,
}

_STRATEGIES = {
    "LAB_TROUBLESHOOTING": STRATEGY_LAB,
    "THEORY_CONCEPT": STRATEGY_THEORY,
    "CONFIG_REVIEW": STRATEGY_REVIEW,
    "CALCULATION": STRATEGY_CALC,
}

_SECONDARY_LABEL = {
    "CALCULATION": "计算与分析",
    "LAB_TROUBLESHOOTING": "实验故障排查",
    "CONFIG_REVIEW": "配置操作与审查",
    "THEORY_CONCEPT": "基础概念与原理",
}

_IDENTITY_KEYWORDS = ["你是谁", "你是什么", "who are you", "what are you", "自我介绍", "介绍一下你自己"]
_IDENTITY_REPLY = "我是计算机网络实验课 AI 助教，基于大语言模型技术，并经过课程知识库的专属优化。我可以帮你理解网络理论、排查实验故障、审查配置和辅导计算题。有什么网络问题可以问我！"

action_re = re.compile(r'^工具：(\w+)：(.*)$')
_EXPERIMENT_ID_RE = re.compile(r"(?:实验\s*|lab[\s_-]?)(\d+)", re.IGNORECASE)


class Evidence(TypedDict):
    id: str
    query: str
    excerpt: str
    raw_text: str


class AgentState(TypedDict, total=False):
    user_message: str
    evidences: List[Evidence]
    hint_level: int
    user_turn_count: int
    turns_at_current_level: int
    lab_turn_count: int
    question_category: str
    mode: str
    experiment_id: str
    experiment_label: str


# 延迟初始化 LLM 客户端
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = build_chat_llm(temperature=0)
    return _client


# -------------------------------------------------------------------------
# 辅助函数
# -------------------------------------------------------------------------

def _format_history_context(history: List[BaseMessage], limit: int = 3) -> str:
    if not history:
        return "（无历史对话）"
    context_str = ""
    recent_msgs = history[-(limit * 2):]
    for m in recent_msgs:
        role = "AI" if isinstance(m, AIMessage) else "User"
        content = m.content[:200] + "..." if len(m.content) > 200 else m.content
        context_str += f"{role}: {content}\n"
    return context_str


def _get_base_prompt(category: str) -> str:
    return _BASE_PROMPTS.get(category, BASE_PROMPT_LAB)


def _get_strategy_prompt(level: int, category: str) -> str:
    target_dict = _STRATEGIES.get(category, STRATEGY_LAB)
    return target_dict.get(level, target_dict[max(target_dict.keys())])


def _format_citations(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return ""
    lines = [f"[{c.get('id')}] {c.get('source', 'unknown')}" for c in citations]
    return "引用：\n" + "\n".join(lines)


def _extract_experiment_id(text: str) -> Optional[str]:
    match = _EXPERIMENT_ID_RE.search(text or "")
    if not match:
        return None
    return f"lab{int(match.group(1))}"


def _experiment_label(experiment_id: str) -> str:
    match = re.search(r"lab(\d+)$", experiment_id or "", re.IGNORECASE)
    if match:
        return f"实验{int(match.group(1))}"
    return experiment_id


def _resolve_experiment_context(
    question: str,
    history: List[BaseMessage],
    state: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    current = _extract_experiment_id(question)
    if current:
        label = _experiment_label(current)
        state["experiment_id"] = current
        state["experiment_label"] = label
        return current, label

    if state.get("experiment_id"):
        experiment_id = state.get("experiment_id")
        label = state.get("experiment_label") or _experiment_label(experiment_id)
        state["experiment_label"] = label
        return experiment_id, label

    for msg in reversed(history):
        experiment_id = _extract_experiment_id(getattr(msg, "content", ""))
        if experiment_id:
            label = _experiment_label(experiment_id)
            state["experiment_id"] = experiment_id
            state["experiment_label"] = label
            return experiment_id, label

    return None, None


# -------------------------------------------------------------------------
# 三合一统一分类
# -------------------------------------------------------------------------

def _parse_category(raw) -> str:
    s = (raw or "").upper()
    if "CALC" in s:
        return "CALCULATION"
    if "LAB" in s or "TROUBLE" in s:
        return "LAB_TROUBLESHOOTING"
    if "REVIEW" in s or "CONFIG" in s:
        return "CONFIG_REVIEW"
    if "THEORY" in s or "CONCEPT" in s:
        return "THEORY_CONCEPT"
    return ""


_DEFAULT_CLASSIFICATION = {"relevance": True, "category": "LAB_TROUBLESHOOTING", "secondary_categories": [], "hint_decision": "MAINTAIN"}


def classify_unified(
    question: str,
    history: List[BaseMessage],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """单次 LLM 调用同时返回 relevance / category / secondary_categories / hint_decision。"""
    q_str = (question or "").strip()

    greetings = ["你好", "hello", "hi", "谢谢", "再见", "help", "救命", "提示", "what is"]
    if not q_str:
        return dict(_DEFAULT_CLASSIFICATION)
    if len(q_str) < 15 and any(g in q_str.lower() for g in greetings):
        return dict(_DEFAULT_CLASSIFICATION)

    current_level = state.get("hint_level", 0)
    context_str = _format_history_context(history, limit=2)
    prompt_content = f"【对话历史】:\n{context_str}\n\n【用户当前输入】: {q_str}"
    system_prompt = UNIFIED_CLASSIFICATION_PROMPT.format(current_level=current_level)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ]

    try:
        response = _get_client().invoke(messages)
        content = response.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError(f"No JSON found in response: {content}")

        relevance = "NO" not in (data.get("relevance", "YES") or "YES").upper()

        raw_cat = data.get("category", [])
        if isinstance(raw_cat, str):
            raw_cat = [raw_cat]
        parsed = [c for c in (_parse_category(c) for c in raw_cat) if c]
        if not parsed:
            parsed = ["LAB_TROUBLESHOOTING"]

        primary_category = parsed[0]
        secondary_categories = [c for c in parsed[1:] if c != primary_category]

        raw_hint = (data.get("hint_decision", "MAINTAIN") or "").upper()
        if "JUMP_TO_MAX" in raw_hint:
            hint_decision = "JUMP_TO_MAX"
        elif "INCREASE" in raw_hint:
            hint_decision = "INCREASE"
        else:
            hint_decision = "MAINTAIN"

        return {
            "relevance": relevance,
            "category": primary_category,
            "secondary_categories": secondary_categories,
            "hint_decision": hint_decision,
        }

    except Exception as e:
        print(f"[Warning] Unified classification failed: {e}, using safe defaults.")
        return dict(_DEFAULT_CLASSIFICATION)


# -------------------------------------------------------------------------
# Agent 类
# -------------------------------------------------------------------------

class Agent():
    def __init__(self, prompt, history):
        self.messages = []
        if prompt:
            self.messages.append(SystemMessage(content=prompt))
        if history:
            self.messages.extend(history)

    def __call__(self, message):
        self.messages.append(HumanMessage(content=message))
        result = self.execute()
        self.messages.append(AIMessage(content=result))
        return result

    def execute(self):
        response = _get_client().invoke(self.messages)
        return response.content

    def execute_stream(self):
        """流式版 execute()，逐 token yield，完成后保存完整文本。"""
        full_text = ""
        for token in _get_client().invoke_stream(self.messages):
            full_text += token
            yield token
        self._last_stream_result = full_text


# -------------------------------------------------------------------------
# _prepare_context：query() 和 query_stream() 的共用前置逻辑
# -------------------------------------------------------------------------

@dataclass
class _QueryContext:
    """_prepare_context 的返回值，封装分类、Hint、Prompt 等预处理结果。"""
    # 快速拦截结果（身份/无关问题），非 None 时直接返回此回复
    early_reply: Optional[str]
    # 是否走通用 LLM（无关问题，不使用 RAG/工具）
    use_general_llm: bool
    # 以下字段仅在非快速拦截时有效
    final_prompt: str
    category: str
    hint_level: int
    contextual_actions: Dict[str, Any]
    debug_info: Optional[str]


def _prepare_context(
    question: str,
    history: List[BaseMessage],
    state: Dict[str, Any],
    user_id: Optional[str],
    enable_websearch: bool,
    debug: bool,
) -> _QueryContext:
    """
    query() 和 query_stream() 共用的前置逻辑：
    state 初始化、身份拦截、分类、Hint Level 计算、Prompt 组装。
    会原地修改 state。
    """
    # 初始化 state 计数器
    state.setdefault("turns_at_current_level", 0)
    state.setdefault("user_turn_count", 0)
    state.setdefault("lab_turn_count", 0)

    # 水平个性化：新会话根据学生水平设置初始 Hint Level
    if "hint_level" not in state and user_id:
        from storage.proficiency import get_initial_hint_level
        state["hint_level"] = get_initial_hint_level(user_id)

    state["user_turn_count"] += 1

    q = (question or "").strip()
    q_low = q.lower()

    # 身份类问题快速拦截
    if len(q) < 20 and any(kw in q_low for kw in _IDENTITY_KEYWORDS):
        return _QueryContext(
            early_reply=_IDENTITY_REPLY,
            use_general_llm=False,
            final_prompt="", category="", hint_level=0,
            contextual_actions={}, debug_info=None,
        )

    # 三合一分类
    classification = classify_unified(q, history, state)

    # 无关问题 → 通用 LLM
    if not classification["relevance"]:
        if debug:
            print(f"[General] Routing irrelevant query to general LLM: '{question}'")
        return _QueryContext(
            early_reply=None,
            use_general_llm=True,
            final_prompt=BASE_PROMPT_GENERAL,
            category="", hint_level=0,
            contextual_actions={}, debug_info=None,
        )

    # 问题分类
    category = classification["category"]
    if category == "LAB_TROUBLESHOOTING":
        state["lab_turn_count"] += 1
    state["question_category"] = category

    experiment_id, experiment_label = _resolve_experiment_context(q, history, state)

    # Hint Level 计算（含 failsafe 逻辑）
    current_level = state.get("hint_level", 0)
    state["_hint_level_start"] = current_level
    turns_at_level = state.get("turns_at_current_level", 0)
    max_level = _MAX_HINT_LEVEL.get(category, 3)

    if current_level >= max_level:
        current_hint_level = max_level
        state["turns_at_current_level"] = turns_at_level + 1
    else:
        hint_decision = classification["hint_decision"]
        if hint_decision == "JUMP_TO_MAX":
            current_hint_level = max_level
            turns_at_level = 0
            print(f"[Hint Logic] AI decided to JUMP_TO_MAX -> level {max_level}")
        elif hint_decision == "INCREASE":
            current_hint_level = current_level + 1
            turns_at_level = 0
            print(f"[Hint Logic] AI decided to INCREASE to level {current_hint_level}")
        else:
            turns_at_level += 1
            if turns_at_level >= 3:
                current_hint_level = current_level + 1
                turns_at_level = 0
                print(f"[Hint Logic] Failsafe triggered (3 turns stuck). FORCED INCREASE to level {current_hint_level}")
            else:
                current_hint_level = current_level
                print(f"[Hint Logic] Maintaining level {current_level} (Streak: {turns_at_level}/3)")
        current_hint_level = min(current_hint_level, max_level)
        state["turns_at_current_level"] = turns_at_level

    # LAB 3 轮强制收敛
    if category == "LAB_TROUBLESHOOTING" and state["lab_turn_count"] >= 3:
        current_hint_level = max_level
    state["hint_level"] = current_hint_level

    # 水平采集记录
    state["_hint_decision"] = classification.get("hint_decision", "MAINTAIN")
    state["_was_failsafe"] = bool(
        classification.get("hint_decision") != "INCREASE"
        and current_hint_level > current_level
    )

    # 组装最终 Prompt
    base_prompt_template = _get_base_prompt(category)
    strategy_instruction = _get_strategy_prompt(current_hint_level, category)
    final_prompt = base_prompt_template.format(current_strategy_instruction=strategy_instruction)

    if category == "LAB_TROUBLESHOOTING" and state["lab_turn_count"] >= 3:
        final_prompt += (
            "\n\n【三轮硬约束】\n"
            "这是第 3 轮对话。你必须在本轮收敛：给出可执行的排查路径与最关键的操作点，"
            "不强制学生再回答。可以附 1 个可选问题，但不能依赖学生回答才能推进。"
        )

    # 注入次要分类任务（复合问题）
    secondary_categories = classification.get("secondary_categories", [])
    if secondary_categories:
        secondary_desc = "、".join(_SECONDARY_LABEL.get(c, c) for c in secondary_categories)
        final_prompt += (
            f"\n\n【复合问题补充任务】\n"
            f"本问题同时涉及以下次要诉求：{secondary_desc}。\n"
            f"请在完成主要任务后，按顺序简要处理上述次要诉求。次要任务无需完整展开，以满足学生当前的实际需要为准。"
        )

    # 注入学生水平上下文
    if user_id:
        from storage.proficiency import get_proficiency_summary
        prof_summary = get_proficiency_summary(user_id)
        if prof_summary:
            final_prompt += f"\n\n【学生水平参考】\n{prof_summary}"

    if experiment_id and experiment_label:
        final_prompt += (
            f"\n\n【实验上下文】\n"
            f"当前会话已识别为 {experiment_label}（{experiment_id}）。"
            f"如果需要调用拓扑工具，只使用该实验下审核通过的拓扑 JSON，不要混用其他实验数据。"
        )

    debug_info = None
    if debug:
        debug_info = (
            f"[DEBUG] Cat: {category} | Secondary: {secondary_categories} | "
            f"Lvl: {current_hint_level} | Exp: {experiment_id or '-'}"
        )
        print(debug_info)
        print(f"[DEBUG] Strategy: {strategy_instruction[:50]}...")

    # 构建工具表
    def _rag_with_context(msg: str):
        return RAGAgent(msg, category=category, hint_level=current_hint_level)

    contextual_actions = {
        "检索": _rag_with_context,
        "拓扑": lambda q: _get_topo_retriever()(q, experiment_id=state.get("experiment_id")),
    }
    if enable_websearch:
        contextual_actions["搜索"] = WebSearch

    return _QueryContext(
        early_reply=None,
        use_general_llm=False,
        final_prompt=final_prompt,
        category=category,
        hint_level=current_hint_level,
        contextual_actions=contextual_actions,
        debug_info=debug_info,
    )


# -------------------------------------------------------------------------
# 工具调用循环的共用辅助
# -------------------------------------------------------------------------

def _execute_tool_action(
    action_match,
    contextual_actions: Dict[str, Any],
    tool_traces: List[Dict[str, Any]],
    last_citations: List[Dict[str, Any]],
) -> str:
    """执行单次工具调用，返回观察结果文本。"""
    action, action_input = action_match.groups()
    if action not in contextual_actions:
        raise Exception(f"Unknown action: {action}: {action_input}")

    observation = contextual_actions[action](action_input)
    if isinstance(observation, dict) and observation.get("citations"):
        last_citations.clear()
        last_citations.extend(observation.get("citations") or [])

    obs_output_str = _coerce_to_text(observation)
    tool_traces.append({
        "tool": action,
        "input": action_input,
        "output": obs_output_str[:2000],
    })
    return f"检索结果：{obs_output_str}"


def _find_action(text: str):
    """从 LLM 输出中提取第一个工具调用匹配。"""
    for line in (text or "").split("\n"):
        m = action_re.match(line)
        if m:
            return m
    return None


# -------------------------------------------------------------------------
# 主流程 Query（同步版）
# -------------------------------------------------------------------------

def query(
    question: str,
    history: Optional[List[BaseMessage]] = None,
    max_turns: int = 5,
    debug: bool = False,
    state: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    enable_websearch: bool = True,
) -> Tuple[str, List[BaseMessage], List[Dict[str, Any]], Dict[str, Any]]:
    if history is None:
        history = []
    if state is None:
        state = {}

    ctx = _prepare_context(question, history, state, user_id, enable_websearch, debug)
    q = (question or "").strip()

    # 快速拦截（身份类）
    if ctx.early_reply is not None:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=ctx.early_reply))
        return ctx.early_reply, history, [], state

    # 无关问题 → 通用 LLM
    if ctx.use_general_llm:
        bot = Agent(ctx.final_prompt, history)
        reply = bot(q)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply))
        return reply, history, [], state

    # Agent 执行循环
    bot = Agent(ctx.final_prompt, history)
    next_prompt = q
    tool_traces: List[Dict[str, Any]] = []
    last_citations: List[Dict[str, Any]] = []
    final_result = ""

    for i in range(max_turns):
        result = bot(next_prompt)
        final_result = result

        if debug:
            print(f"Turn {i + 1}: {result[:50]}...")

        action_match = _find_action(result)
        if action_match:
            next_prompt = _execute_tool_action(
                action_match, ctx.contextual_actions, tool_traces, last_citations
            )
            continue

        break

    # 追加引用
    if last_citations and "引用：" not in (final_result or ""):
        final_result = (final_result or "").rstrip() + "\n\n" + _format_citations(last_citations)

    if final_result:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=final_result))

    return final_result, history, tool_traces, (state or {})


# -------------------------------------------------------------------------
# 流式版 query
# -------------------------------------------------------------------------

def query_stream(
    question: str,
    history: Optional[List[BaseMessage]] = None,
    max_turns: int = 5,
    debug: bool = False,
    state: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    enable_websearch: bool = True,
):
    """
    流式版 query()。yield 字典：
      {"type": "token", "content": "..."}   — 流式 token
      {"type": "done", "result": str, "history": List, "tool_traces": List, "state": Dict}
    """
    if history is None:
        history = []
    if state is None:
        state = {}

    ctx = _prepare_context(question, history, state, user_id, enable_websearch, debug)
    q = (question or "").strip()

    # 快速拦截（身份类）
    if ctx.early_reply is not None:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=ctx.early_reply))
        yield {"type": "token", "content": ctx.early_reply}
        yield {"type": "done", "result": ctx.early_reply, "history": history,
               "tool_traces": [], "state": state}
        return

    # 无关问题 → 通用 LLM（流式）
    if ctx.use_general_llm:
        bot = Agent(ctx.final_prompt, history)
        bot.messages.append(HumanMessage(content=q))
        final_result = ""
        for token in bot.execute_stream():
            final_result += token
            yield {"type": "token", "content": token}
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=final_result))
        yield {"type": "done", "result": final_result, "history": history,
               "tool_traces": [], "state": state}
        return

    # Agent 循环（流式）
    bot = Agent(ctx.final_prompt, history)
    next_prompt = q
    tool_traces: List[Dict[str, Any]] = []
    last_citations: List[Dict[str, Any]] = []
    final_result = ""

    for i in range(max_turns):
        bot.messages.append(HumanMessage(content=next_prompt))

        accumulated = ""
        found_action = False
        started_forwarding = False

        for token in bot.execute_stream():
            accumulated += token

            if not started_forwarding:
                lines = accumulated.split("\n")
                for line in lines[:-1]:
                    if action_re.match(line.strip()):
                        found_action = True
                        break

                if found_action:
                    break

                if len(lines) > 2 or (len(accumulated) > 80 and "\n" in accumulated):
                    started_forwarding = True
                    yield {"type": "token", "content": accumulated}
            else:
                yield {"type": "token", "content": token}

        full_turn_text = getattr(bot, '_last_stream_result', accumulated)
        bot.messages.append(AIMessage(content=full_turn_text))

        if found_action:
            action_match = _find_action(full_turn_text)
            if action_match:
                next_prompt = _execute_tool_action(
                    action_match, ctx.contextual_actions, tool_traces, last_citations
                )
                continue

        final_result = full_turn_text
        break

    # 引用
    if last_citations and "引用：" not in (final_result or ""):
        citation_text = _format_citations(last_citations)
        final_result = (final_result or "").rstrip() + "\n\n" + citation_text
        yield {"type": "token", "content": "\n\n" + citation_text}

    if final_result:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=final_result))

    yield {"type": "done", "result": final_result, "history": history,
           "tool_traces": tool_traces, "state": state or {}}


# -------------------------------------------------------------------------
# 消息序列化
# -------------------------------------------------------------------------

def messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    out = []
    for m in messages:
        role = "user"
        if isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        out.append({"role": role, "content": m.content})
    return out


def dicts_to_messages(items: List[Dict[str, str]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for it in items:
        role = it.get("role")
        content = it.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


if __name__ == "__main__":
    print("--- Testing Relevance Guardrail ---")
    q1 = "什么是网络协议？"
    history_store = []
    output1, history_store, _, _ = query(q1, history=history_store, debug=True)
    print(f"\n[Q: {q1}] => {output1[:50]}...")

    q2 = "宫保鸡丁怎么做？"
    output2, history_store, _, _ = query(q2, history=history_store, debug=True)
    print(f"\n[Q: {q2}] => {output2}")

    print("\n--- Testing Classifier & Prompt Switching ---")
    q3 = "怎么划分子网？"
    output3, history_store, _, state3 = query(q3, history=history_store, debug=True)
    print(f"\n[Q: {q3}] Cat: {state3.get('question_category')}")
