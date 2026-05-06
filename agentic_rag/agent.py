import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agentic_rag.rag import RAGAgent
from agentic_rag.web_search import WebSearch
from dataclasses import dataclass
from agentic_rag.utils import _coerce_to_text
from agentic_rag.llm_config import build_chat_llm
from agentic_rag.chat_format import split_assistant_content


# 流式 forward 时需要保护的标签前缀：当 visible 末尾正在形成下列任意标签的开头，
# 暂不 forward 这段不完整后缀，等下一帧 token 到来再决定。
_PROTECTED_TAG_PREFIXES = (
    "<思考>",
    "</思考>",
    "<thinking>",
    "</thinking>",
    "<tool_calls>",
    "</tool_calls>",
)


def _strip_unsafe_tail(text: str) -> str:
    """
    若 text 末尾正在形成某个受保护标签（如 "<思考" / "<too" 等不完整片段），
    返回去掉该不完整片段后的安全前缀；否则原样返回。

    保守策略：从最长可能的尾部子串向短的方向找匹配，命中即截断。
    """
    if not text:
        return text
    # 限制扫描深度，避免长度极端时浪费
    max_tag_len = max(len(t) for t in _PROTECTED_TAG_PREFIXES)
    scan_len = min(len(text), max_tag_len)
    for k in range(scan_len, 0, -1):
        suffix = text[-k:]
        # case-insensitive 匹配（兼容 <thinking>）
        suffix_lower = suffix.lower()
        for tag in _PROTECTED_TAG_PREFIXES:
            tag_lower = tag.lower()
            # text 末尾正好是 tag 的某个前缀（且不是完整 tag —— 完整 tag 应交给 split_assistant_content 处理）
            if k < len(tag) and tag_lower.startswith(suffix_lower):
                return text[:-k]
    return text
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
    LAB4_SPECIALIST_GUIDANCE,
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
_DIRECT_ANSWER_KEYWORDS = [
    "直接告诉我", "直接给答案", "直接说答案", "给我答案", "别引导了",
    "别问我了", "别绕弯子", "你就说吧", "直接说", "直接讲", "不要引导",
]
_CONFUSION_KEYWORDS = [
    "不知道", "不会", "不懂", "看不懂", "没思路", "卡住", "不会做",
    "没看懂", "想不出来", "不会弄", "不太会",
]
_FRUSTRATION_KEYWORDS = [
    "烦", "急", "麻烦", "算了", "别绕弯", "能不能直说", "怎么这么麻烦",
    "别再问", "看不下去", "太绕了",
]
_RESOLVED_KEYWORDS = [
    "解决了", "搞定了", "好了", "通了", "成功了", "恢复了",
    "明白了", "懂了", "可以了", "没问题了",
]
_LAB_SYMPTOM_PATTERNS = [
    r"ping\s*不通",
    r"邻居.*(起不来|down|init|exstart|full)",
    r"接口.*down",
    r"(无法|不能|没法).*(连通|通信|访问|建立)",
    r"(学不到|没有).*(路由|地址)",
    r"(失败|超时|timeout|丢包|中断|异常)",
]
_LAB_OUTPUT_PATTERNS = [
    r"```",
    r"\b(show|display|ping|tracert|traceroute|ipconfig|ifconfig)\b",
    r"\b(?:ge|gigabitethernet|ethernet|serial|loopback)\s*\d+(?:/\d+){0,2}\b",
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    r"\b(up|down|administratively down|full|init|2-way|exstart)\b",
]
_LAB_TOPOLOGY_PATTERNS = [
    r"\b(?:pc|sw|s|r)\d+\b",
    r"\b(?:router|switch|vlan|ospf|area|acl|trunk)\b",
    r"\b(?:ge|gigabitethernet|ethernet|serial|loopback)\s*\d+(?:/\d+){0,2}\b",
]
_LAB_ACTION_PATTERNS = [
    r"(执行了|跑了|试了|配置了|配了|查看了|检查了|改了|抓了包)",
    r"\b(show|display|ping|tracert|traceroute|ipconfig|ifconfig|debug)\b",
]
_LAB_OUTPUT_RESULT_PATTERNS = [
    r"\b(up|down|administratively down|full|init|2-way|exstart)\b",
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    r"(超时|timeout|成功|失败|丢包|loss|reachable|unreachable)",
    r"(状态|结果|输出|回显|显示).{0,12}(是|为|如下|正常|异常)",
]
_LAB_OUTPUT_MARKER_PATTERNS = [
    r"(输出|结果|回显|显示|状态如下|命令结果)",
    r"\b(show|display|ping|tracert|traceroute|ipconfig|ifconfig|debug)\b",
]
_LAB_SLOT_KEYS = ("symptom", "output", "topology", "action")

action_re = re.compile(r'^工具：(\w+)：(.*)$')
tool_calls_block_re = re.compile(r"<tool_calls>\s*(.*?)\s*</tool_calls>", re.IGNORECASE | re.DOTALL)
_EXPERIMENT_ID_RE = re.compile(r"(?:实验\s*|lab[\s_-]?)(\d+)", re.IGNORECASE)
_MAX_TOOL_ACTIONS_PER_TURN = 5
_TOOL_API_NAME_MAP = {
    "检索": "rag_retrieve",
    "拓扑": "topology_retrieve",
    "搜索": "web_search",
}

# ── 实验4 小节检测与 query 增强 ──────────────────────────────────────────────
# 关键词按优先级排列：越具体、歧义越小的放越前面。
# "tc" 太通用，不单独匹配，改用 "netem" / "qdisc" / "tc qdisc" 等精准词。
_LAB4_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "4.6_acceptance": ["截图", "验收", "要交", "交什么", "提交", "分析与总结", "实验报告"],
    "4.3_mahimahi":   ["mahimahi", "mm-delay", "mm-loss", "mm-link", "trace文件",
                       "0.6mbps", "24mbps", "0.6m", "24m", "1500字节", "虚拟管道"],
    "4.4_mininet":    ["mininet", "sudo mn", "pingall", "xterm", "h1 ping", "h2 ping",
                       "特定链路", "节点", "nodes", "net dump"],
    "4.1_iperf":      ["iperf", "bandwidth", "jitter", "datagrams", "吞吐", "抖动"],
    "4.2_tc":         ["netem", "qdisc", "tc netem", "tc qdisc", "ens33", "loopback",
                       "发送方向", "出口方向", "入口方向", "步骤二", "步骤三", "步骤六",
                       "lo 网卡", "lo上", "lo 上", "lo加", "rtt"],
}

_LAB4_SECTION_AUGMENTATION: Dict[str, str] = {
    "4.1_iperf":    ("实验4 网络仿真工具 4.1 iperf3 TCP UDP Bandwidth Jitter "
                     "Lost/Total Datagrams 实验结果分析与总结 原问题："),
    "4.2_tc":       ("实验4 网络仿真工具 4.2 TC netem lo ens33 eth0 qdisc delay loss "
                     "发送方向 RTT 步骤二 步骤三 步骤六 原问题："),
    "4.3_mahimahi": ("实验4 网络仿真工具 4.3 Mahimahi mm-delay mm-loss mm-link trace "
                     "1500字节 12Kb 0.6Mbps 24Mbps 原问题："),
    "4.4_mininet":  ("实验4 网络仿真工具 4.4 Mininet sudo mn nodes net dump pingall "
                     "xterm tc iperf 特定链路 TCP带宽 原问题："),
    "4.6_acceptance": ("实验4 网络仿真工具 4.6 验收标准 截图要求 实验报告 "
                       "实验结果分析与总结 原问题："),
}

_LAB4_GENERAL_PREFIX = "实验4 网络仿真工具 iperf3 tc netem Mahimahi Mininet 原问题："


def _detect_lab4_section(query: str) -> Optional[str]:
    """检测实验4 query 所属小节，返回 section key；无法识别时返回 None。"""
    q_low = (query or "").lower()
    for section, keywords in _LAB4_SECTION_KEYWORDS.items():
        if any(kw.lower() in q_low for kw in keywords):
            return section
    return None


def _augment_lab4_query(query: str) -> str:
    """对实验4检索 query 做轻量 prefix 增强，保留原问题文本不变。"""
    section = _detect_lab4_section(query)
    prefix = _LAB4_SECTION_AUGMENTATION.get(section, _LAB4_GENERAL_PREFIX) if section else _LAB4_GENERAL_PREFIX
    return f"{prefix}{query}"


@dataclass(frozen=True)
class ToolActionMatch:
    tool: str
    action_input: str
    raw: str = ""
    source: str = "legacy"

    def groups(self) -> Tuple[str, str]:
        return self.tool, self.action_input


class Evidence(TypedDict):
    id: str
    query: str
    excerpt: str
    raw_text: str


class LabEvidenceSlots(TypedDict, total=False):
    symptom: List[str]
    output: List[str]
    topology: List[str]
    action: List[str]


@dataclass(frozen=True)
class HintSignals:
    llm_decision: str
    topic_shift: bool
    direct_answer_request: bool
    explicit_confusion: bool
    frustration: bool
    solved: bool
    short_reply: bool
    repeated_reply: bool
    evidence_score: int
    evidence_complete: bool
    has_new_evidence: bool
    phase: str
    stagnation_turns: int


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
    hint_state_phase: str
    hint_stagnation_turns: int
    lab_evidence_score: int
    lab_evidence_flags: Dict[str, bool]
    lab_evidence_slots: LabEvidenceSlots


# 每个线程独立持有一个 DeepSeekChatClient（requests.Session 非线程安全）
_client_local = threading.local()


def _get_client():
    if not hasattr(_client_local, "client"):
        _client_local.client = build_chat_llm(temperature=0)
    return _client_local.client


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


def _strip_citation_headers(context: str) -> str:
    """去掉 RAG context 中的 [id] source 标题行，仅保留正文。"""
    if not context:
        return ""
    cleaned_blocks: List[str] = []
    for raw_block in re.split(r"\n\s*\n", context):
        lines = raw_block.splitlines()
        if lines and re.match(r"^\[\d+\]\s+.+$", lines[0].strip()):
            block_text = "\n".join(lines[1:]).strip()
        else:
            block_text = raw_block.strip()
        if block_text:
            cleaned_blocks.append(block_text)
    return "\n\n".join(cleaned_blocks).strip()


def _build_tool_observation_for_model(observation: Any) -> str:
    """构造给模型的工具结果文本，避免暴露 citation/source 元数据。"""
    if isinstance(observation, dict):
        sanitized = dict(observation)
        sanitized.pop("citations", None)
        if isinstance(sanitized.get("context"), str):
            sanitized["context"] = _strip_citation_headers(sanitized["context"])
        return _coerce_to_text(sanitized)
    return _coerce_to_text(observation)


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


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def _matches_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _recent_user_messages(history: List[BaseMessage], limit: int = 3) -> List[str]:
    messages: List[str] = []
    for msg in reversed(history):
        if isinstance(msg, HumanMessage):
            content = _coerce_to_text(getattr(msg, "content", ""))
            if content:
                messages.append(content.strip())
            if len(messages) >= limit:
                break
    return list(reversed(messages))


def _normalize_slot_values(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _load_lab_evidence_slots(state: Dict[str, Any]) -> LabEvidenceSlots:
    raw = state.get("lab_evidence_slots", {})
    slots: LabEvidenceSlots = {}
    for key in _LAB_SLOT_KEYS:
        slots[key] = _normalize_slot_values(raw.get(key, [])) if isinstance(raw, dict) else []
    return slots


def _truncate_fragment(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _extract_lab_evidence_slots(question: str) -> LabEvidenceSlots:
    text = _coerce_to_text(question or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    slots: LabEvidenceSlots = {key: [] for key in _LAB_SLOT_KEYS}

    if _matches_any_pattern(text, _LAB_SYMPTOM_PATTERNS):
        slots["symptom"].append(_truncate_fragment(text))

    has_output_marker = _matches_any_pattern(text, _LAB_OUTPUT_MARKER_PATTERNS)
    has_output_result = _matches_any_pattern(text, _LAB_OUTPUT_RESULT_PATTERNS)
    has_code_block = "```" in text
    if has_code_block or (has_output_marker and has_output_result):
        slots["output"].append(_truncate_fragment(text, limit=180))
    elif lines:
        for line in lines:
            if _matches_any_pattern(line, _LAB_OUTPUT_MARKER_PATTERNS) and _matches_any_pattern(line, _LAB_OUTPUT_RESULT_PATTERNS):
                slots["output"].append(_truncate_fragment(line, limit=180))
                break

    if _matches_any_pattern(text, _LAB_TOPOLOGY_PATTERNS):
        slots["topology"].append(_truncate_fragment(text))

    if _matches_any_pattern(text, _LAB_ACTION_PATTERNS):
        slots["action"].append(_truncate_fragment(text))

    return slots


def _merge_lab_evidence_slots(
    existing: LabEvidenceSlots,
    incoming: LabEvidenceSlots,
) -> Tuple[LabEvidenceSlots, Dict[str, bool]]:
    merged: LabEvidenceSlots = {}
    updates: Dict[str, bool] = {}
    for key in _LAB_SLOT_KEYS:
        current_values = list(existing.get(key, []))
        new_values = []
        for value in incoming.get(key, []):
            if value not in current_values and value not in new_values:
                new_values.append(value)
        merged[key] = current_values + new_values
        updates[key] = bool(new_values)
    return merged, updates


def _score_lab_evidence(state: Dict[str, Any], question: str) -> Tuple[int, Dict[str, bool], LabEvidenceSlots, Dict[str, bool]]:
    existing_slots = _load_lab_evidence_slots(state)
    incoming_slots = _extract_lab_evidence_slots(question)
    merged_slots, slot_updates = _merge_lab_evidence_slots(existing_slots, incoming_slots)
    flags = {key: bool(merged_slots.get(key)) for key in _LAB_SLOT_KEYS}
    score = sum(1 for value in flags.values() if value)
    return score, flags, merged_slots, slot_updates


def _compute_hint_signals(
    question: str,
    history: List[BaseMessage],
    state: Dict[str, Any],
    category: str,
    prev_category: str,
    classification: Dict[str, Any],
) -> HintSignals:
    q = (question or "").strip()
    q_low = q.lower()
    topic_shift = bool(prev_category and prev_category != category)
    llm_decision = classification.get("hint_decision", "MAINTAIN")

    recent_user_msgs = _recent_user_messages(history, limit=2)
    prev_user = recent_user_msgs[-1].strip().lower() if recent_user_msgs else ""
    repeated_reply = bool(prev_user and q_low and prev_user == q_low)
    short_reply = len(q) <= 12

    direct_answer_request = _contains_any_keyword(q, _DIRECT_ANSWER_KEYWORDS)
    explicit_confusion = _contains_any_keyword(q, _CONFUSION_KEYWORDS)
    frustration = _contains_any_keyword(q, _FRUSTRATION_KEYWORDS)
    solved = _contains_any_keyword(q, _RESOLVED_KEYWORDS)

    evidence_score, evidence_flags, evidence_slots, slot_updates = _score_lab_evidence(state, q)
    previous_evidence_score = int(state.get("lab_evidence_score", 0))
    previous_phase = state.get("hint_state_phase", "probing")
    previous_stagnation = int(state.get("hint_stagnation_turns", 0))
    has_new_evidence = any(slot_updates.values())
    evidence_complete = bool(
        evidence_flags.get("symptom")
        and evidence_flags.get("output")
        and evidence_flags.get("topology")
    )

    if category != "LAB_TROUBLESHOOTING":
        phase = "resolved" if solved else "guiding"
    elif solved:
        phase = "resolved"
    elif evidence_score <= 1:
        phase = "probing"
    elif evidence_score == 2:
        phase = "gathering_evidence"
    elif evidence_complete and (explicit_confusion or llm_decision == "INCREASE"):
        phase = "proposing_fix"
    elif evidence_complete:
        phase = "narrowing_root_cause"
    else:
        phase = previous_phase

    progress_signal = solved or has_new_evidence or topic_shift or (
        category == "LAB_TROUBLESHOOTING" and previous_phase != phase
    )
    stuck_signal = (
        repeated_reply
        or explicit_confusion
        or frustration
        or (llm_decision == "INCREASE" and not progress_signal)
        or (
            category == "LAB_TROUBLESHOOTING"
            and not has_new_evidence
            and evidence_score <= previous_evidence_score
            and not topic_shift
        )
    )

    if solved or topic_shift or progress_signal:
        stagnation_turns = 0
    elif stuck_signal:
        stagnation_turns = previous_stagnation + 1
    else:
        stagnation_turns = max(0, previous_stagnation - 1)

    state["lab_evidence_score"] = evidence_score
    state["lab_evidence_flags"] = evidence_flags
    state["lab_evidence_slots"] = evidence_slots

    return HintSignals(
        llm_decision=llm_decision,
        topic_shift=topic_shift,
        direct_answer_request=direct_answer_request,
        explicit_confusion=explicit_confusion,
        frustration=frustration,
        solved=solved,
        short_reply=short_reply,
        repeated_reply=repeated_reply,
        evidence_score=evidence_score,
        evidence_complete=evidence_complete,
        has_new_evidence=has_new_evidence,
        phase=phase,
        stagnation_turns=stagnation_turns,
    )


def _apply_hint_state_machine(
    category: str,
    current_level: int,
    max_level: int,
    signals: HintSignals,
) -> Tuple[int, bool, str]:
    if current_level >= max_level:
        return max_level, False, "max_level_cap"

    if signals.direct_answer_request:
        return max_level, False, "direct_answer_request"

    if signals.solved:
        return current_level, False, "resolved"

    if category == "LAB_TROUBLESHOOTING":
        if signals.llm_decision == "JUMP_TO_MAX" and (signals.direct_answer_request or signals.frustration):
            return max_level, False, "lab_user_requested_direct_answer"
        if signals.phase == "probing" and signals.stagnation_turns >= 3:
            return min(current_level + 1, max_level), True, "stalled_without_evidence"
        if signals.phase == "gathering_evidence" and signals.stagnation_turns >= 4:
            return min(current_level + 1, max_level), True, "evidence_collection_stalled"
        if signals.phase == "narrowing_root_cause" and signals.explicit_confusion:
            return min(current_level + 1, max_level), False, "root_cause_confirmed_need_more_help"
        if signals.phase == "proposing_fix" and signals.evidence_complete:
            return max_level, False, "evidence_complete_ready_to_converge"
        if signals.llm_decision == "INCREASE" and (signals.explicit_confusion or signals.frustration):
            return min(current_level + 1, max_level), False, "llm_and_user_distress_agree"
        return current_level, False, "maintain_lab_phase"

    if signals.llm_decision == "JUMP_TO_MAX":
        return max_level, False, "llm_jump_to_max"
    if signals.llm_decision == "INCREASE" and (signals.explicit_confusion or signals.frustration or signals.stagnation_turns >= 2):
        return min(current_level + 1, max_level), False, "non_lab_increase"
    if signals.stagnation_turns >= 3 and not signals.short_reply:
        return min(current_level + 1, max_level), True, "non_lab_stagnation_failsafe"
    return current_level, False, "maintain_non_lab"


def _strip_json_code_fence(payload: str) -> str:
    text = (payload or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _dedupe_tool_actions(actions: List[ToolActionMatch]) -> List[ToolActionMatch]:
    deduped: List[ToolActionMatch] = []
    seen = set()
    for action in actions:
        key = (action.tool, action.action_input)
        if not action.tool or not action.action_input or key in seen:
            continue
        seen.add(key)
        deduped.append(action)
        if len(deduped) >= _MAX_TOOL_ACTIONS_PER_TURN:
            break
    return deduped


def _normalize_structured_tool_actions(data: Any) -> List[ToolActionMatch]:
    if isinstance(data, dict):
        data = data.get("tool_calls") or data.get("actions") or data.get("calls") or []
    if not isinstance(data, list):
        return []

    actions: List[ToolActionMatch] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("action") or item.get("name") or "").strip()
        action_input = item.get("input", item.get("query"))
        if isinstance(action_input, (dict, list)):
            action_input = json.dumps(action_input, ensure_ascii=False)
        action_input = str(action_input or "").strip()
        if tool and action_input:
            actions.append(
                ToolActionMatch(
                    tool=tool,
                    action_input=action_input,
                    raw=json.dumps(item, ensure_ascii=False),
                    source="structured",
                )
            )
    return _dedupe_tool_actions(actions)


def _find_structured_actions(text: str) -> List[ToolActionMatch]:
    for match in tool_calls_block_re.finditer(text or ""):
        payload = _strip_json_code_fence(match.group(1))
        try:
            data = json.loads(payload)
        except Exception:
            continue
        actions = _normalize_structured_tool_actions(data)
        if actions:
            return actions
    return []


def _find_legacy_actions(text: str, *, allow_incomplete_last_line: bool = True) -> List[ToolActionMatch]:
    content = text or ""
    lines = content.split("\n")
    if not allow_incomplete_last_line and content and not content.endswith("\n"):
        lines = lines[:-1]

    actions: List[ToolActionMatch] = []
    for line in lines:
        stripped = line.strip()
        match = action_re.match(stripped)
        if not match:
            continue
        tool = (match.group(1) or "").strip()
        action_input = (match.group(2) or "").strip()
        if tool and action_input:
            actions.append(
                ToolActionMatch(
                    tool=tool,
                    action_input=action_input,
                    raw=stripped,
                    source="legacy",
                )
            )
    return _dedupe_tool_actions(actions)


def _find_actions(text: str) -> List[ToolActionMatch]:
    structured = _find_structured_actions(text)
    if structured:
        return structured
    return _find_legacy_actions(text)


def _find_actions_in_stream_buffer(text: str) -> List[ToolActionMatch]:
    structured = _find_structured_actions(text)
    if structured:
        return structured
    return _find_legacy_actions(text, allow_incomplete_last_line=False)


def _has_open_tool_calls_block(text: str) -> bool:
    lowered = (text or "").lower()
    start = lowered.find("<tool_calls>")
    if start == -1:
        return False
    return lowered.find("</tool_calls>", start) == -1


def _tool_api_name(tool_name: str) -> str:
    alias = _TOOL_API_NAME_MAP.get(tool_name)
    if alias:
        return alias
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", tool_name or "").strip("_")
    return (sanitized or "tool_call")[:64]


def _build_tool_calls(action_matches: List[ToolActionMatch], turn_index: int) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for idx, action in enumerate(action_matches):
        tool_calls.append({
            "id": f"call_{turn_index}_{idx}",
            "type": "tool_call",
            "name": _tool_api_name(action.tool),
            "args": {"input": action.action_input},
        })
    return tool_calls


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

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str, *, tool_calls: Optional[List[Dict[str, Any]]] = None):
        if tool_calls:
            self.messages.append(AIMessage(content=content, tool_calls=tool_calls))
            return
        self.messages.append(AIMessage(content=content))

    def add_tool_message(self, content: str, *, tool_call_id: str, name: Optional[str] = None):
        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))

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
    allow_process_explanations: bool,
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
    state.setdefault("hint_state_phase", "probing")
    state.setdefault("hint_stagnation_turns", 0)
    state.setdefault("lab_evidence_score", 0)
    state.setdefault("lab_evidence_flags", {})
    state.setdefault("lab_evidence_slots", {})

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
    prev_category = state.get("question_category", "")
    if category == "LAB_TROUBLESHOOTING":
        if prev_category == "LAB_TROUBLESHOOTING":
            state["lab_turn_count"] += 1
        else:
            state["lab_turn_count"] = 1
    else:
        state["lab_turn_count"] = 0
    state["question_category"] = category

    # 水平个性化：新会话根据学生水平设置初始 Hint Level（分类完成后才能用 category）
    if "hint_level" not in state and user_id:
        from storage.proficiency import get_initial_hint_level
        state["hint_level"] = get_initial_hint_level(user_id, category)

    experiment_id, experiment_label = _resolve_experiment_context(q, history, state)

    # Hint Level 计算（显式信号 + 状态机）
    current_level = state.get("hint_level", 0)
    state["_hint_level_start"] = current_level
    turns_at_level = state.get("turns_at_current_level", 0)
    max_level = _MAX_HINT_LEVEL.get(category, 3)
    signals = _compute_hint_signals(q, history, state, category, prev_category, classification)
    current_hint_level, was_failsafe, transition_reason = _apply_hint_state_machine(
        category=category,
        current_level=current_level,
        max_level=max_level,
        signals=signals,
    )
    state["hint_state_phase"] = signals.phase
    state["hint_stagnation_turns"] = signals.stagnation_turns

    if current_hint_level > current_level:
        turns_at_level = 0
    elif current_hint_level >= max_level:
        turns_at_level = turns_at_level + 1
    else:
        turns_at_level = turns_at_level + 1 if signals.stagnation_turns > 0 else 0
    state["turns_at_current_level"] = turns_at_level

    print(
        "[Hint Logic] "
        f"phase={signals.phase} evidence={signals.evidence_score} "
        f"stagnation={signals.stagnation_turns} llm={signals.llm_decision} "
        f"-> level {current_hint_level} ({transition_reason})"
    )
    state["hint_level"] = current_hint_level

    # 水平采集记录
    state["_hint_decision"] = classification.get("hint_decision", "MAINTAIN")
    state["_was_failsafe"] = was_failsafe
    state["_hint_transition_reason"] = transition_reason
    state["_hint_phase"] = signals.phase
    state["_hint_evidence_score"] = signals.evidence_score
    state["_hint_stagnation_turns"] = signals.stagnation_turns

    # 组装最终 Prompt
    base_prompt_template = _get_base_prompt(category)
    strategy_instruction = _get_strategy_prompt(current_hint_level, category)
    final_prompt = base_prompt_template.format(current_strategy_instruction=strategy_instruction)

    if category == "LAB_TROUBLESHOOTING" and signals.phase == "proposing_fix" and signals.evidence_complete:
        final_prompt += (
            "\n\n【证据驱动收敛】\n"
            "当前故障现象、关键输出与拓扑上下文已经基本齐全。你应当收敛到可执行的修复路径，"
            "直接给出最关键的操作点与判断依据。可以附 1 个可选确认问题，但不能依赖学生继续补证据才能推进。"
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

    # 构建工具表（需要在 prompt 生成之前完成，prompt 会引用工具名）
    # 实验4会话：对检索 query 做小节 prefix 增强，提升实验4文档召回率
    if experiment_id == "lab4":
        def _rag_with_context(msg: str):
            augmented = _augment_lab4_query(msg)
            return RAGAgent(augmented, category=category, hint_level=current_hint_level)
    else:
        def _rag_with_context(msg: str):
            return RAGAgent(msg, category=category, hint_level=current_hint_level)

    contextual_actions = {
        "检索": _rag_with_context,
        "拓扑": lambda q: _get_topo_retriever()(q, experiment_id=state.get("experiment_id")),
    }
    if enable_websearch:
        contextual_actions["搜索"] = WebSearch

    if experiment_id and experiment_label:
        if experiment_id == "lab4":
            # 实验4无结构化拓扑数据，不提示调用拓扑工具
            final_prompt += (
                f"\n\n【实验上下文】\n"
                f"当前会话已识别为 {experiment_label}（{experiment_id}）。"
                f"实验4使用 iperf3 / tc netem / Mahimahi / Mininet 等仿真工具，"
                f"其拓扑描述为数据流文字图，无结构化拓扑 JSON，请勿调用拓扑工具。"
            )
        else:
            final_prompt += (
                f"\n\n【实验上下文】\n"
                f"当前会话已识别为 {experiment_label}（{experiment_id}）。"
                f"如果需要调用拓扑工具，只使用该实验下审核通过的拓扑 JSON，不要混用其他实验数据。"
            )

    # 实验4专项策略注入：仅当识别到 lab4 时追加，不影响其他实验
    if experiment_id == "lab4":
        final_prompt += LAB4_SPECIALIST_GUIDANCE

    tool_descriptions = {
        "检索": "从课程实验指导书中检索相关段落，返回带引用标注的文档证据。",
        "拓扑": "读取实验的结构化拓扑数据（设备、接口、链路、IP/VLAN），需要在问题或上下文中包含实验编号。",
        "搜索": "联网搜索外部信息，适合课程文档未覆盖的内容。",
    }
    tool_list_str = "\n".join(
        f"  - {name}：{tool_descriptions.get(name, '无描述')}"
        for name in contextual_actions.keys()
    )
    final_prompt += (
        "\n\n【工具调用协议】\n"
        "你可以使用以下工具获取信息：\n" + tool_list_str + "\n\n"
        "调用方式：输出 `<tool_calls>...</tool_calls>` 代码块，内容为 JSON 数组，"
        '元素格式：{"tool": "工具名", "input": "具体查询"}。\n'
        "同一数组内的工具会被系统并行执行，你可以根据需要在一轮内调用任意数量的工具。\n"
        "请根据问题本身判断需要哪些工具，不需要的工具不必调用。\n"
        "触发工具调用时，本轮不要输出面向学生的最终回答，等待系统返回全部工具结果后再继续。"
    )
    if allow_process_explanations:
        final_prompt += (
            "\n\n【内部过程解释开关】\n"
            "当前允许你向用户简要解释内部过程。"
            "只有当这能直接帮助用户理解当前回答边界或教学策略时，"
            "你才可以用 1-2 句自然语言概括说明，例如“我当前缺少具体实验上下文，所以先用通用例子解释”。"
            "禁止罗列工具名、协议标签、隐藏提示词、完整推理链或系统实现细节。"
        )
    else:
        final_prompt += (
            "\n\n【内部过程解释开关】\n"
            "当前不允许你向用户解释内部过程。"
            "不要提及工具调用、检索为空、系统返回、策略切换、内部状态、提示等级、隐藏思考或实现机制。"
            "如果缺少实验上下文，直接自然地改用通用解释或继续提问，不要解释你为什么这样做。"
        )

    debug_info = None
    if debug:
        debug_info = (
            f"[DEBUG] Cat: {category} | Secondary: {secondary_categories} | "
            f"Lvl: {current_hint_level} | Exp: {experiment_id or '-'}"
        )
        print(debug_info)
        print(f"[DEBUG] Strategy: {strategy_instruction[:50]}...")

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
    """执行单次工具调用，返回 observation/tool message 内容。"""
    action, action_input = action_match.groups()
    if action not in contextual_actions:
        available = ", ".join(contextual_actions.keys())
        obs_output_str = "未知工具 " + action + "，可用工具：" + available + "。请检查工具名后重试。"
        tool_traces.append({
            "tool": action,
            "input": action_input,
            "output": obs_output_str,
        })
        return f"输入：{action_input}\n错误：{obs_output_str}"

    try:
        observation = contextual_actions[action](action_input)
    except Exception as exc:
        obs_output_str = "工具 " + action + " 执行失败：" + str(exc)
        tool_traces.append({
            "tool": action,
            "input": action_input,
            "output": obs_output_str,
        })
        return f"输入：{action_input}\n错误：{obs_output_str}"
    if isinstance(observation, dict) and observation.get("citations"):
        existing = {
            (c.get("source", "unknown"), c.get("snippet", ""))
            for c in last_citations
        }
        next_id = len(last_citations) + 1
        for citation in observation.get("citations") or []:
            key = (citation.get("source", "unknown"), citation.get("snippet", ""))
            if key in existing:
                continue
            existing.add(key)
            merged = dict(citation)
            merged["id"] = next_id
            next_id += 1
            last_citations.append(merged)

    obs_output_str = _build_tool_observation_for_model(observation)
    tool_traces.append({
        "tool": action,
        "input": action_input,
        "output": obs_output_str[:2000],
    })
    return f"输入：{action_input}\n结果：{obs_output_str}"


def _execute_tool_actions(
    action_matches: List[ToolActionMatch],
    contextual_actions: Dict[str, Any],
    tool_traces: List[Dict[str, Any]],
    last_citations: List[Dict[str, Any]],
) -> List[str]:
    """并行执行一批工具调用，按原始顺序返回 observation/tool message 内容。"""
    if len(action_matches) <= 1:
        return [
            _execute_tool_action(m, contextual_actions, tool_traces, last_citations)
            for m in action_matches
        ]

    per_action_traces: List[List[Dict[str, Any]]] = [[] for _ in action_matches]
    per_action_citations: List[List[Dict[str, Any]]] = [[] for _ in action_matches]
    results: Dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=min(len(action_matches), 4)) as pool:
        futures = {
            pool.submit(
                _execute_tool_action, m, contextual_actions,
                per_action_traces[idx], per_action_citations[idx],
            ): idx
            for idx, m in enumerate(action_matches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    for traces_batch in per_action_traces:
        tool_traces.extend(traces_batch)
    for citations_batch in per_action_citations:
        existing = {(c.get("source", "unknown"), c.get("snippet", "")) for c in last_citations}
        for c in citations_batch:
            key = (c.get("source", "unknown"), c.get("snippet", ""))
            if key not in existing:
                existing.add(key)
                c["id"] = len(last_citations) + 1
                last_citations.append(c)

    return [results[i] for i in range(len(action_matches))]


def _find_action(text: str):
    """从 LLM 输出中提取第一个工具调用匹配。"""
    actions = _find_actions(text)
    return actions[0] if actions else None


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
    allow_process_explanations: bool = True,
) -> Tuple[str, List[BaseMessage], List[Dict[str, Any]], Dict[str, Any]]:
    if history is None:
        history = []
    if state is None:
        state = {}

    ctx = _prepare_context(
        question, history, state, user_id, enable_websearch, allow_process_explanations, debug
    )
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
    tool_traces: List[Dict[str, Any]] = []
    last_citations: List[Dict[str, Any]] = []
    final_result = ""
    bot.add_user_message(q)

    for i in range(max_turns):
        result = bot.execute()
        final_result = result

        if debug:
            print(f"Turn {i + 1}: {result[:50]}...")

        action_matches = _find_actions(result)
        if action_matches:
            if debug:
                print(f"[DEBUG] Parsed {len(action_matches)} tool actions in turn {i + 1}")
            tool_calls = _build_tool_calls(action_matches, i + 1)
            bot.add_ai_message(result, tool_calls=tool_calls)
            observations = _execute_tool_actions(
                action_matches, ctx.contextual_actions, tool_traces, last_citations
            )
            for action_match, tool_call, observation in zip(action_matches, tool_calls, observations):
                bot.add_tool_message(
                    observation,
                    tool_call_id=tool_call["id"],
                    name=action_match.tool,
                )
            continue

        bot.add_ai_message(result)
        break

    if _find_actions(final_result):
        final_result = "抱歉，我在多轮工具调用后未能生成最终回答，请尝试换一种方式提问。"

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
    allow_process_explanations: bool = True,
):
    """
    流式版 query()。yield 字典：
      {"type": "stage", "stage": "analyzing|tools|generating", "tools": [..]?}  — 阶段提示
      {"type": "thinking", "content": "..."}                                    — 流式思考增量（<思考> 标签内）
      {"type": "token", "content": "..."}                                       — 流式可见 token
      {"type": "done", "result": str, "history": List, "tool_traces": List, "state": Dict}
    说明：
      - 思考块（<思考>...</思考> 或未闭合的 <思考>...）会被流式拆成 thinking 增量发出，
        前端展示在气泡上方；token 流不再包含 <思考>/</思考>/<tool_calls> 等标签。
    """
    if history is None:
        history = []
    if state is None:
        state = {}

    # 阶段：意图分析 / 上下文准备
    yield {"type": "stage", "stage": "analyzing"}

    ctx = _prepare_context(
        question, history, state, user_id, enable_websearch, allow_process_explanations, debug
    )
    q = (question or "").strip()

    # 快速拦截（身份类）
    if ctx.early_reply is not None:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=ctx.early_reply))
        yield {"type": "stage", "stage": "generating"}
        yield {"type": "token", "content": ctx.early_reply}
        yield {"type": "done", "result": ctx.early_reply, "history": history,
               "tool_traces": [], "state": state}
        return

    # 无关问题 → 通用 LLM（流式）
    if ctx.use_general_llm:
        bot = Agent(ctx.final_prompt, history)
        bot.messages.append(HumanMessage(content=q))
        final_result = ""
        first_token = True
        for token in bot.execute_stream():
            if first_token:
                yield {"type": "stage", "stage": "generating"}
                first_token = False
            final_result += token
            yield {"type": "token", "content": token}
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=final_result))
        yield {"type": "done", "result": final_result, "history": history,
               "tool_traces": [], "state": state}
        return

    # Agent 循环（流式）
    bot = Agent(ctx.final_prompt, history)
    tool_traces: List[Dict[str, Any]] = []
    last_citations: List[Dict[str, Any]] = []
    final_result = ""
    # 跨轮聚合的"思考过程"——done 时返回前端，确保用户在生成回答后能查看完整思考记录
    aggregated_thinking_segments: List[str] = []
    bot.add_user_message(q)

    for i in range(max_turns):
        accumulated = ""
        found_actions: List[ToolActionMatch] = []
        started_forwarding = False
        last_visible = ""
        last_thinking = ""

        for token in bot.execute_stream():
            accumulated += token

            # 实时把累积文本拆成思考块与可见正文（兼容未闭合的 <思考>...）
            parsed = split_assistant_content(accumulated)
            cur_thinking = parsed.get("thinking") or ""
            cur_visible = parsed.get("visible") or ""

            # 思考块增量 yield（思考流式期间通常单调追加；末尾 strip 抖动也用 startswith 判定）
            if cur_thinking and cur_thinking != last_thinking:
                if cur_thinking.startswith(last_thinking):
                    delta = cur_thinking[len(last_thinking):]
                else:
                    # 罕见：strip 抖动导致前缀不一致；退化为整段重发首段
                    delta = cur_thinking
                if delta:
                    yield {"type": "thinking", "content": delta}
                last_thinking = cur_thinking

            # 末尾若正在形成新标签，先剪掉不完整后缀，避免 "<思考" / "<too" 这种半成品被 forward
            safe_visible = _strip_unsafe_tail(cur_visible)

            if not started_forwarding:
                found_actions = _find_actions_in_stream_buffer(accumulated)
                if found_actions:
                    break

                if _has_open_tool_calls_block(accumulated):
                    continue

                # 用"可见正文"长度判定 forward 阈值，避免 <思考> 把阈值打满
                visible_lines = safe_visible.split("\n")
                if len(visible_lines) > 2 or (len(safe_visible) > 80 and "\n" in safe_visible):
                    started_forwarding = True
                    yield {"type": "stage", "stage": "generating"}
                    if safe_visible:
                        yield {"type": "token", "content": safe_visible}
                    last_visible = safe_visible
            else:
                # forward 阶段：只推送 safe_visible 增量，绝不把 <思考> / <tool_calls> 转给前端
                if safe_visible != last_visible and safe_visible.startswith(last_visible):
                    delta_v = safe_visible[len(last_visible):]
                    if delta_v:
                        yield {"type": "token", "content": delta_v}
                    last_visible = safe_visible
                elif safe_visible != last_visible and last_visible.startswith(safe_visible):
                    # safe_visible 比 last_visible 短：可能是新的不完整标签出现导致回退
                    # 不发送回退 token（前端无法 unyield），保持 last_visible 等待下一帧
                    pass
                elif safe_visible != last_visible:
                    # 异常路径：完全非追加变化，退化整体替换不再重发
                    last_visible = safe_visible

        full_turn_text = getattr(bot, '_last_stream_result', accumulated)

        # 把这一轮的最终思考段落沉淀到聚合列表里
        if last_thinking:
            aggregated_thinking_segments.append(last_thinking)

        action_matches = found_actions or _find_actions(full_turn_text)
        if action_matches:
            if debug:
                print(f"[DEBUG] Parsed {len(action_matches)} tool actions in streaming turn {i + 1}")
            tool_calls = _build_tool_calls(action_matches, i + 1)
            bot.add_ai_message(full_turn_text, tool_calls=tool_calls)
            yield {
                "type": "stage",
                "stage": "tools",
                "tools": [m.tool for m in action_matches],
            }
            observations = _execute_tool_actions(
                action_matches, ctx.contextual_actions, tool_traces, last_citations
            )
            for action_match, tool_call, observation in zip(action_matches, tool_calls, observations):
                bot.add_tool_message(
                    observation,
                    tool_call_id=tool_call["id"],
                    name=action_match.tool,
                )
            if not started_forwarding:
                continue
            else:
                # 正常协议下，工具调用轮不应向前端暴露可见文本；若模型混出可见内容，
                # 仍以工具执行优先，交由下一轮继续推理。
                continue

        bot.add_ai_message(full_turn_text)
        final_result = full_turn_text
        break

    if _find_actions(final_result):
        final_result = "抱歉，我在多轮工具调用后未能生成最终回答，请尝试换一种方式提问。"
        yield {"type": "token", "content": final_result}

    if final_result:
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=final_result))

    # 聚合所有轮思考；与最后一轮模型自带的 <思考>...</思考> 合并去重
    aggregated_thinking = "\n\n".join(
        seg.strip() for seg in aggregated_thinking_segments if seg and seg.strip()
    ).strip()

    yield {"type": "done", "result": final_result, "history": history,
           "tool_traces": tool_traces, "state": state or {},
           "thinking_full": aggregated_thinking}


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
        elif isinstance(m, ToolMessage):
            role = "tool"
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
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=it.get("tool_call_id", "tool_call")))
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
