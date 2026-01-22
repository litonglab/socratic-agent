from email import message
import json
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agentic_rag.rag import RAGAgent
from agentic_rag.topo_rag import TopoRetriever
from dataclasses import dataclass, field
from agentic_rag.utils import extract_excerpt, _coerce_to_text
from agentic_rag.socratic.ping_controller import handle_ping_socratic
from agentic_rag.llm_config import build_chat_llm
from .prompts import (
    RELEVANCE_PROMPT,
    CATEGORY_DETECT_PROMPT,
    BASE_PROMPT_LAB,
    BASE_PROMPT_THEORY,
    BASE_PROMPT_REVIEW,
    BASE_PROMPT_CALC,
    STRATEGY_LAB,
    STRATEGY_THEORY,
    STRATEGY_REVIEW,
    STRATEGY_CALC
)

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
    question_category: str # [新增] 记录问题分类
    mode: str  # "socratic" | "direct"

KEYWORDS = [
    "步骤", "检查", "预期", "原因", "常见", "故障", "排查",
    "OSPF", "BGP", "VLAN", "NAT", "ACL", "ARP", "STP",
    "show", "display", "ping", "traceroute", "邻居", "路由表"
]

client = build_chat_llm(temperature=0)

# -------------------------------------------------------------------------
# [原功能] 相关性检查函数 (保持不变)
# -------------------------------------------------------------------------
def check_relevance(question: str) -> bool:
    """
    判断用户问题是否与计算机网络课程相关（含理论与实验）。
    返回: True (相关) | False (无关)
    """
    q_str = (question or "").strip()
    if not q_str:
        return True # 空消息放行

    # 快速放行常见交互词
    greetings = ["你好", "hello", "hi", "谢谢", "再见", "help", "救命", "提示", "是什么", "what is"]
    if len(q_str) < 15 and any(g in q_str.lower() for g in greetings):
        return True

    # 使用LLM进行意图分类
    relevance_prompt = RELEVANCE_PROMPT
    
    messages = [
        SystemMessage(content=relevance_prompt),
        HumanMessage(content=f"用户输入: {q_str}")
    ]
    
    try:
        response = client.invoke(messages)
        result = response.content.strip().upper()
        return "NO" not in result
    except Exception as e:
        print(f"[Warning] Relevance check failed: {e}, defaulting to relevant.")
        return True

# -------------------------------------------------------------------------
# [新增] 问题分类器 (4类)
# -------------------------------------------------------------------------
def detect_question_category(question: str) -> str:
    """
    将问题归类为四大场景之一。
    """
    prompt = CATEGORY_DETECT_PROMPT
    messages = [SystemMessage(content=prompt), HumanMessage(content=f"用户输入: {question}")]
    try:
        response = client.invoke(messages)
        content = response.content.strip().upper()
        if "THEORY" in content: return "THEORY_CONCEPT"
        if "REVIEW" in content: return "CONFIG_REVIEW"
        if "CALC" in content: return "CALCULATION"
        return "LAB_TROUBLESHOOTING" # 默认兜底
    except Exception:
        return "LAB_TROUBLESHOOTING"

# -------------------------------------------------------------------------
# [新增] 动态 Base Prompt 生成器
# -------------------------------------------------------------------------
def get_base_prompt(category: str) -> str:
    """
    根据问题分类，返回完全不同的 System Prompt 框架。
    """
    
    # 1. 实验与排错 (保留你原有的 BASE_PROMPT 内容)
    prompt_lab = BASE_PROMPT_LAB

    # 2. 理论与概念 (优化版 - 对齐 Lab 格式与工具调用)
    prompt_theory = BASE_PROMPT_THEORY
    
    # 3. 配置审查 (优化版 - 引入拓扑校验与逻辑审计)
    prompt_review = BASE_PROMPT_REVIEW
    
    # 4. 计算与分析 (优化版 - 引入场景化计算与分步验证)
    prompt_calc = BASE_PROMPT_CALC

    mapping = {
        "LAB_TROUBLESHOOTING": prompt_lab,
        "THEORY_CONCEPT": prompt_theory,
        "CONFIG_REVIEW": prompt_review,
        "CALCULATION": prompt_calc
    }
    
    return mapping.get(category, prompt_lab)

# -------------------------------------------------------------------------
# [原功能] Hint Level 管理逻辑
# -------------------------------------------------------------------------
def determine_hint_level(state: Dict[str, Any], user_question: str) -> int:
    current_turn = state.get("user_turn_count", 0) + 1
    state["user_turn_count"] = current_turn
    turn_based_floor = (current_turn - 1) // 3
    previous_level = state.get("hint_level", 0)
    help_keywords = [
        "不懂", "不会", "不知道", "提示", "又错了", "还是不对", 
        "怎么办", "怎么做", "为什么", "结果是啥", "给个答案",
        "太难", "迷糊", "仔细", "解释", "看不懂", "好难"
    ]
    q_lower = user_question.lower()
    keyword_triggered_level = previous_level
    if any(k in q_lower for k in help_keywords):
        keyword_triggered_level = previous_level + 1
    final_level = max(turn_based_floor, keyword_triggered_level)
    final_level = min(final_level, 3)
    return final_level

# -------------------------------------------------------------------------
# [升级版] Hint Strategy 生成器 (支持分类)
# -------------------------------------------------------------------------
def get_strategy_prompt(level: int, category: str) -> str:
    """
    根据 Level 和 Category 返回具体的指导策略。
    """
    # 1. LAB_TROUBLESHOOTING (保留你原有的策略)
    s_lab = STRATEGY_LAB

    # 2. THEORY_CONCEPT (理论类策略 - 从定义到应用)
    s_theory = STRATEGY_THEORY

    # 3. CONFIG_REVIEW (配置审查类策略 - 从审计到修正)
    s_review = STRATEGY_REVIEW

    # 4. CALCULATION (计算类策略 - 从公式到结果)
    s_calc = STRATEGY_CALC
    
    strategies = {
        "LAB_TROUBLESHOOTING": s_lab,
        "THEORY_CONCEPT": s_theory,
        "CONFIG_REVIEW": s_review,
        "CALCULATION": s_calc
    }
    
    target_dict = strategies.get(category, s_lab)
    return target_dict.get(level, target_dict[0])

# -------------------------------------------------------------------------
# [原功能] 其他辅助函数
# -------------------------------------------------------------------------
def call_rag_and_record(state: AgentState, query: str) -> str:
    raw = RAGAgent(query)
    raw_text = _coerce_to_text(raw)
    evidences = state.get("evidences", [])
    eid = f"E{len(evidences) + 1}"
    evidences.append({
        "id": eid,
        "query": query,
        "excerpt": extract_excerpt(raw),
        "raw_text": raw_text
    })
    state["evidences"] = evidences
    return raw_text

def retrieve_evidence_node(state: AgentState) -> AgentState:
    user_msg = state["user_message"]
    queries = [
        f"{user_msg} 相关定义与原理",
        f"{user_msg} 实验步骤与检查点",
        f"{user_msg} 常见错误原因与排查",
    ]
    for q in queries:
        _ = call_rag_and_record(state, q)
    return state

class Agent():
    def __init__(self, prompt,history):
        self.prompt = prompt
        self.history = []
        self.history = history
        self.messages = [] 
        if self.history != []:
            for h in self.history:
                self.messages.append(h)
        if  self.prompt: 
            self.messages.append(SystemMessage(content=prompt))
    def __call__(self, message,history) :
        self.messages.append(HumanMessage(content=message))
        # history.append(HumanMessage(content=message)) 
        result = self.execute() 
        self.messages.append(AIMessage(content=result)) 
        # history.append(AIMessage(content=result)) 
        return result,history

    def execute(self):
        response = client.invoke(self.messages)
        return response.content

known_actions={
    "检索": RAGAgent,
    "拓扑":TopoRetriever,
}

action_re = re.compile(r'^工具：(\w+)：(.*)$')

# -------------------------------------------------------------------------
# [修改] 主流程 Query (整合分类逻辑)
# -------------------------------------------------------------------------
def query(
    question: str,
    history: Optional[List[BaseMessage]] = None,
    max_turns: int = 5,
    debug: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[BaseMessage], List[Dict[str, Any]], Dict[str, Any]]:
    
    if history is None:
        history = []
    if state is None:
        state = {}

    # 1. 相关性检查
    is_relevant = check_relevance(question)
    if not is_relevant:
        reply = "与本课程无关，不予回答。"
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply))
        if debug:
            print(f"[Guardrail] Blocked irrelevant query: '{question}'")
        return reply, history, [], state

    # 2. Ping 场景特殊处理
    q = (question or "").strip()
    q_low = q.lower()
    ping_trigger = (
        ("ping" in q_low)
        or ("不通" in q)
        or ("连不通" in q)
        or ("超时" in q)
        or ("不可达" in q)
        or ("unreachable" in q_low)
        or ("timed out" in q_low)
    )
    if ping_trigger:
        reply, history, tool_traces, new_state = handle_ping_socratic(
            user_message=q,
            history=history,
            state_dict=state,
        )
        return reply, history, tool_traces, (new_state or {})

    # -----------------------------
    # 3. 计算 Hint Level & 判断问题分类
    # -----------------------------
    # (A) 计算 Level
    current_hint_level = determine_hint_level(state, q)
    state["hint_level"] = current_hint_level
    
    # (B) 识别问题分类 (LAB / THEORY / REVIEW / CALC)
    category = detect_question_category(q)
    state["question_category"] = category
    
    # (C) 获取 System Prompt 模板
    base_prompt_template = get_base_prompt(category)
    
    # (D) 获取 Strategy Instruction
    strategy_instruction = get_strategy_prompt(current_hint_level, category)
    
    # (E) 组装最终 Prompt
    final_prompt = base_prompt_template.format(current_strategy_instruction=strategy_instruction)

    if debug:
        print(f"[DEBUG] Cat: {category} | Lvl: {current_hint_level}")
        print(f"[DEBUG] Strategy: {strategy_instruction[:50]}...")

    # -----------------------------
    # 4. Agent 执行循环
    # -----------------------------
    i = 0
    bot = Agent(final_prompt, history)
    next_prompt = q
    tool_traces: List[Dict[str, Any]] = []
    last_citations: List[Dict[str, Any]] = []

    def _format_citations(citations: List[Dict[str, Any]]) -> str:
        if not citations:
            return ""
        lines = [f"[{c.get('id')}] {c.get('source', 'unknown')}" for c in citations]
        return "引用：\n" + "\n".join(lines)

    while i < max_turns:
        i += 1
        result, _ = bot(next_prompt, history)

        if debug:
            print(f"Turn {i}: {result[:50]}...")

        actions = [
            action_re.match(a)
            for a in (result or "").split("\n")
            if action_re.match(a)
        ]

        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")

            observation = known_actions[action](action_input)
            if isinstance(observation, dict) and observation.get("citations"):
                last_citations = observation.get("citations") or []

            obs_output_str = _coerce_to_text(observation)

            tool_traces.append({
                "tool": action,
                "input": action_input,
                "output": obs_output_str[:2000],
            })

            next_prompt = f"检索结果：{obs_output_str}"
            continue

        if last_citations and "引用：" not in (result or ""):
            result = (result or "").rstrip() + "\n\n" + _format_citations(last_citations)
        
        return result, history, tool_traces, (state or {})

    if last_citations and "引用：" not in (result or ""):
        result = (result or "").rstrip() + "\n\n" + _format_citations(last_citations)
    return result, history, tool_traces, (state or {})


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
    # 测试相关性守卫
    print("--- Testing Relevance Guardrail ---")
    q1 = "什么是网络协议？"
    output1, _, _, _ = query(q1, debug=True)
    print(f"\n[Q: {q1}] => {output1[:50]}...") 
    
    q2 = "宫保鸡丁怎么做？"
    output2, _, _, _ = query(q2, debug=True)
    print(f"\n[Q: {q2}] => {output2}")
    
    # 测试分类器效果
    print("\n--- Testing Classifier & Prompt Switching ---")
    q3 = "怎么划分子网？" # 应该触发 CALCULATION 或 LAB
    output3, _, _, state3 = query(q3, debug=True)
    print(f"\n[Q: {q3}] Cat: {state3.get('question_category')}")