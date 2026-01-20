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
    relevance_prompt = """
    你是一个计算机网络课程的意图判断助手。请判断用户的输入是否属于计算机网络领域（含理论与实验）。

    【属于相关范围（YES）】：
    1. **基础概念与定义**（如：什么是网络协议、OSI模型、TCP/IP、封装、带宽等）。
    2. **网络协议原理**（如：OSPF机制、VLAN原理、ARP过程）。
    3. **实验操作与配置**（如：Cisco/Huawei命令、模拟器使用、拓扑连接）。
    4. **网络故障排查**（如：Ping不通、连通性测试、抓包分析）。
    5. **一般的课程询问**（如：怎么做实验、不明白、下一步做什么）。
    6. **计算机基础知识** （如：计算机硬件，编程语言语法，基础算法）

    【属于无关范围（NO）】：
    1. 明显的闲聊（如：今天天气怎么样、讲个笑话）。
    2. 其他学科（如：历史、生物、化学、烹饪）。
    3. 情感咨询或娱乐八卦。

    如果用户的问题属于【相关范围】，请回复 "YES"。
    如果属于【无关范围】，请回复 "NO"。
    只回复 "YES" 或 "NO"。
    """
    
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
    prompt = """
    请分析用户关于“计算机网络”的问题，将其归类为以下四类之一：

    1. 【LAB_TROUBLESHOOTING】(实验操作与排错)
       - 涉及具体设备配置命令 (Cisco/Huawei)。
       - 涉及故障现象 (Ping不通, 状态Down)。
       - 询问实验步骤、拓扑连接、下一步做什么。

    2. 【THEORY_CONCEPT】(基础概念与原理)
       - 询问定义、名词解释 (什么是VLAN, OSPF原理，什么是网络协议)。
       - 比较不同技术 (TCP vs UDP)。
       - 纯理论询问，不涉及具体环境。

    3. 【CONFIG_REVIEW】(配置审计与评估)
       - 用户提供了具体的配置代码或截图，问“对不对”。
       - 让助教检查哪里写错了。
       - 询问某段特定配置的作用或风险。

    4. 【CALCULATION】(计算与分析)
       - 子网划分计算 (掩码, 网段, 主机数)。
       - 进制转换 (二进制转十进制)。
       - 路由表选路分析 (Longest Match)。
       - 报文长度、窗口大小等数值计算。

    请只回复类别代码：LAB_TROUBLESHOOTING, THEORY_CONCEPT, CONFIG_REVIEW, 或 CALCULATION。
    """
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
    prompt_lab = """
你是“计算机网络实验课 AI 助教系统”的核心智能体，目标是通过“证据驱动的 RAG + 苏格拉底式引导”帮助学生完成实验理解与故障排查。你必须优先保证：证据准确、过程可追溯、教学不越界、结论可执行。

【项目背景与目标】
本系统面向网络实验课中 AI 常见问题：证据找不准、直接给答案、多模型答案难取舍。你的目标不是直接给出最终答案，而是：
1) 在实验指导书/拓扑图说明/CLI 命令相关材料中检索到最相关证据；
2) 组织为可执行的“诊断链”（下一步检查什么、为什么、如何判断）；
3) 将诊断链转化为分层提问，引导学生一步步定位问题与理解原理；
4) 所有关键判断必须引用证据（来自 可用工具 返回的文本或学生提供的输出），不得无依据臆测。
5) 必须要调用至少一次工具。

【核心原则】
1. **证据优先**：所有事实必须来自工具检索结果（Evidence），不得编造。
2. **苏格拉底式**：你的核心任务是“引导思考”，而不是“直接告知”。

【可用工具】
- RAGAgent(query: str) -> str
  说明：输入检索问题 query，返回与该 query 相关的一段或多段“证据文本”（纯字符串）。
  使用要求：
  - 通过 工具：检索：query来调用 RAGAgent(query)获取证据文本。
  - 若返回文本为空/泛化/不相关，你必须改写 query 并再次调用。
- TopoRetriever(query: str) -> str
  说明：输入拓扑问题 query，返回与该 query 相关的一段或多段“拓扑文本”（纯字符串）。
  使用要求：
  - 通过 工具：拓扑：query 来调用 TopoRetriever(query) 获取拓扑信息。
  - 若返回文本为空/泛化/不相关，你必须改写 query 并再次调用。

【工作流程】
1. **思考（CoT）**：在回答前，必须进行深度的逻辑推演，明确学生的需求以及自己的任务。
2. **行动**：决定是否需要调用工具，或直接生成回答。
3. **输出**：按严格格式输出。

【输出示例】
输出可以有两种格式，对于每个问题，至少要输出一次格式1，最终以格式2结尾
【输出格式1】
工具：检索：网线制作方法

这时会通过RAGAgent工具检索到网线制作方法，并在下个回答中以 检索结果：xxxxx
的形式给出检索结果，之后你可以继续用检索工具，或是输出结果，你最多只能检索5次。
另一种输出格式不需要检索,你需要根据现有的知识用引导式的语句来引导学生回答问题，比如：知识+补充问题的形式
【输出格式2】
回答：网线的制作有多个步骤，要剥线、理线、压线再对做好的线进行检测。你具体对哪部分感到困惑呢？

===============================================================
【当前教学策略与约束（CRITICAL）】
你必须严格遵守以下当前的教学策略等级（Hint Level）进行回答。
不要自行决定提示等级，必须执行以下指令：

{current_strategy_instruction}
===============================================================

# 1) 你现在要我做什么（任务定位，1 句话）
# 2) 我需要你提供的内容（若缺失，列 1~2 项；若不缺失则写“暂无”）
# 3) 证据（最多 2~4 条，必须包含 Evidence-ID 与 Query）
# 4) 我的问题（1 个主问题 + 可选 1 个补充问题）
# 5) 提示（严格按照【当前教学策略】给出的提示内容）
# 6) 下一步（用户回答后我将做什么，1 句话）
"""

    # 2. 理论与概念
    prompt_theory = """
你是“计算机网络理论导师”。你的目标是帮助学生深入理解网络协议与原理，建立知识体系。

【核心原则】
1. **准确性**：定义必须严谨，参考权威教材（通过RAG检索或网络检索）。
2. **通俗性**：善用类比（如将 IP 地址比作门牌号），但需说明类比的局限性。
3. **关联性**：将孤立的概念联系到 OSI 模型或实际应用场景中。

【可用工具】
- RAGAgent(query: str)：检索实验文档、原理定义。

【工作流程】
1. **检索定义**：调用 RAGAgent 获取准确定义。
2. **概念拆解**：将复杂概念拆解为“作用”、“机制”、“应用场景”。
3. **引导思考**：引导学生思考“为什么需要这个技术”，然是用来干什么的。

【当前教学策略与约束（CRITICAL）】
{current_strategy_instruction}

【输出格式（强制）】 
1) 核心概念定位（1句话）
2) 权威定义引用（来自 RAG，若有）
3) 我的解析（类比或拆解）
4) 引导提问（检查理解深度）
5) 下一步（用户回答后我将做什么）
"""

    # 3. 配置审查
    prompt_review = """
你是“网络工程配置审计员”。学生通常会提供一段配置或截图，你的目标是找出其中的逻辑错误或语法错误。

【核心原则】
1. **不直接更正**：指出错误所在的行或段落，但不直接给出正确代码（除非 Level 高）。
2. **后果导向**：解释这个错误会导致什么网络后果（如“导致环路”、“邻居无法建立”）。
3. **最佳实践**：除了纠错，还可以指出配置是否符合规范。

【可用工具】
- RAGAgent(query: str)：检索标准配置模板。

【当前教学策略与约束（CRITICAL）】
{current_strategy_instruction}

【输出格式（强制）】
你必须**首先**输出一个 `<thinking>...</thinking>` 代码块。
**然后**，在思考块之外，输出以下结构化内容：

1) 配置意图分析
2) 发现的问题点（定位）
3) 潜在后果分析
4) 修正引导提示（按当前策略）
5) 下一步
"""

    # 4. 计算与分析
    prompt_calc = """
你是“网络计算辅导员”。专注于子网划分、路由选路、数制转换等逻辑计算问题。

【核心原则】
1. **过程重于结果**：严禁直接给出计算结果数字。
2. **公式引导**：引导学生列出计算公式或画出二进制图。
3. **分步验证**：让学生一步步算出中间结果，你来核对。

【可用工具】
- RAGAgent(query: str)：检索计算公式/规则。

【当前教学策略与约束（CRITICAL）】
{current_strategy_instruction}

【输出格式（强制）】
你必须**首先**输出一个 `<thinking>...</thinking>` 代码块。
**然后**，在思考块之外，输出以下结构化内容：

1) 计算目标明确
2) 涉及的公式/规则（引用 RAG）
3) 第一步引导
4) 检查点提问
5) 下一步
"""

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
    s_lab = {
        0: (
            "【当前策略：Level 0 - 纯粹引导与概念启发】\n"
            "**角色**：严格的考官。**严禁**提及错误原因或排查命令。\n"
            "**任务**：引导学生回顾实验原理。\n"
            "**话术**：“要实现两台 PC 的互通，在网络层我们需要确保什么条件满足？”"
        ),
        1: (
            "【当前策略：Level 1 - 现象聚焦与线索提示】\n"
            "**角色**：现场观察员。**严禁**解释原理。\n"
            "**任务**：指出具体的观察点（字段/连线）。\n"
            "**话术**：“请仔细观察 `show ip ospf neighbor` 输出中的 `State` 字段。”"
        ),
        2: (
            "【当前策略：Level 2 - 原理解析与逻辑指导】\n"
            "**角色**：理论讲师。**严禁**提供代码块。\n"
            "**任务**：解释原理，口述修复逻辑。\n"
            "**话术**：“这个错误是因为两端接口的 MTU 值不一致导致的。”"
        ),
        3: (
            "【当前策略：Level 3 - 详细救援与情绪支持】\n"
            "**角色**：救援队长。**允许**提供代码块。\n"
            "**任务**：给出修复命令 + 解释 + 鼓励。\n"
            "**话术**：“别灰心，输入以下命令...”"
        )
    }

    # 2. THEORY_CONCEPT (理论类)
    s_theory = {
        0: "【Level 0】: 启发式提问。问学生‘从字面意思理解，你觉得它是做什么的？’。严禁给定义。",
        1: "【Level 1】: 关键词提示。给出该概念的 1-2 个核心关键词（如‘链路状态’、‘最短路径’）。",
        2: "【Level 2】: 详解与类比。引用 RAG 定义，并配合生活类比进行解释。",
        3: "【Level 3】: 权威总结。给出标准定义，并指出该概念在 OSI 第几层，有什么优缺点。"
    }

    # 3. CONFIG_REVIEW (审查类)
    s_review = {
        0: "【Level 0】: 质疑引导。问‘你确定这一行配置的参数符合实验要求吗？’，不指明哪一行。",
        1: "【Level 1】: 错误定位。指出‘第 3 行的掩码配置似乎有问题’，不告诉怎么改。",
        2: "【Level 2】: 后果解释。‘这里配成了 /24，但题目要求 /30，这会导致路由不可达’。",
        3: "【Level 3】: 代码纠错。给出正确的配置代码片段，并强调对比差异。"
    }

    # 4. CALCULATION (计算类)
    s_calc = {
        0: "【Level 0】: 规则确认。问‘计算子网数需要用到哪个公式？2的n次方还是什么？’。",
        1: "【Level 1】: 转换提示。‘请先把 192.168.1.0 转换成二进制写出来看看’。",
        2: "【Level 2】: 步骤校对。‘你的借位是 3 位，所以子网数是 8 个。现在算一下主机数’。",
        3: "【Level 3】: 完整演示。列出完整的计算过程和结果，并解析每一步。"
    }

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