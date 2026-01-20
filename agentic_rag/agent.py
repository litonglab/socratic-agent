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
    user_turn_count: int  # [新增] 记录用户对话轮数
    mode: str  # "socratic" | "direct"

KEYWORDS = [
    "步骤", "检查", "预期", "原因", "常见", "故障", "排查",
    "OSPF", "BGP", "VLAN", "NAT", "ACL", "ARP", "STP",
    "show", "display", "ping", "traceroute", "邻居", "路由表"
]

# -------------------------------------------------------------------------
# [新增] 相关性检查函数
# -------------------------------------------------------------------------
def check_relevance(question: str) -> bool:
    """
    判断用户问题是否与计算机网络课程高度相关。
    返回: True (相关) | False (无关)
    """
    q_str = (question or "").strip()
    if not q_str:
        return True # 空消息放行

    # 快速放行常见交互词，节省Token
    greetings = ["你好", "hello", "hi", "谢谢", "再见", "help", "救命", "提示"]
    if len(q_str) < 10 and any(g in q_str.lower() for g in greetings):
        return True

    # 使用LLM进行意图分类
    relevance_prompt = """
    你是一个严格的分类器。你的任务是判断用户输入是否与“计算机网络实验课程”强相关。

    强相关的主题包括：
    - 计算机相关知识
    - 网络协议（TCP/IP, OSPF, BGP, VLAN, STP, ARP等）
    - IP地址、子网划分、路由
    - 网络设备配置（路由器、交换机命令）
    - 网络故障排查（ping, traceroute, 抓包分析）
    - 课程实验内容、拓扑图
    - 对助教回复的追问（为什么、怎么做、不明白）

    如果输入明确属于上述范围，请只回答 "YES"。
    如果输入明显是其他领域（例如：历史、文学、菜谱、通用聊天），请只回答 "NO"。
    只回答 "YES" 或 "NO"。
    """

# -------------------------------------------------------------------------
# [新增] Hint Level 管理逻辑
# -------------------------------------------------------------------------
def determine_hint_level(state: Dict[str, Any], user_question: str) -> int:
    """
    根据用户输入关键词 和 对话轮数 共同决定 Hint Level。
    逻辑：
    1. 轮数逻辑：每 3 轮自动提升一级保底 Level (0->1->2->3)。
    2. 关键词逻辑：用户喊不懂/求助时，在当前基础上 +1。
    3. 取两者最大值。
    """
    # 1. 获取并更新轮数
    # 如果是新会话，默认为 0，加 1 后变为第 1 轮
    current_turn = state.get("user_turn_count", 0) + 1
    state["user_turn_count"] = current_turn
    
    # 2. 计算【轮数保底等级】
    # 第 1-3 轮 -> 0
    # 第 4-6 轮 -> 1
    # 第 7-9 轮 -> 2
    # 第 10+ 轮 -> 3
    turn_based_floor = (current_turn - 1) // 3

    # 3. 计算【关键词触发等级】
    # 获取上一次的状态等级，如果不存在则从 0 开始
    previous_level = state.get("hint_level", 0)
    
    help_keywords = [
        "不懂", "不会", "不知道", "提示", "又错了", "还是不对", 
        "怎么办", "怎么做", "为什么", "结果是啥", "给个答案",
        "太难", "迷糊", "仔细", "解释", "看不懂", "好难"
    ]
    
    q_lower = user_question.lower()
    keyword_triggered_level = previous_level
    
    # 如果用户明确表示困惑，尝试升级
    if any(k in q_lower for k in help_keywords):
        keyword_triggered_level = previous_level + 1

    # 4. 最终决策：取最大值，并限制在 0~3 之间
    final_level = max(turn_based_floor, keyword_triggered_level)
    final_level = min(final_level, 3) # 上限锁定为 3
    
    return final_level

def get_strategy_prompt(level: int) -> str:
    """
    根据 Level 返回给 LLM 的强制策略指令。
    这些指令将注入到 System Prompt 的 {current_strategy_instruction} 占位符中。
    """
    strategies = {
        0: (
            "【当前策略：Level 0 - 纯粹引导与概念启发】\n"
            "--------------------------------------------------\n"
            "**你的角色**：你是一位严格的“苏格拉底式导师”。你的目标是**暴露学生的知识盲区**，而不是填补它。\n"
            "**核心原则**：授人以鱼不如授人以渔。学生目前处于探索初期，必须让他们自己去“撞墙”并发现墙在哪里。\n"
            "\n"
            "**【严禁事项（违反将视为教学事故）】**：\n"
            "1. ❌ 严禁提及任何具体的错误原因（如“MTU不匹配”、“缺少路由”）。\n"
            "2. ❌ 严禁提及具体的排查命令（如“用 show ip interface brief 看看”）。\n"
            "3. ❌ 严禁写出完整的配置命令。\n"
            "\n"
            "**【必须执行的思维步骤】**：\n"
            "1. **知识检索**：根据 RAG 证据，确定学生当前问题背后的核心概念是什么（例如：OSPF 邻居建立的条件、VLAN 的标签规则）。\n"
            "2. **盲区定位**：判断学生是“不知道这个概念”还是“忘了做某一步检查”。\n"
            "3. **反向发问**：设计一个问题，引导学生自己去翻阅实验指导书或回忆理论课内容。\n"
            "\n"
            "**【输出话术范例】**：\n"
            "- “要实现两台 PC 的互通，在网络层我们需要确保什么条件满足？请回顾一下实验指导书第 2 章关于‘网关’的定义。”\n"
            "- “你现在的现象是 Ping 不通。在判断是线路问题还是配置问题之前，通常我们第一步会检查什么物理状态？”\n"
            "- “根据证据 E1，OSPF 建立邻居有几个必要参数。你确认过这些参数在你的设备上都配置了吗？”"
            "- “根据证据 E1，OSPF 建立邻居有几个必要参数。你确认过这些参数在你的设备上都配置了吗？”"
            "- “实验13需要你掌握掌握子网划分的方法与原理；能够根据各部门PC数量划分子网，合理分配IP地址；配置交换机VLAN、Trunk口、三层交换机接口；配置静态路由，实现跨网段通信；理解路由聚合的作用，简化路由表配置。你有好好理解相关知识吗？”"
        ),
        
        1: (
            "【当前策略：Level 1 - 现象聚焦与线索提示】\n"
            "--------------------------------------------------\n"
            "**你的角色**：你是一位敏锐的“现场观察员”。学生已经有了基础概念，但迷失在海量的 CLI 输出或拓扑细节中，找不着北。\n"
            "**核心原则**：缩小搜查范围。把学生的手指引向错误发生的地方，但**闭口不谈**具体的错误原因。\n"
            "\n"
            "**【严禁事项】**：\n"
            "1. ❌ 严禁解释“为什么错”（不要说原理）。\n"
            "2. ❌ 严禁给出修改建议（不要说“你应该把这个改大一点”）。\n"
            "3. ❌ 严禁写出完整的配置命令。\n"
            "\n"
            "**【必须执行的思维步骤】**：\n"
            "1. **证据对比**：在内心对比 RAG 提供的“正常状态证据”和学生提供的“当前状态”。\n"
            "2. **锁定差异**：找到差异所在的具体字段（如 `Status`、`Protocol`、`Mask`）或拓扑位置（如 `G0/0/1` 接口）。\n"
            "3. **定向引导**：告诉学生“去盯着这个地方看”。\n"
            "\n"
            "**【输出话术范例】**：\n"
            "- “请仔细观察 `show ip ospf neighbor` 输出中的 `State` 字段。根据证据 E2，正常的建立状态应该是 Full，但你那里显示的是什么？”\n"
            "- “注意看拓扑图上路由器 R1 和 R2 之间的连线。你配置 IP 地址的接口号，和物理连线的接口号是一致的吗？”\n"
            "- “你的 Ping 虽然通了，但是延迟很大。请检查一下接口的双工模式（Duplex）字段。”"
        ),
        
        2: (
            "【当前策略：Level 2 - 原理解析与逻辑指导】\n"
            "--------------------------------------------------\n"
            "**你的角色**：你是一位资深的“理论讲师”。学生已经看到了错误现象，但不懂背后的机制，不知道如何下手修改。\n"
            "**核心原则**：讲透原理，给出“自然语言”的修复逻辑，但**绝不代写代码**。\n"
            "\n"
            "**【严禁事项】**：\n"
            "1. ❌ 严禁提供可直接复制粘贴的代码块（Code Block）。\n"
            "2. ❌ 严禁直接给出具体的参数值（如“你把掩码改成 255.255.255.0”），要说“修改掩码使其匹配”。\n"
            "\n"
            "**【必须执行的思维步骤】**：\n"
            "1. **归因分析**：结合 RAG 证据，明确解释导致该现象的技术原因（Mechanism）。\n"
            "2. **逻辑构建**：将修复步骤拆解为“自然语言流程”（Step-by-Step Logic）。\n"
            "3. **引用背书**：必须引用 E1/E2 等证据来证明你的原理解释是权威的。\n"
            "\n"
            "**【输出话术范例】**：\n"
            "- “这个错误是因为两端接口的 MTU 值不一致导致的（见证据 E1）。OSPF 在 ExStart 阶段会协商 MTU，如果不匹配就会卡住。”\n"
            "- “解决这个问题的逻辑是：你需要进入两端的接口视图，分别检查当前的 MTU 设置，并将它们修改为相同的值（通常是 1500）。”\n"
            "- “这是因为你配置了 ACL 拒绝了 ICMP 流量。你需要找到应用在入方向的那个 ACL，添加一条允许 ICMP 的规则，或者在接口上取消应用。”"
        ),
        
        3: (
            "【当前策略：Level 3 - 详细救援与情绪支持】\n"
            "--------------------------------------------------\n"
            "**你的角色**：你是一位负责兜底的“救援队长”。学生已经尝试多次失败，情绪可能焦躁。现在必须止损，确保实验能继续进行。\n"
            "**核心原则**：给出答案（代码/命令），但必须“买一送一”（解释 + 鼓励 + 建议）。\n"
            "\n"
            "**【允许事项】**：\n"
            "✅ 允许（且必须）提供正确的、具体的 CLI 配置命令。\n"
            "✅ 允许直接指出哪里配错了。\n"
            "\n"
            "**【必须执行的思维步骤】**：\n"
            "1. **情绪安抚**：先肯定学生的努力，缓解挫败感。\n"
            "2. **直接方案**：给出精准的修复步骤或命令序列。\n"
            "3. **知识回扣**：解释为什么这条命令能解决问题（防止死记硬背）。\n"
            "4. **后续建议**：建议学生课后复习哪个知识点。\n"
            "\n"
            "**【输出话术范例】**：\n"
            "- “别灰心，OSPF 的认证配置确实很容易搞混。我们先解决问题：\n"
            "  请在 R1 上执行：\n"
            "  ```\n"
            "  interface g0/0\n"
            "  ip ospf authentication message-digest\n"
            "  ```\n"
            "  这样做是因为（证据 E3）你的区域开启了 MD5 认证，接口下必须显式启用。课后建议复习一下‘接口认证’与‘区域认证’的区别。”\n"
            "- “看起来这里卡了很久了，我们直接通过吧。问题出在你的静态路由下一跳写错了。请输入 `ip route 192.168.2.0 255.255.255.0 10.1.1.2`。注意下一跳必须是直连链路的对端 IP。”"
        )
    }
    return strategies.get(level, strategies[0])
# -------------------------------------------------------------------------

def call_rag_and_record(state: AgentState, query: str) -> str:
    raw = RAGAgent(query)  # 保持签名不变
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

# client=ChatOpenAI(model="gpt-4o-mini", temperature=0)
client = build_chat_llm(temperature=0)

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
        # history.append(HumanMessage(content=message)) # 注：query loop中维护history，此处仅处理当前流
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

# [修改] 改名为 BASE_PROMPT，并添加占位符 {current_strategy_instruction}
BASE_PROMPT="""
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

action_re = re.compile(r'^工具：(\w+)：(.*)$')

def query(
    question: str,
    history: Optional[List[BaseMessage]] = None,
    max_turns: int = 5,
    debug: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[BaseMessage], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Return: (reply_text, new_history_messages, tool_traces, new_state)

    - tool_traces: for frontend debug panel
    - new_state: per-session socratic state (saved by server.py)
    """
    if history is None:
        history = []
    if state is None:
        state = {}

    # ---------------------------------------------------------------------
    # [新增] 1. 课程相关性检查 (Guardrail)
    # ---------------------------------------------------------------------
    is_relevant = check_relevance(question)
    if not is_relevant:
        reply = "与本课程无关，不予回答。"
        # 保持对话历史的完整性
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply))
        if debug:
            print(f"[Guardrail] Blocked irrelevant query: '{question}'")
        # 直接返回，中断后续所有逻辑
        return reply, history, [], state

    # ---------------------------------------------------------------------
    # [原功能] 2. Intent routing: ping scenario -> socratic controller (无修改)
    # ---------------------------------------------------------------------
    q = (question or "").strip()
    q_low = q.lower()

    # 一个实用的 ping 触发条件：包含 ping 或 “不通/连不通/超时/不可达”等
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
    # [NEW LOGIC] Determine Hint Level & Prompt Strategy
    # -----------------------------
    # 1. 计算 Hint Level (根据轮数 + 关键词)
    current_hint_level = determine_hint_level(state, q)
    state["hint_level"] = current_hint_level
    
    # 2. 获取 Prompt 策略文本
    strategy_instruction = get_strategy_prompt(current_hint_level)
    
    # 3. 动态生成最终 Prompt
    final_prompt = BASE_PROMPT.format(current_strategy_instruction=strategy_instruction)

    # -----------------------------
    # Default: your original tool-loop agent
    # -----------------------------
    i = 0
    # 注意：这里使用 final_prompt 初始化 Agent
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
        # 调用 Agent
        # Agent.__call__ 会处理 history 的 append（在内存中），
        # 但我们需要确保返回的 history 列表正确反映了对话流
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

            # 工具返回值可能是字符串或字典，展示/日志层统一转成可读文本
            obs_output_str = _coerce_to_text(observation)

            tool_traces.append({
                "tool": action,
                "input": action_input,
                "output": obs_output_str[:2000],
            })

            next_prompt = f"检索结果：{obs_output_str}"
            continue

        # 没有工具调用，直接返回最终输出
        if last_citations and "引用：" not in (result or ""):
            result = (result or "").rstrip() + "\n\n" + _format_citations(last_citations)
        
        # 此时的 history 已经被 bot 在内部更新过了（如果引用传递生效）
        # 即使引用传递没生效，我们可以通过 append 手动维护
        if isinstance(history, list):
            # 为了保险起见，这里显式追加一次（假设 Agent 内部只是追加到了 self.messages 而不是 history 引用）
            # 根据你原本的 Agent.__call__ 代码：history.append(AIMessage(content=result)) 是在内部做的
            # 所以这里不需要额外 append，只需要返回传入的 history 引用即可
            pass

        return result, history, tool_traces, (state or {})

    # 超过 max_turns 兜底返回（仍然返回 4 元组）
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
    message="我对子网划分感到困惑"
    # 模拟第一次调用，状态为空
    state = {}
    output, history, _, new_state = query(message, state=state)
    print(output)
    print(f"DEBUG: Hint Level = {new_state.get('hint_level')}, Turn = {new_state.get('user_turn_count')}")

    message1="那我该怎么子网划分IP地址呢"
    # 模拟第二次调用，传入上一次的状态
    output1, history1, _, new_state1 = query(message1, history, state=new_state)
    print(output1)
    print(f"DEBUG: Hint Level = {new_state1.get('hint_level')}, Turn = {new_state1.get('user_turn_count')}")