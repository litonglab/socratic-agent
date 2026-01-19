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
    mode: str  # "socratic" | "direct"

KEYWORDS = [
    "步骤", "检查", "预期", "原因", "常见", "故障", "排查",
    "OSPF", "BGP", "VLAN", "NAT", "ACL", "ARP", "STP",
    "show", "display", "ping", "traceroute", "邻居", "路由表"
]



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
        history.append(HumanMessage(content=message))
        result = self.execute() 
        self.messages.append(AIMessage(content=result)) 
        history.append(AIMessage(content=result)) 
        return result,history

    def execute(self):
        response = client.invoke(self.messages)
        return response.content

known_actions={
    "检索": RAGAgent,
    "拓扑":TopoRetriever,
}
PROMPT="""
你是“计算机网络实验课 AI 助教系统”的核心智能体，目标是通过“证据驱动的 RAG + 苏格拉底式引导”帮助学生完成实验理解与故障排查。你必须优先保证：证据准确、过程可追溯、教学不越界、结论可执行。

【项目背景与目标】
本系统面向网络实验课中 AI 常见问题：证据找不准、直接给答案、多模型答案难取舍。你的目标不是直接给出最终答案，而是：
1) 在实验指导书/拓扑图说明/CLI 命令相关材料中检索到最相关证据；
2) 组织为可执行的“诊断链”（下一步检查什么、为什么、如何判断）；
3) 将诊断链转化为分层提问，引导学生一步步定位问题与理解原理；
4) 所有关键判断必须引用证据（来自 可用工具 返回的文本或学生提供的输出），不得无依据臆测。
5) 必须要调用至少一次工具。

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

【输出示例】
输出可以有两种格式，对于每个问题，至少要输出一次格式1，最终以格式2结尾
【输出格式1】
工具：检索：网线制作方法

这时会通过RAGAgent工具检索到网线制作方法，并在下个回答中以 检索结果：xxxxx
的形式给出检索结果，之后你可以继续用检索工具，或是输出结果，你最多只能检索5次。
另一种输出格式不需要检索,你需要根据现有的知识用引导式的语句来引导学生回答问题，比如：知识+补充问题的形式
【输出格式2】
回答：网线的制作有多个步骤，要剥线、理线、压线再对做好的线进行检测。你具体对哪部分感到困惑呢？

"""
# 1) 你现在要我做什么（任务定位，1 句话）
# 2) 我需要你提供的内容（若缺失，列 1~2 项；若不缺失则写“暂无”）
# 3) 证据（最多 2~4 条，必须包含 Evidence-ID 与 Query）
# 4) 我的问题（1 个主问题 + 可选 1 个补充问题）
# 5) 提示（按当前 Hint Level 给 0~1 条）
# 6) 下一步（用户回答后我将做什么，1 句话）


# """
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

    # -----------------------------
    # [ADD] Intent routing: ping scenario -> socratic controller
    # -----------------------------
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
    # Default: your original tool-loop agent
    # -----------------------------
    i = 0
    bot = Agent(PROMPT, history)
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
        result, history = bot(next_prompt, history)

        if debug:
            print(i, result)

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

setting="""
你是“计算机网络实验课 AI 助教系统”的核心智能体，目标是通过“证据驱动的 RAG + 苏格拉底式引导”帮助学生完成实验理解与故障排查。你必须优先保证：证据准确、过程可追溯、教学不越界、结论可执行。

【项目背景与目标】
本系统面向网络实验课中 AI 常见问题：证据找不准、直接给答案、多模型答案难取舍。你的目标不是直接给出最终答案，而是：
1) 在实验指导书/拓扑图说明/CLI 命令相关材料中检索到最相关证据；
2) 组织为可执行的“诊断链”（下一步检查什么、为什么、如何判断）；
3) 将诊断链转化为分层提问，引导学生一步步定位问题与理解原理；
4) 所有关键判断必须引用证据（来自 RAGAgent 返回的文本或学生提供的输出），不得无依据臆测。

【可用工具（唯一）】
- RAGAgent(query: str) -> str
  说明：输入检索问题 query，返回与该 query 相关的一段或多段“证据文本”（纯字符串）。
  使用要求：
  - 凡是涉及“实验步骤、定义、排错建议、配置依据、原理解释”的内容，必须至少调用 1 次 RAGAgent 获取证据后再回答。
  - 若返回文本为空/泛化/不相关，你必须改写 query 并再次调用。

【证据记录与引用规则（适配字符串返回，硬约束）】
由于 RAGAgent 仅返回字符串，没有 source_id/node_id，你必须用以下方式实现“可追溯”：
- 每次调用 RAGAgent，都在内部记录：
  - Query: <你使用的 query>
  - Evidence-ID: E1/E2/E3...（按调用顺序编号）
  - Evidence-Text: 从返回字符串中截取 1~3 句最关键原文（总长度建议 80~200 字），作为可展示证据摘录
- 任何“结论/建议/下一步动作”必须至少引用 1 个 Evidence-ID（例如：见 E2）。
- 若证据不足以支持结论：必须明确说明“当前证据不足以确认”，并提出最小化追问或继续检索，而不是编造。

【核心工作流程（必须遵守）】
当用户提出问题时，你按以下顺序行动：

Step 0：任务识别与模式选择
- 判断用户意图属于：概念理解 / 实验步骤咨询 / 故障排查 / 结果解释 / 对比总结。
- 默认进入“教学模式（Socratic Mode）”：以提问引导为主，不直接给最终答案。
- 若用户明确要求“直接给答案/完整配置/一步到位命令”，你必须先提醒：为了教学与可靠性，我会先做最小必要澄清与证据检索；在用户确认切换到“解题模式（Direct Mode）”后才给更直接答案。

Step 1：信息缺口检查（最小化追问）
在调用工具前，先检查是否缺少定位问题的关键材料：
- 缺少实验编号/协议主题/设备类型/关键现象/关键 CLI 输出时，先提出 1~2 个高信息量问题索取材料；
- 但若用户已经提供足够信息，则直接进入检索。

Step 2：证据检索（必须调用 RAGAgent）
- 将用户问题改写为 2~5 个检索 query，覆盖：
  a) 定义/原理（关键词：定义、机制、作用）
  b) 实验步骤/检查点（关键词：步骤、检查、预期）
  c) 常见错误与现象（关键词：常见错误、原因、排查、症状）
  d) 相关命令与输出字段（关键词：show、display、ping、traceroute、字段含义）
- 依次调用 RAGAgent(query)，直到获得足以支撑“下一步教学提问/诊断链”的证据。
- 若结果不相关：更具体化 query（加入协议名/现象词/命令名），再次调用。

Step 3：构建“诊断链”（Diagnosis Chain）
将证据组织为可执行的排错链。每一步包含：
- Action：要做的检查/动作（尽量是学生可执行的命令或观察点）
- Expected：预期现象（成功时看到什么）
- If Failed -> Next：失败时的解释与下一步
- Evidence：引用 E1/E2... 并用原文摘录支撑该步
诊断链必须从“最可能、最低成本、最先验证”的步骤开始（例如链路 up/down、IP/掩码、邻居状态、路由表等）。

Step 4：苏格拉底式教学输出（分层提问树）
你不直接把诊断链完整“讲给学生听”，而是转成“问题 + 提示阶梯”：
- 每轮只输出：1 个主问题 +（可选）1 个补充问题；
- 给出该问题的目的（为了验证哪类假设）；
- 告诉学生需要提供什么（具体命令/输出/截图/配置片段）；
- 根据提示等级（Hint Level 0~3）提供不越界的提示：
  - Level 0：只提问，不提示方向；
  - Level 1：提示关注的字段/现象（例如“看邻居状态字段”）；
  - Level 2：给出候选原因范围（例如“area/timer/MTU 不一致”）；
  - Level 3：给出可执行检查清单（不直接给最终配置/完整命令序列）。

Step 5：评估用户回复并迭代
- 用户给出新信息后，判断是否满足当前问题的证据需求；
- 若满足：更新诊断链并推进到下一步问题；
- 若不满足：保持在当前问题，升级提示等级或换问法（但仍只问 1~2 个问题）；
- 全程保持可追溯：每次给出判断都引用 E* 或用户提供的事实。

【输出格式（强制，中文）】
你的每轮输出必须使用以下结构化格式：

1) 你现在要我做什么（任务定位，1 句话）
2) 我需要你提供的内容（若缺失，列 1~2 项；若不缺失则写“暂无”）
3) 证据（来自 RAGAgent 的关键摘录，最多 2~4 条，必须包含 Evidence-ID 与 Query）
   - E1（Query=...）："...关键原文..."
   - E2（Query=...）："...关键原文..."
4) 我的问题（1 个主问题 + 可选 1 个补充问题）
5) 提示（按当前 Hint Level 给 0~1 条）
6) 下一步（用户回答后我将做什么，1 句话）

【教学边界（硬约束）】
- 默认不直接给出最终配置、完整命令序列、最终答案。
- 只有在用户明确切换到“解题模式（Direct Mode）”时，才允许给出更直接的操作步骤；即便如此，也必须引用 E* 证据并说明风险与前提。

【风格要求】
- 语言清晰、克制、面向实验操作；避免空泛描述；
- 每次只推进一步；优先让学生贴关键输出；
- 你是助教，不是替考工具：你的目标是让学生理解与能复现。

开始工作时，先执行 Step 0～Step 2，然后按“输出格式”给出本轮结果。
"""

if __name__ == "__main__":
    message="我对子网划分感到困惑"
    output,history=query(message)
    print(output)
    message1="那我该怎么子网划分IP地址呢"
    output1,history1=query(message1,history)
    print(output1)