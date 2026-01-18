# agentic_rag/socratic/ping_controller.py
from __future__ import annotations

# [MOD] 新增：json / pydantic / LLM messages
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.rag import RAGAgent  # 你项目里实际路径如不同请改
from agentic_rag.utils import extract_excerpt, _coerce_to_text
from agentic_rag.llm_config import build_chat_llm


IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

# [MOD] 新增：更正关键词（触发允许覆盖）
_CORRECTION_KEYWORDS = ("更正", "改成", "改为", "不是", "我写错了", "写错了", "正确是", "纠正")
def _allow_overwrite_slots(text: str) -> bool:
    t = text or ""
    return any(k in t for k in _CORRECTION_KEYWORDS)

# [MOD] 新增：LLM 槽位抽取结构
class PingSlotExtraction(BaseModel):
    src_ip: Optional[str] = Field(default=None, description="Source host IP, e.g. 192.168.1.10")
    src_mask: Optional[str] = Field(default=None, description="Source host mask, e.g. 255.255.255.0 or /24")
    src_gw: Optional[str] = Field(default=None, description="Source default gateway IP")
    dst_ip: Optional[str] = Field(default=None, description="Destination host IP")
    ping_gw_result: Optional[str] = Field(default=None, description="Ping output to gateway (truncate <=800 chars)")
    ping_dst_result: Optional[str] = Field(default=None, description="Ping output to destination (truncate <=800 chars)")

# [MOD] 新增：懒加载 LLM（避免 import 时初始化）
_SLOT_LLM: Optional[ChatDeepSeek] = None
def _get_slot_llm() -> ChatDeepSeek:
    global _SLOT_LLM
    if _SLOT_LLM is None:
        # _SLOT_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        _SLOT_LLM = build_chat_llm(temperature=0)
    return _SLOT_LLM

# [MOD] 新增：从 LLM 输出中提取 JSON（容错）
def _extract_json_object(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    # 优先找第一段 {...}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return "{}"

# [MOD] 新增：LLM 槽位抽取（结构化）
_SLOT_SYSTEM = """你是网络故障排查助手的“信息抽取器”。
任务：从用户输入中抽取 ping 不通排查所需的槽位信息，并输出严格 JSON（只输出 JSON，不要输出任何额外文本）。
槽位如下（可为空）：
- src_ip: 源主机IP
- src_mask: 源主机掩码（允许 255.255.255.0 或 /24）
- src_gw: 源主机默认网关
- dst_ip: 目标IP
- ping_gw_result: ping 网关的原始输出（如存在，截断 <=800 字符）
- ping_dst_result: ping 目标的原始输出（如存在，截断 <=800 字符）

规则：
1) 不要臆造。用户未提供则为 null。
2) 如果文本里同时出现多段 ping 输出，尽量区分“ping 网关”和“ping 目标”；无法区分则优先填 ping_dst_result。
3) 输出必须是单个 JSON object，key 只允许上述 6 个。
"""

def _llm_extract_ping_slots(text: str) -> Dict[str, Any]:
    llm = _get_slot_llm()
    try:
        resp = llm.invoke([
            SystemMessage(content=_SLOT_SYSTEM),
            HumanMessage(content=f"用户输入：\n{text}")
        ])
        content = getattr(resp, "content", str(resp))
        obj_text = _extract_json_object(content)
        data = json.loads(obj_text)
        parsed = PingSlotExtraction.model_validate(data).model_dump()
        # 只返回非空字段
        return {k: v for k, v in parsed.items() if v}
    except Exception:
        return {}


@dataclass
class PingState:
    scenario: str = "ping"
    hint_level: int = 0  # 0~3
    last_slot: Optional[str] = None
    fail_count_on_last_slot: int = 0
    slots: Dict[str, Any] = field(default_factory=dict)
    evidences: List[Dict[str, Any]] = field(default_factory=list)


PING_SLOTS_ORDER = [
    "src_ip",
    "src_mask",
    "src_gw",
    "dst_ip",
    "ping_gw_result",
    "ping_dst_result",
]

SLOT_ASK = {
    "src_ip": "请提供源主机的 IP 地址（贴出 ipconfig/ifconfig 关键行即可）。",
    "src_mask": "请提供源主机的子网掩码/前缀长度（同样贴出关键行）。",
    "src_gw": "请提供源主机的默认网关（gateway）是多少？",
    "dst_ip": "请提供目标主机的 IP 地址是多少？",
    "ping_gw_result": "请从源主机 ping 一下默认网关（第一跳）并把输出贴出来。",
    "ping_dst_result": "请从源主机 ping 目标 IP 并把输出贴出来（包含超时/不可达/回复等信息）。",
}

SLOT_HINTS = {
    "src_ip": [
        None,
        "通常在主机网卡配置里（IP Address/inet）那一行。",
        "如果你不确定，把 ipconfig/ifconfig 的整段网卡信息贴出来也可以。",
        "截图网卡配置页面/贴出完整输出都可以（我会帮你定位 IP 那一行）。",
    ],
    "src_mask": [
        None,
        "掩码可能是 255.255.255.0 或 /24 这种形式。",
        "请确认掩码与目标网段是否匹配（同网段才能直连）。",
        "直接贴 ipconfig/ifconfig 的整段网卡信息（含掩码/前缀）。",
    ],
    "src_gw": [
        None,
        "默认网关通常是同网段的路由器/SVI 地址（例如 x.x.x.1）。",
        "若是 VLAN 实验，网关可能是交换机的 Vlanif 地址。",
        "贴出 Default Gateway 或 route -n / ip route 的默认路由行。",
    ],
    "dst_ip": [
        None,
        "目标 IP 通常在题目/拓扑或目标主机配置里。",
        "请确认你 ping 的不是网段地址/广播地址。",
        "贴出目标主机的 IP 配置或题目里给的目标地址。",
    ],
    "ping_gw_result": [
        None,
        "在源主机执行：ping <默认网关IP>，看是否能通。",
        "如果 ping 网关都不通，优先检查二层/本机配置（IP/掩码/VLAN/端口up）。",
        "请贴完整输出（含丢包率/timeout/unreachable），不要只说“通/不通”。",
    ],
    "ping_dst_result": [
        None,
        "执行：ping <目标IP>，并观察是超时还是不可达（unreachable）。",
        "若能 ping 网关但不通目标，通常是路由/回程路由/ACL/NAT 等问题。",
        "请贴完整输出；如果可以，再补一条 traceroute（第一处失败点很关键）。",
    ],
}

# [MOD] 将原来的正则抽取逻辑保留为 fallback（不改变你原算法）
def _regex_extract_ping_slots(text: str) -> Dict[str, Any]:
    s: Dict[str, Any] = {}
    ips = IP_RE.findall(text or "")

    if ips:
        s.setdefault("src_ip", ips[0])
        if len(ips) >= 2:
            s.setdefault("dst_ip", ips[1])

    low = (text or "").lower()
    if ("reply from" in low) or ("bytes from" in low) or ("ttl=" in low) or ("timed out" in low) \
       or ("unreachable" in low) or ("请求超时" in (text or "")) or ("不可达" in (text or "")):
        s.setdefault("ping_dst_result", (text or "")[:800])

    return s

# [MOD] extract_ping_slots：优先用 LLM 结构化抽取，失败再 fallback
def extract_ping_slots(text: str) -> Dict[str, Any]:
    llm_slots = _llm_extract_ping_slots(text)
    if llm_slots:
        return llm_slots
    return _regex_extract_ping_slots(text)


def need_slot(state: PingState) -> Optional[str]:
    for slot in PING_SLOTS_ORDER:
        if not state.slots.get(slot):
            return slot
    return None


def ensure_ping_evidence(state: PingState) -> List[Dict[str, Any]]:
    """
    硬约束：至少调用一次工具（RAGAgent）。
    返回 tool_traces（便于前端 debug 展示）。
    """
    if state.evidences:
        return []

    q = "ping 不通 排查步骤 默认网关 二层 三层 路由表 ARP 常见原因"
    raw = RAGAgent(q)
    raw_text = _coerce_to_text(raw)
    state.evidences.append({
        "id": "E1",
        "query": q,
        "excerpt": extract_excerpt(raw),
        "raw_text": raw_text,
    })
    return [{"tool": "检索", "input": q, "output": raw_text[:2000]}]


def format_ping_reply(state: PingState, missing_slot: Optional[str], final: bool) -> str:
    ev_lines = []
    for ev in state.evidences[:2]:
        ev_lines.append(f'- {ev["id"]}（Query={ev["query"]}）：" {ev["excerpt"]} "')
    ev_block = "\n".join(ev_lines) if ev_lines else "暂无"

    if final:
        return f"""1) 你现在要我做什么：协助你定位“ping 不通”的根因，并给出可执行排查链（最终提示）。
2) 我需要你提供的内容：暂无（你已多轮未能提供关键输出，我先给通用排查链；你补充输出后可进一步精确定位）。
3) 证据：
{ev_block}
4) 我的问题：你愿意先按下面第 1~3 步执行并把输出贴回来吗？（这能把问题从“猜”变成“证据定位”）
5) 提示：按照从低成本到高成本的顺序排查：
   - Step1 本机配置：确认源主机 IP/掩码/网关正确（ipconfig/ifconfig）。
   - Step2 二层可达：ping 默认网关；若不通，检查端口 up、VLAN/Access/Trunk、ARP 表。
   - Step3 三层路由：若能 ping 网关但不通目标，查路由表（show/display ip route）、回程路由、ACL/NAT。
   - Step4 定位失败点：traceroute 目标，找到第一处失败跳。
6) 下一步：你贴出 Step1~Step2 的输出后，我会把排错链收敛到具体设备/接口/配置点。"""

    q = SLOT_ASK.get(missing_slot, "请补充关键配置与现象。") if missing_slot else "请补充更多现象与输出。"
    hint = SLOT_HINTS.get(missing_slot, [None, None, None, None])[min(state.hint_level, 3)]
    hint_line = hint if hint else "暂无"

    return f"""1) 你现在要我做什么：引导你一步步定位“ping 不通”的原因（不直接给答案）。
2) 我需要你提供的内容：{q}
3) 证据：
{ev_block}
4) 我的问题：{q}
   补充问题（可选）：你是在同网段直连还是跨网段通信？
5) 提示（Hint Level={state.hint_level}）：{hint_line}
6) 下一步：你提供该项输出后，我会判断问题是在“本机/二层/三层/策略”哪一层，并给出下一步最小检查。"""


def handle_ping_socratic(user_message: str, history, state_dict: Optional[Dict[str, Any]]) -> Tuple[str, Any, List[Dict[str, Any]], Dict[str, Any]]:
    # 恢复或初始化
    if state_dict and state_dict.get("scenario") == "ping":
        state = PingState(**state_dict)
    else:
        state = PingState()

    # [MOD] 关键：如果用户在本轮表达“更正/改成/不是/我写错了/正确是”，允许覆盖 slot
    allow_overwrite = _allow_overwrite_slots(user_message)

    # 抽槽位（LLM 结构化优先）
    extracted = extract_ping_slots(user_message)
    for k, v in extracted.items():
        if not v:
            continue
        if allow_overwrite:
            state.slots[k] = v  # 覆盖
        else:
            if not state.slots.get(k):
                state.slots[k] = v  # 仅填空

    tool_traces = []
    tool_traces += ensure_ping_evidence(state)

    missing = need_slot(state)

    # 升级 hint：如果上一轮问的 slot 仍未补齐
    if state.last_slot and not state.slots.get(state.last_slot):
        state.fail_count_on_last_slot += 1
        state.hint_level = min(3, state.hint_level + 1)

        # 你要求：hint_level=3 仍答不上来才给最终答案
        # 实现：已经到 3 且本轮依然没补齐 last_slot -> final
        if state.hint_level == 3 and state.fail_count_on_last_slot >= 1:
            reply = format_ping_reply(state, missing_slot=missing, final=True)
            return reply, history, tool_traces, state.__dict__

    # 正常推进：问缺失 slot
    if missing:
        state.last_slot = missing
        reply = format_ping_reply(state, missing_slot=missing, final=False)
    else:
        reply = format_ping_reply(state, missing_slot=None, final=True)

    # 如果 last_slot 已补齐，重置失败计数（可选：hint_level 是否归零取决于你想要的教学节奏）
    if state.last_slot and state.slots.get(state.last_slot):
        state.fail_count_on_last_slot = 0
        state.hint_level = 0

    return reply, history, tool_traces, state.__dict__
