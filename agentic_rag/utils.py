import re
import json
from typing import Any

# 用于从 RAG 返回文本中抽取“更像证据”的句子（偏网络实验场景关键词）
KEYWORDS = [
    "步骤", "检查", "预期", "原因", "常见", "故障", "排查",
    "OSPF", "BGP", "VLAN", "NAT", "ACL", "ARP", "STP",
    "show", "display", "ping", "traceroute", "邻居", "路由表",
]


def _coerce_to_text(raw: Any) -> str:
    """
    将可能的工具/RAG 输出（str/dict/None/其它）统一转为可读字符串，便于：
    - 摘录（extract_excerpt）
    - 前端 debug 展示（tool_traces）
    - 日志记录
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return str(raw)
    if isinstance(raw, dict):
        # 常见返回：{"result": "...", "context": "...", "sources": [...]}
        for k in ("result", "output_text", "answer", "context", "output"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def extract_excerpt(raw: Any, max_len: int = 180) -> str:
    raw_text = _coerce_to_text(raw)
    if not raw_text:
        return ""
    # 按句子/行切分
    parts = re.split(r"[。\n\r]+", raw_text)
    parts = [p.strip() for p in parts if p.strip()]

    # 优先选包含关键词的句子
    scored = []
    for p in parts:
        score = sum(1 for k in KEYWORDS if k.lower() in p.lower())
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0][1] if scored else raw_text.strip()
    if len(best) > max_len:
        best = best[:max_len].rstrip() + "…"
    return best