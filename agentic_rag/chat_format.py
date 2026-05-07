from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# 兜底：无论格式是否完整，只要 <tool_calls> 出现在可见文本里就整体剔除
_TOOL_CALLS_STRIP_RE = re.compile(r"<tool_calls>.*", re.IGNORECASE | re.DOTALL)

_THINKING_TAGS: Tuple[Tuple[str, str, bool], ...] = (
    ("<思考>", "</思考>", False),
    ("<thinking>", "</thinking>", True),
)


def split_assistant_content(text: str) -> Dict[str, Any]:
    raw = text or ""
    lowered = raw.lower()
    visible_parts: List[str] = []
    thinking_parts: List[str] = []
    cursor = 0
    saw_thinking_tag = False
    in_thinking = False

    while cursor < len(raw):
        next_match = None
        for open_tag, close_tag, case_insensitive in _THINKING_TAGS:
            search_source = lowered if case_insensitive else raw
            pos = search_source.find(open_tag, cursor)
            if pos != -1 and (next_match is None or pos < next_match[0]):
                next_match = (pos, open_tag, close_tag, case_insensitive)

        if next_match is None:
            visible_parts.append(raw[cursor:])
            break

        pos, open_tag, close_tag, case_insensitive = next_match
        saw_thinking_tag = True
        visible_parts.append(raw[cursor:pos])
        content_start = pos + len(open_tag)
        search_source = lowered if case_insensitive else raw
        close_pos = search_source.find(close_tag, content_start)

        if close_pos == -1:
            tail = raw[content_start:]
            if tail.strip():
                thinking_parts.append(tail.strip())
            in_thinking = True
            break

        thought = raw[content_start:close_pos].strip()
        if thought:
            thinking_parts.append(thought)
        cursor = close_pos + len(close_tag)

    return {
        "visible": "".join(visible_parts).strip(),
        "thinking": "\n\n".join(part for part in thinking_parts if part).strip(),
        "has_thinking": saw_thinking_tag or bool(thinking_parts),
        "in_thinking": in_thinking,
        "raw": raw,
    }


def _strip_tool_calls(text: str) -> str:
    """从可见文本中移除任何残留的 <tool_calls> 块（含格式损坏的情况）。"""
    return _TOOL_CALLS_STRIP_RE.sub("", text or "").strip()


def split_visible_and_thinking(text: str) -> Tuple[str, str]:
    parsed = split_assistant_content(text)
    if parsed["has_thinking"]:
        return _strip_tool_calls(parsed["visible"]), parsed["thinking"]
    return _strip_tool_calls((text or "").strip()), ""
