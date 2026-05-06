from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import List, Optional, Dict, Any, Iterable, Tuple, Literal
from uuid import uuid4
from datetime import datetime, timezone

from dotenv import load_dotenv

# ✅ 尽早加载 env，避免后续导入的模块在 import 时拿到默认配置
load_dotenv()

# FastAPI 相关：保留，但不要在 import 时强制创建 app
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

# import your agent entrypoints
from agentic_rag.agent import (
    query,
    query_stream,
    dicts_to_messages,
    messages_to_dicts,
    _detect_lab4_section,
)
from agentic_rag.chat_format import split_visible_and_thinking
from agentic_rag.llm_config import build_chat_llm
from storage.auth import validate_password, validate_username, hash_password
from storage.user_store import (
    authenticate_user,
    create_user,
    find_user_by_username,
    issue_token_for_user,
    get_user_by_token,
    get_session,
    find_session,
    update_session,
    update_session_summary,
    delete_session as delete_user_session,
    list_user_session_snapshots,
    set_session_archived as set_user_session_archived,
    append_log,
    upsert_message_feedback,
    delete_message_feedback,
    get_message_feedback,
    record_interaction_metric,
)
from storage.proficiency import update_proficiency_from_metric

# -------------------------
# Lazy / shared configs
# -------------------------

SUMMARY_PROMPT = (
    "你是对话摘要器。给定已有摘要与最新一轮对话，输出更新后的简洁摘要，"
    "保留关键事实与上下文，不超过 200 字。"
)

MAX_CHAT_CONCURRENCY = int(os.getenv("MAX_CHAT_CONCURRENCY", "50"))
CHAT_QUEUE_TIMEOUT = float(os.getenv("CHAT_QUEUE_TIMEOUT", "30"))
CHAT_STREAM_CHUNK_SIZE = int(os.getenv("CHAT_STREAM_CHUNK_SIZE", "20"))
CHAT_STREAM_PING_INTERVAL = float(os.getenv("CHAT_STREAM_PING_INTERVAL", "0.6"))
MAX_RUNTIME_HISTORY_MESSAGES = int(os.getenv("MAX_RUNTIME_HISTORY_MESSAGES", "18"))
MAX_RUNTIME_HISTORY_CHARS = int(os.getenv("MAX_RUNTIME_HISTORY_CHARS", "12000"))
PERSISTED_LAST_TURNS = int(os.getenv("PERSISTED_LAST_TURNS", "6"))
CHAT_SEMAPHORE = threading.BoundedSemaphore(MAX_CHAT_CONCURRENCY)

# ✅ LLM 延迟初始化（import 时不会立刻构建）
_SUMMARY_LLM = None
_SUMMARY_LLM_LOCK = threading.Lock()
_SESSION_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()
_LEGACY_SESSION_USER_ID = "__legacy__"
_LEGACY_SESSIONS: Dict[str, List[Dict[str, str]]] = {}
_LEGACY_SESSION_STATES: Dict[str, Dict[str, Any]] = {}
_LEGACY_STORE_LOCK = threading.Lock()


def _anonymize_session_id(session_id: str) -> str:
    raw = (session_id or "").strip()
    if not raw:
        return "s_unknown"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"s_{digest}"


def _infer_lab4_problem_type(question: str, section: Optional[str]) -> Optional[str]:
    """基于轻量关键词推断实验4问题类型，仅用于课堂评估日志标签。"""
    q = (question or "").strip().lower()
    if not q:
        return None

    if section == "4.6_acceptance" or any(k in q for k in ("截图", "验收", "评分", "分值", "要交", "提交")):
        return "screenshot_acceptance"
    if any(k in q for k in ("报错", "失败", "不通", "连不上", "怎么办", "error", "file exists", "noqueue")):
        return "troubleshooting"
    if any(k in q for k in ("命令", "怎么做", "怎么写", "怎么配", "如何", "步骤", "参数", "-u", "-b")):
        return "command_execution"
    if any(k in q for k in ("报告", "分析", "总结", "规律", "简述", "为什么不是刚好")):
        return "report_analysis"
    if any(k in q for k in ("为什么", "结果", "差异", "不一样", "增加", "变化", "rtt")):
        return "phenomenon_explanation"
    if any(k in q for k in ("是什么", "区别", "作用", "入口", "出口", "怎么看", "含义", "什么意思")):
        return "concept_explanation"
    return None


def _append_classroom_eval_light_log(
    *,
    user_id: str,
    session_id: str,
    latency_ms: int,
    status: str,
    sse_completed: bool,
    input_chars: int,
    state: Optional[Dict[str, Any]],
    question: str,
) -> None:
    """课堂试用轻量日志：不记录原始问答文本，仅记录匿名标签与基础指标。"""
    payload: Dict[str, Any] = {
        "session_id": _anonymize_session_id(session_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": max(0, int(latency_ms)),
        "status": status,
        "sse_completed": bool(sse_completed),
        "input_chars": max(0, int(input_chars)),
    }

    st = state or {}
    if st.get("experiment_id") == "lab4":
        lab_section = st.get("lab_section") or _detect_lab4_section(question)
        problem_type = st.get("problem_type") or _infer_lab4_problem_type(question, lab_section)
        if lab_section:
            payload["lab_section"] = lab_section
        if problem_type:
            payload["problem_type"] = problem_type

    try:
        append_log(user_id, "classroom_eval_light", json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        print(f"[Warning] classroom_eval_light log failed: {exc}")


def get_summary_llm():
    global _SUMMARY_LLM
    if _SUMMARY_LLM is None:
        with _SUMMARY_LLM_LOCK:
            if _SUMMARY_LLM is None:
                _SUMMARY_LLM = build_chat_llm(temperature=0)
    return _SUMMARY_LLM


def _acquire_chat_slot() -> None:
    acquired = CHAT_SEMAPHORE.acquire(timeout=CHAT_QUEUE_TIMEOUT)
    if not acquired:
        raise HTTPException(status_code=429, detail="server busy, please retry later")


def _release_chat_slot() -> None:
    try:
        CHAT_SEMAPHORE.release()
    except ValueError:
        pass


def _get_session_lock(user_id: str, session_id: str) -> threading.Lock:
    key = (user_id, session_id)
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[key] = lock
        return lock


def _remove_session_lock(user_id: str, session_id: str) -> None:
    key = (user_id, session_id)
    with _SESSION_LOCKS_GUARD:
        _SESSION_LOCKS.pop(key, None)


@contextmanager
def _hold_session_lock(user_id: str, session_id: str):
    lock = _get_session_lock(user_id, session_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


# -------------------------
# Pydantic models (API & local)
# -------------------------

class ChatImageInput(BaseModel):
    base64: str
    mime: Optional[str] = None


class ChatFileInput(BaseModel):
    """非图片附件（pdf / docx / xlsx / pptx / zip / 纯文本/代码 等）。

    与 ChatImageInput 共用 ``base64`` 形式承载内容，但需要文件名以决定
    后端用哪种解析器；mime 仅作展示用，不参与分发。
    """
    name: str
    base64: str
    mime: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    debug: bool = False
    max_turns: int = 5
    # 重新生成 / 编辑重发场景：在沿用 stored_history 之前先截断到指定长度（含义：保留前 N 条对话）。
    # None 表示不截断；0 表示清空。仅影响本次 + 持久化；不会改变其他会话。
    truncate_history_to: Optional[int] = None
    enable_websearch: bool = True
    allow_process_explanations: bool = True
    images: Optional[List[ChatImageInput]] = None
    files: Optional[List[ChatFileInput]] = None


class ToolTrace(BaseModel):
    tool: str
    input: str
    output: str


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    reply: str
    thinking: str = ""
    history: List[Dict[str, str]]
    tool_traces: List[ToolTrace]


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    feedback: Literal["like", "dislike", "cancel"]


class RenameSessionRequest(BaseModel):
    title: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str
    student_id: str
    nickname: str
    class_name: str
    email: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: Dict[str, Any]


# -------------------------
# Auth helpers
# -------------------------

def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip()


def _require_user_by_token(token: str) -> Dict[str, Any]:
    user = get_user_by_token(token or "")
    if not user:
        raise HTTPException(status_code=401, detail="unauthorized")
    return user


def _require_user(authorization: Optional[str]) -> Dict[str, Any]:
    token = _extract_bearer_token(authorization)
    return _require_user_by_token(token or "")


def _resolve_optional_user(authorization: Optional[str]) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    token = _extract_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    return _require_user_by_token(token or "")


# -------------------------
# Chat payload helpers
# -------------------------

def _sanitize_message_dict(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    role = item.get("role")
    if role not in {"user", "assistant", "system"}:
        return None
    content = item.get("content", "") or ""
    if role == "assistant":
        visible, _ = split_visible_and_thinking(content)
        return {"role": role, "content": visible}
    return {"role": role, "content": content}


def _sanitize_history_dicts(items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    sanitized: List[Dict[str, str]] = []
    for item in items or []:
        normalized = _sanitize_message_dict(item)
        if normalized is not None:
            sanitized.append(normalized)
    return sanitized


def _filter_dialogue_history(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [item for item in items if item.get("role") in {"user", "assistant"}]


def _normalize_client_history(
    client_history: Optional[List[Dict[str, Any]]],
    current_message: str,
) -> List[Dict[str, str]]:
    sanitized = _sanitize_history_dicts(client_history)
    current = (current_message or "").strip()
    if (
        sanitized
        and current
        and sanitized[-1].get("role") == "user"
        and (sanitized[-1].get("content", "") or "").strip() == current
    ):
        sanitized = sanitized[:-1]
    return sanitized


def _history_char_count(items: List[Dict[str, str]]) -> int:
    return sum(len(item.get("content", "") or "") for item in items)


def _default_session_title(text: str, max_len: int = 12) -> str:
    content = (text or "").strip().replace("\n", " ")
    if len(content) <= max_len:
        return content or "新会话"
    return content[:max_len].rstrip() + "…"


def _session_title_from_messages(messages: List[Dict[str, Any]], fallback: str = "新会话") -> str:
    for item in messages or []:
        if item.get("role") == "user":
            title = _default_session_title(item.get("content", ""))
            if title:
                return title
    return fallback


def _copy_message_for_storage(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    copied = copy.deepcopy(item)
    # 历史会话回看：user 消息上的图片缩略图（image_b64）与文件 chip 元信息（files）
    # 都允许写入持久层。原图与文件原文不在这里产生，外层只会塞已经压缩好的缩略图 + 文件 metadata。
    if copied.get("role") == "assistant":
        visible, _ = split_visible_and_thinking(copied.get("content", "") or "")
        copied["content"] = visible
        traces = copied.get("tool_traces") or []
        copied["tool_traces"] = [dict(trace) for trace in traces if isinstance(trace, dict)]
    return copied


def _normalize_message_records(items: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items or []:
        copied = _copy_message_for_storage(item)
        if copied is not None:
            normalized.append(copied)
    return normalized


def _make_image_thumbnail_b64(raw_b64: str) -> Optional[str]:
    """把图片 base64 压成 ≤800x800 的 JPEG 缩略图 base64（不带 data: 前缀）。

    用途：让历史会话回看时仍能看到当时上传的图片，但不让 DB 被原图撑大。
    失败时返回 None，调用方应跳过这张缩略图。
    """
    try:
        import io as _io
        from PIL import Image
        raw = base64.b64decode(raw_b64)
        img = Image.open(_io.BytesIO(raw))
        img.thumbnail((800, 800))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=75, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # pragma: no cover
        print(f"[server] thumbnail failed: {exc}")
        return None


def _build_session_history_records(
    session_snapshot: Dict[str, Any],
    client_history: Optional[List[Dict[str, Any]]],
    current_message: str,
    assistant_message: str,
    thinking: str,
    message_id: str,
    tool_traces: List[Dict[str, Any]],
    *,
    user_image_b64: Optional[List[str]] = None,
    user_files: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, bool]:
    stored_history = _normalize_message_records(session_snapshot.get("history", []))
    if stored_history:
        base_messages = stored_history
    else:
        base_messages = _normalize_message_records(_normalize_client_history(client_history, current_message))

    full_messages = list(base_messages)
    user_record: Dict[str, Any] = {"role": "user", "content": current_message}
    # 仅在确实有内容时才挂字段，避免空数组占用存储
    if user_image_b64:
        user_record["image_b64"] = list(user_image_b64)
    if user_files:
        user_record["files"] = [dict(f) for f in user_files if isinstance(f, dict)]
    full_messages.append(user_record)
    full_messages.append({
        "role": "assistant",
        "content": assistant_message,
        "thinking": thinking or "",
        "tool_traces": [dict(trace) for trace in tool_traces or [] if isinstance(trace, dict)],
        "message_id": message_id,
        "feedback": None,
    })
    title = session_snapshot.get("title") or _session_title_from_messages(full_messages, "新会话")
    if title in {"", "新会话"}:
        title = _session_title_from_messages(full_messages, "新会话")
    dialogue_messages = [msg for msg in full_messages if msg.get("role") in {"user", "assistant"}]
    last_turns = dialogue_messages[-PERSISTED_LAST_TURNS:]
    archived = bool(session_snapshot.get("archived", False))
    return full_messages, last_turns, title, archived


def _choose_history_context(
    session_snapshot: Dict[str, Any],
    client_history: Optional[List[Dict[str, Any]]],
    current_message: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    summary = session_snapshot.get("summary", "") or ""
    stored_history = _filter_dialogue_history(_sanitize_history_dicts(session_snapshot.get("history", [])))
    stored_last_turns = _filter_dialogue_history(_sanitize_history_dicts(session_snapshot.get("last_turns", [])))
    normalized_client = _filter_dialogue_history(_normalize_client_history(client_history, current_message))

    authoritative_history = stored_history or stored_last_turns
    if not authoritative_history:
        authoritative_history = normalized_client

    if (
        authoritative_history
        and len(authoritative_history) <= MAX_RUNTIME_HISTORY_MESSAGES
        and _history_char_count(authoritative_history) <= MAX_RUNTIME_HISTORY_CHARS
    ):
        return authoritative_history, authoritative_history, summary

    runtime_history = stored_last_turns or authoritative_history[-PERSISTED_LAST_TURNS:]
    if summary:
        runtime_history = [{"role": "system", "content": f"对话摘要：{summary}"}] + runtime_history
    return runtime_history, authoritative_history, summary


def _finalize_history(raw_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    dialogue = _filter_dialogue_history(_sanitize_history_dicts(raw_history))
    return dialogue, dialogue[-PERSISTED_LAST_TURNS:]


def _persist_session_snapshot(
    *,
    summary: str,
    user_id: str,
    session_id: str,
    full_history: List[Dict[str, Any]],
    last_turns: List[Dict[str, Any]],
    state: Dict[str, Any],
    title: str,
    archived: bool,
) -> str:
    return update_session(
        user_id,
        session_id,
        summary or "",
        last_turns,
        state or {},
        history=full_history,
        title=title or "新会话",
        archived=archived,
    )


def _persist_chat_async(
    *,
    assistant_msg: str,
    user_id: str,
    session_id: str,
    state: Dict[str, Any],
    traces: List[Dict[str, Any]],
) -> None:
    def _persist():
        append_log(user_id, "request")
        record_interaction_metric(user_id, session_id, state or {}, traces or [], len(assistant_msg or ""))
        update_proficiency_from_metric(user_id, state or {})

    threading.Thread(target=_persist, daemon=True).start()


def _persist_session_summary(
    *,
    prev_summary: str,
    user_msg: str,
    assistant_msg: str,
    user_id: str,
    session_id: str,
    persisted_updated_at: Optional[str],
) -> None:
    try:
        new_summary = _update_summary(prev_summary, user_msg, assistant_msg)
        update_session_summary(
            user_id,
            session_id,
            new_summary,
            expected_updated_at=persisted_updated_at,
        )
    except Exception:
        pass


def set_session_archive_state(*, token: str, session_id: str, archived: bool) -> Dict[str, Any]:
    user = _require_user_by_token(token)
    with _hold_session_lock(user["id"], session_id):
        snapshot = find_session(user["id"], session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="session not found")
        set_user_session_archived(user["id"], session_id, archived)
        updated = find_session(user["id"], session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "archived": bool(updated.get("archived", archived)) if updated else archived,
        "updated_at": updated.get("updated_at") if updated else None,
    }


_MAX_SESSION_TITLE_LEN = 80


def rename_session_title(*, token: str, session_id: str, title: str) -> Dict[str, Any]:
    """前端"重命名会话"用：仅更新 title，保留其他字段不变。"""
    user = _require_user_by_token(token)
    new_title = (title or "").strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="title required")
    if len(new_title) > _MAX_SESSION_TITLE_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"title too long, max {_MAX_SESSION_TITLE_LEN} chars",
        )
    with _hold_session_lock(user["id"], session_id):
        snapshot = find_session(user["id"], session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="session not found")
        update_session(
            user["id"],
            session_id,
            summary=snapshot.get("summary", "") or "",
            last_turns=snapshot.get("last_turns", []) or [],
            state=snapshot.get("state", {}) or {},
            history=snapshot.get("history", []) or [],
            title=new_title,
            archived=bool(snapshot.get("archived", False)),
        )
        updated = find_session(user["id"], session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "title": new_title,
        "updated_at": updated.get("updated_at") if updated else None,
    }


# -------------------------
# Core logic (shared by HTTP endpoints)
# -------------------------

def _update_summary(prev_summary: str, user_msg: str, assistant_msg: str) -> str:
    human_input = f"已有摘要：{prev_summary or '（空）'}\n最新对话：\nUser: {user_msg}\nAssistant: {assistant_msg}"
    try:
        llm = get_summary_llm()
        resp = llm.invoke([
            SystemMessage(content=SUMMARY_PROMPT),
            HumanMessage(content=human_input),
        ])
        summary = getattr(resp, "content", "").strip()
        return summary or prev_summary
    except Exception:
        fallback = (prev_summary + "\n" + f"User: {user_msg}\nAssistant: {assistant_msg}").strip()
        return fallback[:400]


def _load_legacy_history(session_id: str, client_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    if client_history is not None:
        return _sanitize_history_dicts(client_history)
    with _LEGACY_STORE_LOCK:
        return copy.deepcopy(_LEGACY_SESSIONS.get(session_id, []))


def _load_legacy_state(session_id: str) -> Dict[str, Any]:
    with _LEGACY_STORE_LOCK:
        return copy.deepcopy(_LEGACY_SESSION_STATES.get(session_id, {}))


def _persist_legacy_session(session_id: str, history: List[Dict[str, str]], state: Dict[str, Any]) -> None:
    with _LEGACY_STORE_LOCK:
        _LEGACY_SESSIONS[session_id] = copy.deepcopy(history)
        _LEGACY_SESSION_STATES[session_id] = copy.deepcopy(state or {})


def _handle_legacy_chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or f"s_{uuid4().hex[:10]}"
    with _hold_session_lock(_LEGACY_SESSION_USER_ID, session_id):
        message_id = f"m_{uuid4().hex[:12]}"
        history_dicts = _load_legacy_history(session_id, req.history)
        history_msgs = dicts_to_messages(history_dicts)
        state_dict = _load_legacy_state(session_id) or None

        raw_reply, new_history_msgs, tool_traces, new_state = query(
            req.message,
            history=history_msgs,
            max_turns=req.max_turns,
            debug=req.debug,
            state=state_dict,
            user_id=None,
            enable_websearch=req.enable_websearch,
            allow_process_explanations=req.allow_process_explanations,
        )

        visible_reply, thinking = split_visible_and_thinking(raw_reply)
        response_history = _sanitize_history_dicts(messages_to_dicts(new_history_msgs))
        _persist_legacy_session(session_id, response_history, new_state or {})

        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            reply=visible_reply,
            thinking=thinking,
            history=response_history,
            tool_traces=[ToolTrace(**t) for t in _sanitize_tool_traces_for_client(tool_traces)],
        )


def _legacy_chat_stream_events(req: ChatRequest):
    session_id = req.session_id or f"s_{uuid4().hex[:10]}"
    with _hold_session_lock(_LEGACY_SESSION_USER_ID, session_id):
        message_id = f"m_{uuid4().hex[:12]}"
        history_dicts = _load_legacy_history(session_id, req.history)
        history_msgs = dicts_to_messages(history_dicts)
        state_dict = _load_legacy_state(session_id) or None

        yield _sse_event("meta", {"session_id": session_id, "message_id": message_id})

        final_result = ""
        final_history = history_msgs
        final_tool_traces: List[Dict[str, Any]] = []
        final_state = state_dict or {}

        for event in query_stream(
            req.message,
            history=history_msgs,
            max_turns=req.max_turns,
            debug=req.debug,
            state=state_dict,
            user_id=None,
            enable_websearch=req.enable_websearch,
            allow_process_explanations=req.allow_process_explanations,
        ):
            if event["type"] == "token":
                yield _sse_event("delta", {"content": event["content"]})
            elif event["type"] == "done":
                final_result = event["result"]
                final_history = event["history"]
                final_tool_traces = event["tool_traces"]
                final_state = event["state"]

        visible_reply, thinking = split_visible_and_thinking(final_result)
        response_history = _sanitize_history_dicts(messages_to_dicts(final_history))
        _persist_legacy_session(session_id, response_history, final_state or {})

        yield _sse_event("done", {
            "session_id": session_id,
            "message_id": message_id,
            "reply": visible_reply,
            "thinking": thinking,
            "history": response_history,
            "tool_traces": _sanitize_tool_traces_for_client(final_tool_traces),
        })


def _handle_chat(req: ChatRequest, user: Dict[str, Any]) -> ChatResponse:
    user_id = user["id"]
    session_id = req.session_id or f"s_{uuid4().hex[:10]}"
    with _hold_session_lock(user_id, session_id):
        message_id = f"m_{uuid4().hex[:12]}"

        session_snapshot = get_session(user_id, session_id)
        history_dicts, _authoritative_history, summary = _choose_history_context(
            session_snapshot,
            req.history,
            req.message,
        )
        history_msgs = dicts_to_messages(history_dicts)
        state_dict = session_snapshot.get("state", None)

        raw_reply, _new_history_msgs, tool_traces, new_state = query(
            req.message,
            history=history_msgs,
            max_turns=req.max_turns,
            debug=req.debug,
            state=state_dict,
            user_id=user_id,
            enable_websearch=req.enable_websearch,
            allow_process_explanations=req.allow_process_explanations,
        )

        visible_reply, thinking = split_visible_and_thinking(raw_reply)
        full_history, last_turns, session_title, session_archived = _build_session_history_records(
            session_snapshot,
            req.history,
            req.message,
            visible_reply,
            thinking,
            message_id,
            tool_traces,
        )
        response_history, _ = _finalize_history(full_history)
        persisted_updated_at = _persist_session_snapshot(
            summary=summary,
            user_id=user_id,
            session_id=session_id,
            full_history=full_history,
            last_turns=last_turns,
            state=new_state or {},
            title=session_title,
            archived=session_archived,
        )
        _persist_session_summary(
            prev_summary=summary,
            user_msg=req.message,
            assistant_msg=visible_reply,
            user_id=user_id,
            session_id=session_id,
            persisted_updated_at=persisted_updated_at,
        )
        _persist_chat_async(
            assistant_msg=visible_reply,
            user_id=user_id,
            session_id=session_id,
            state=new_state or {},
            traces=tool_traces,
        )

        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            reply=visible_reply,
            thinking=thinking,
            history=response_history,
            tool_traces=[ToolTrace(**t) for t in _sanitize_tool_traces_for_client(tool_traces)],
        )


def register_user(req: RegisterRequest) -> Dict[str, Any]:
    if not validate_username(req.username):
        raise HTTPException(status_code=400, detail="invalid username")
    if not validate_password(req.password):
        raise HTTPException(status_code=400, detail="invalid password")
    if find_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="username exists")

    salt, password_hash = hash_password(req.password)
    user = {
        "id": f"u_{uuid4().hex[:10]}",
        "username": req.username,
        "password_salt": salt,
        "password_hash": password_hash,
        "profile": {
            "name": req.name,
            "student_id": req.student_id,
            "nickname": req.nickname,
            "class_name": req.class_name,
            "email": req.email,
        },
        "preferences": {},
        "tokens": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_login_at": None,
    }
    create_user(user)
    token = issue_token_for_user(user, days=30)
    append_log(user["id"], "login", "register")
    return {"token": token, "user": {"id": user["id"], "username": user["username"], "profile": user["profile"]}}


def login_user(req: LoginRequest) -> Dict[str, Any]:
    user, ok = authenticate_user(req.username, req.password)
    if not ok or not user:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = issue_token_for_user(user, days=30)
    append_log(user["id"], "login", "password")
    return {"token": token, "user": {"id": user["id"], "username": user["username"], "profile": user["profile"]}}


# -------------------------
# SSE helpers (API only, keep for later)
# -------------------------

def _sse_event(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _sanitize_trace_output_for_client(text: str) -> str:
    """仅用于前端展示：移除 citation/source 标题行，保留正文。"""
    raw = (text or "").strip()
    if not raw:
        return raw
    if "引用：" in raw:
        raw = raw.split("引用：", 1)[0].rstrip()
    cleaned_lines: List[str] = []
    for line in raw.splitlines():
        if re.match(r"^\[\d+\]\s+.+$", line.strip()):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _sanitize_tool_traces_for_client(traces: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "tool": str(t.get("tool", "")),
            "input": str(t.get("input", "")),
            "output": _sanitize_trace_output_for_client(str(t.get("output", ""))),
        }
        for t in traces or []
        if isinstance(t, dict)
    ]


def _chunk_text(text: str, size: int) -> Iterable[str]:
    if size <= 0:
        yield text
        return
    for idx in range(0, len(text), size):
        yield text[idx: idx + size]


# -------------------------
# FastAPI app factory (optional)
# -------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    from agentic_rag.rag import _ensure_rag_initialized
    # 开发模式（RAG_DEV_FAST_START=1）：后台线程预热，服务器立即可访问，首次请求可能有冷启动
    # 生产模式（默认）：await 阻塞直到 RAG 完全就绪，uvicorn 在此期间不接受任何连接
    dev_fast = os.getenv("RAG_DEV_FAST_START", "0").lower() in {"1", "true", "yes"}
    if dev_fast:
        threading.Thread(target=_ensure_rag_initialized, daemon=True, name="rag-warmup").start()
    else:
        await asyncio.to_thread(_ensure_rag_initialized)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Networking Lab Agent API", lifespan=_lifespan)

    _cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/health/ready")
    def health_ready():
        """RAG 模型完全加载后才返回 200，供生产启动脚本轮询。"""
        from agentic_rag.rag import _initialized
        if _initialized:
            return {"status": "ready"}
        raise HTTPException(status_code=503, detail="initializing")

    @app.post("/api/register", response_model=AuthResponse)
    def register(req: RegisterRequest):
        data = register_user(req)
        return AuthResponse(**data)

    @app.post("/api/login", response_model=AuthResponse)
    def login(req: LoginRequest):
        data = login_user(req)
        return AuthResponse(**data)

    @app.get("/api/me")
    def me(authorization: Optional[str] = Header(None)):
        user = _require_user(authorization)
        return {"id": user["id"], "username": user["username"], "profile": user["profile"], "preferences": user.get("preferences", {})}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
        user = _resolve_optional_user(authorization)
        _acquire_chat_slot()
        try:
            if user is None:
                return _handle_legacy_chat(req)
            return _handle_chat(req, user)
        finally:
            _release_chat_slot()

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, authorization: Optional[str] = Header(None)):
        user = _require_user(authorization)
        user_id = user["id"]
        if req.feedback in {"like", "dislike"}:
            upsert_message_feedback(user_id, req.session_id, req.message_id, req.feedback)
            append_log(user_id, "feedback", f"{req.feedback}:{req.session_id}:{req.message_id}")
            value = get_message_feedback(user_id, req.session_id, req.message_id)
            return {"ok": True, "session_id": req.session_id, "message_id": req.message_id, "feedback": value}
        if req.feedback == "cancel":
            delete_message_feedback(user_id, req.session_id, req.message_id)
            append_log(user_id, "feedback", f"cancel:{req.session_id}:{req.message_id}")
            return {"ok": True, "session_id": req.session_id, "message_id": req.message_id, "feedback": None}
        raise HTTPException(status_code=400, detail="invalid feedback")

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest, authorization: Optional[str] = Header(None)):
        # 关键设计：
        #   - original_user_message：用户在输入框真正打的字，仅用于 dedupe / 持久化 / 摘要。
        #   - enriched_message：在原文基础上把附件抽取结果拼到前面，仅作为 LLM 的输入。
        # 这样 stored_history 里 user 消息的 content 永远是用户真实输入，
        # 重新打开会话时不会看到长长的附件抽取文本（与"附件本身不持久化"的设计一致）。
        original_user_message = req.message or ""
        enriched_message = original_user_message

        # 若客户端附带图片：先做 OCR / 视觉描述，再把内容拼到 enriched_message 前
        if req.images:
            try:
                from agentic_rag.vision import describe_image
                desc_lines: List[str] = []
                for i, img in enumerate(req.images):
                    if not img or not img.base64:
                        continue
                    try:
                        img_bytes = base64.b64decode(img.base64)
                        desc = describe_image(img_bytes, filename=f"image_{i}.png")
                        desc_lines.append(f"[图片 {i+1}]: {desc}")
                    except Exception as ex:
                        desc_lines.append(f"[图片 {i+1}]: (识别失败: {ex})")
                if desc_lines:
                    desc_block = "\n".join(desc_lines)
                    user_text = enriched_message.strip()
                    if user_text:
                        enriched_message = f"以下是用户上传的图片内容：\n{desc_block}\n\n用户的问题：{user_text}"
                    else:
                        enriched_message = f"以下是用户上传的图片内容：\n{desc_block}\n\n请根据图片内容回答。"
            except Exception as ex:  # pragma: no cover
                # vision 模块不可用时仅打印，不阻塞 chat
                print(f"[chat_stream] image preprocessing failed: {ex}")

        # 若客户端附带非图片附件（pdf/docx/xlsx/pptx/zip/纯文本等）：抽取文本后拼到 enriched_message 前。
        # 总字符上限 24000，避免 prompt 爆 token；超出则按附件顺序截断尾部。
        if req.files:
            try:
                from agentic_rag.file_extract import extract_text
                TOTAL_CHAR_LIMIT = 24000
                file_blocks: List[str] = []
                remaining = TOTAL_CHAR_LIMIT
                for i, f in enumerate(req.files):
                    if not f or not f.base64:
                        continue
                    name = (f.name or f"file_{i}").strip() or f"file_{i}"
                    try:
                        f_bytes = base64.b64decode(f.base64)
                        text, _supported = extract_text(name, f_bytes)
                    except Exception as ex:
                        text = f"[抽取失败: {ex}]"
                    if remaining <= 0:
                        file_blocks.append(f"[附件 {i+1} {name}]: (已超出总长度上限，未读取)")
                        continue
                    if len(text) > remaining:
                        text = text[:remaining] + "\n…[累计长度超限，已截断]"
                    remaining -= len(text)
                    file_blocks.append(f"[附件 {i+1} {name}]:\n{text}")
                if file_blocks:
                    files_block = "\n\n".join(file_blocks)
                    user_text = enriched_message.strip()
                    if user_text:
                        enriched_message = f"以下是用户上传的附件内容：\n{files_block}\n\n用户的问题：{user_text}"
                    else:
                        enriched_message = f"以下是用户上传的附件内容：\n{files_block}\n\n请根据附件内容回答。"
            except Exception as ex:  # pragma: no cover
                print(f"[chat_stream] file preprocessing failed: {ex}")

        user = _resolve_optional_user(authorization)
        _acquire_chat_slot()

        def event_stream():
            stream_started_at = time.perf_counter()
            trace_state: Dict[str, Any] = {}
            trace_user_id = user["id"] if user else _LEGACY_SESSION_USER_ID
            trace_sid = req.session_id or f"s_{uuid4().hex[:10]}"
            try:
                if user is None:
                    # legacy 路径行为保持不变：把 enriched_message 临时塞回 req.message 后再走老逻辑
                    saved = req.message
                    saved_sid = req.session_id
                    req.message = enriched_message
                    req.session_id = trace_sid
                    try:
                        yield from _legacy_chat_stream_events(req)
                    finally:
                        req.message = saved
                        req.session_id = saved_sid
                    _append_classroom_eval_light_log(
                        user_id=trace_user_id,
                        session_id=trace_sid,
                        latency_ms=int((time.perf_counter() - stream_started_at) * 1000),
                        status="ok",
                        sse_completed=True,
                        input_chars=len(original_user_message or ""),
                        state=trace_state,
                        question=original_user_message,
                    )
                    return
                user_id = user["id"]
                sid = trace_sid
                with _hold_session_lock(user_id, sid):
                    message_id = f"m_{uuid4().hex[:12]}"

                    session_snapshot = get_session(user_id, sid)
                    # 重新生成 / 编辑重发：在使用 stored_history 之前先截断到指定长度
                    if req.truncate_history_to is not None and req.truncate_history_to >= 0:
                        stored = session_snapshot.get("history", []) or []
                        if isinstance(stored, list):
                            keep = stored[: req.truncate_history_to]
                            session_snapshot["history"] = keep
                            session_snapshot["last_turns"] = keep[-PERSISTED_LAST_TURNS:]
                    # dedupe 用原始用户输入；附件文本不参与判断
                    history_dicts, _authoritative_history, summary = _choose_history_context(
                        session_snapshot,
                        req.history,
                        original_user_message,
                    )
                    history_msgs = dicts_to_messages(history_dicts)
                    state_dict = session_snapshot.get("state", None)

                    yield _sse_event("meta", {"session_id": sid, "message_id": message_id})

                    final_result = ""
                    final_tool_traces: list = []
                    final_state = state_dict or {}
                    final_thinking_full = ""

                    for event in query_stream(
                        enriched_message,
                        history=history_msgs,
                        max_turns=req.max_turns,
                        debug=req.debug,
                        state=state_dict,
                        user_id=user_id,
                        enable_websearch=req.enable_websearch,
                        allow_process_explanations=req.allow_process_explanations,
                    ):
                        if event["type"] == "token":
                            yield _sse_event("delta", {"content": event["content"]})
                        elif event["type"] == "thinking":
                            yield _sse_event("thinking_delta", {"content": event["content"]})
                        elif event["type"] == "stage":
                            stage_payload = {"stage": event.get("stage", "")}
                            if event.get("tools"):
                                stage_payload["tools"] = list(event["tools"])
                            yield _sse_event("stage", stage_payload)
                        elif event["type"] == "done":
                            final_result = event["result"]
                            final_tool_traces = event["tool_traces"]
                            final_state = event["state"]
                            trace_state = final_state or {}
                            final_thinking_full = event.get("thinking_full") or ""

                    visible_reply, thinking = split_visible_and_thinking(final_result)
                    # 把跨轮聚合的思考与最后一轮模型自带的 <思考> 合并：
                    # 用换行分隔，避免覆盖某一边的内容。
                    if final_thinking_full:
                        if thinking and final_thinking_full not in thinking:
                            thinking = (final_thinking_full + "\n\n" + thinking).strip()
                        elif not thinking:
                            thinking = final_thinking_full
                    # 持久化用原始用户输入；附件抽取出来的长文本不入库
                    # 但 user 消息上的"附件元信息"要写进 stored_history，
                    # 让重新打开会话时仍能看到当时的图片缩略图与文件名 chip。
                    user_image_thumbs: List[str] = []
                    if req.images:
                        for img in req.images:
                            if not img or not img.base64:
                                continue
                            thumb = _make_image_thumbnail_b64(img.base64)
                            if thumb:
                                user_image_thumbs.append(thumb)
                    user_files_meta: List[Dict[str, Any]] = []
                    if req.files:
                        for f in req.files:
                            if not f or not f.name:
                                continue
                            meta: Dict[str, Any] = {"name": f.name}
                            if f.base64:
                                # 仅用 base64 长度估算原始字节数（base64 长度 * 3/4），不存原文
                                try:
                                    meta["size"] = max(0, (len(f.base64) * 3) // 4)
                                except Exception:
                                    pass
                            user_files_meta.append(meta)
                    full_history, last_turns, session_title, session_archived = _build_session_history_records(
                        session_snapshot,
                        req.history,
                        original_user_message,
                        visible_reply,
                        thinking,
                        message_id,
                        final_tool_traces,
                        user_image_b64=user_image_thumbs or None,
                        user_files=user_files_meta or None,
                    )
                    response_history, _ = _finalize_history(full_history)
                    persisted_updated_at = _persist_session_snapshot(
                        summary=summary,
                        user_id=user_id,
                        session_id=sid,
                        full_history=full_history,
                        last_turns=last_turns,
                        state=final_state or {},
                        title=session_title,
                        archived=session_archived,
                    )
                    # 暴露 hint_level / category 等教学元信息给前端，用于渲染层级 chip
                    safe_state: Dict[str, Any] = {}
                    if isinstance(final_state, dict):
                        for key in ("hint_level", "question_category", "category"):
                            if key in final_state:
                                safe_state[key] = final_state[key]
                    yield _sse_event("done", {
                        "session_id": sid,
                        "message_id": message_id,
                        "reply": visible_reply,
                        "thinking": thinking,
                        "history": response_history,
                        "tool_traces": _sanitize_tool_traces_for_client(final_tool_traces),
                        "state": safe_state,
                    })
                    _persist_session_summary(
                        prev_summary=summary,
                        user_msg=original_user_message,
                        assistant_msg=visible_reply,
                        user_id=user_id,
                        session_id=sid,
                        persisted_updated_at=persisted_updated_at,
                    )
                    _persist_chat_async(
                        assistant_msg=visible_reply,
                        user_id=user_id,
                        session_id=sid,
                        state=final_state or {},
                        traces=final_tool_traces,
                    )
                    _append_classroom_eval_light_log(
                        user_id=trace_user_id,
                        session_id=trace_sid,
                        latency_ms=int((time.perf_counter() - stream_started_at) * 1000),
                        status="ok",
                        sse_completed=True,
                        input_chars=len(original_user_message or ""),
                        state=trace_state,
                        question=original_user_message,
                    )
            except Exception as exc:
                _append_classroom_eval_light_log(
                    user_id=trace_user_id,
                    session_id=trace_sid,
                    latency_ms=int((time.perf_counter() - stream_started_at) * 1000),
                    status="error",
                    sse_completed=False,
                    input_chars=len(original_user_message or ""),
                    state=trace_state,
                    question=original_user_message,
                )
                yield _sse_event("error", {"detail": str(exc)})
                yield _sse_event("done", {"ok": False})
            finally:
                _release_chat_slot()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str, authorization: Optional[str] = Header(None)):
        user = _resolve_optional_user(authorization)
        if user is None:
            with _hold_session_lock(_LEGACY_SESSION_USER_ID, session_id):
                with _LEGACY_STORE_LOCK:
                    _LEGACY_SESSIONS.pop(session_id, None)
                    _LEGACY_SESSION_STATES.pop(session_id, None)
            _remove_session_lock(_LEGACY_SESSION_USER_ID, session_id)
            return {"ok": True}
        with _hold_session_lock(user["id"], session_id):
            delete_user_session(user["id"], session_id)
        _remove_session_lock(user["id"], session_id)
        return {"ok": True}

    @app.get("/api/sessions")
    def list_sessions(authorization: Optional[str] = Header(None)):
        user = _resolve_optional_user(authorization)
        if user is None:
            with _LEGACY_STORE_LOCK:
                return {"sessions": [{"session_id": sid} for sid in _LEGACY_SESSIONS.keys()]}
        snapshots = list_user_session_snapshots(user["id"])
        sessions = []
        for s in snapshots:
            sessions.append({
                "session_id": s["session_id"],
                "title": s.get("title") or "新会话",
                "archived": bool(s.get("archived", False)),
                "updated_at": s.get("updated_at"),
            })
        return {"sessions": sessions}

    @app.post("/api/sessions/{session_id}/archive")
    def archive_session_endpoint(session_id: str, authorization: Optional[str] = Header(None)):
        token = _extract_bearer_token(authorization) or ""
        return set_session_archive_state(token=token, session_id=session_id, archived=True)

    @app.post("/api/sessions/{session_id}/unarchive")
    def unarchive_session_endpoint(session_id: str, authorization: Optional[str] = Header(None)):
        token = _extract_bearer_token(authorization) or ""
        return set_session_archive_state(token=token, session_id=session_id, archived=False)

    @app.patch("/api/sessions/{session_id}")
    def rename_session_endpoint(
        session_id: str,
        req: RenameSessionRequest,
        authorization: Optional[str] = Header(None),
    ):
        token = _extract_bearer_token(authorization) or ""
        return rename_session_title(token=token, session_id=session_id, title=req.title)

    @app.get("/api/sessions/{session_id}")
    def get_session_detail(session_id: str, authorization: Optional[str] = Header(None)):
        user = _resolve_optional_user(authorization)
        if user is None:
            with _LEGACY_STORE_LOCK:
                history = copy.deepcopy(_LEGACY_SESSIONS.get(session_id))
                state = copy.deepcopy(_LEGACY_SESSION_STATES.get(session_id, {}))
            if history is None:
                raise HTTPException(status_code=404, detail="session not found")
            return {
                "session_id": session_id,
                "history": history,
                "state": state,
            }
        snapshot = find_session(user["id"], session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="session not found")
        history = snapshot.get("history", []) or []
        return {
            "session_id": session_id,
            "title": snapshot.get("title") or "新会话",
            "archived": bool(snapshot.get("archived", False)),
            "summary": snapshot.get("summary", ""),
            "history": history,
            "messages": history,  # 与前端 React 客户端字段对齐
            "last_turns": snapshot.get("last_turns", []),
            "state": snapshot.get("state", {}),
            "updated_at": snapshot.get("updated_at"),
        }

    return app


# ✅ 保持兼容：如果未来你还想 uvicorn server:app 跑起来
app = create_app()
