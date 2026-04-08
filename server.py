from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Iterable, Tuple, Literal
from uuid import uuid4
from datetime import datetime, timezone

from dotenv import load_dotenv

# FastAPI 相关：保留，但不要在 import 时强制创建 app
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

# import your agent entrypoints
from agentic_rag.agent import query, query_stream, dicts_to_messages, messages_to_dicts
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
    delete_session as delete_user_session,
    list_user_sessions,
    list_user_session_snapshots,
    append_log,
    upsert_message_feedback,
    delete_message_feedback,
    get_message_feedback,
    record_interaction_metric,
)
from storage.proficiency import update_proficiency_from_metric

# ✅ 允许 Streamlit import 时也能加载 env
load_dotenv()

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

# ✅ LLM 延迟初始化（Streamlit import 时不会立刻构建）
_SUMMARY_LLM = None
_SUMMARY_LLM_LOCK = threading.Lock()


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


# -------------------------
# Pydantic models (API & local)
# -------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    debug: bool = False
    max_turns: int = 5
    enable_websearch: bool = True


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


def _choose_history_context(
    session_snapshot: Dict[str, Any],
    client_history: Optional[List[Dict[str, Any]]],
    current_message: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    summary = session_snapshot.get("summary", "") or ""
    stored_history = _filter_dialogue_history(_sanitize_history_dicts(session_snapshot.get("history", [])))
    stored_last_turns = _filter_dialogue_history(_sanitize_history_dicts(session_snapshot.get("last_turns", [])))
    normalized_client = _filter_dialogue_history(_normalize_client_history(client_history, current_message))

    authoritative_history = stored_history
    if normalized_client and len(normalized_client) >= len(authoritative_history):
        authoritative_history = normalized_client
    if not authoritative_history:
        authoritative_history = stored_last_turns

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


def _persist_chat_async(
    *,
    prev_summary: str,
    user_msg: str,
    assistant_msg: str,
    user_id: str,
    session_id: str,
    full_history: List[Dict[str, str]],
    last_turns: List[Dict[str, str]],
    state: Dict[str, Any],
    traces: List[Dict[str, Any]],
) -> None:
    def _persist():
        new_summary = _update_summary(prev_summary, user_msg, assistant_msg)
        update_session(
            user_id,
            session_id,
            new_summary,
            last_turns,
            state or {},
            history=full_history,
        )
        append_log(user_id, "request")
        record_interaction_metric(user_id, session_id, state or {}, traces or [], len(assistant_msg or ""))
        update_proficiency_from_metric(user_id, state or {})

    threading.Thread(target=_persist, daemon=True).start()


def load_user_session_cache(*, token: str) -> Dict[str, Any]:
    user = _require_user_by_token(token)
    snapshots = list_user_session_snapshots(user["id"])
    sessions: Dict[str, Dict[str, Any]] = {}
    for snapshot in snapshots:
        history = _filter_dialogue_history(
            _sanitize_history_dicts(snapshot.get("history") or snapshot.get("last_turns") or [])
        )
        sessions[snapshot["session_id"]] = {
            "messages": history,
            "summary": snapshot.get("summary", "") or "",
            "state": snapshot.get("state", {}) or {},
            "updated_at": snapshot.get("updated_at"),
        }
    return {"sessions": sessions}


# -------------------------
# Core logic (shared by API & Streamlit)
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


def _handle_chat(req: ChatRequest, user: Dict[str, Any]) -> ChatResponse:
    user_id = user["id"]
    session_id = req.session_id or f"s_{uuid4().hex[:10]}"
    message_id = f"m_{uuid4().hex[:12]}"

    session_snapshot = get_session(user_id, session_id)
    history_dicts, _authoritative_history, summary = _choose_history_context(
        session_snapshot,
        req.history,
        req.message,
    )
    history_msgs = dicts_to_messages(history_dicts)
    state_dict = session_snapshot.get("state", None)

    raw_reply, new_history_msgs, tool_traces, new_state = query(
        req.message,
        history=history_msgs,
        max_turns=req.max_turns,
        debug=req.debug,
        state=state_dict,
        user_id=user_id,
        enable_websearch=req.enable_websearch,
    )

    visible_reply, thinking = split_visible_and_thinking(raw_reply)
    full_history, last_turns = _finalize_history(messages_to_dicts(new_history_msgs))
    _persist_chat_async(
        prev_summary=summary,
        user_msg=req.message,
        assistant_msg=visible_reply,
        user_id=user_id,
        session_id=session_id,
        full_history=full_history,
        last_turns=last_turns,
        state=new_state or {},
        traces=tool_traces,
    )

    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        reply=visible_reply,
        thinking=thinking,
        history=full_history,
        tool_traces=[ToolTrace(**t) for t in tool_traces],
    )


# ✅ 给 Streamlit 调用的“纯函数入口”（不走 HTTP）
def chat_once(
    *,
    token: str,
    message: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    debug: bool = False,
    max_turns: int = 5,
    enable_websearch: bool = True,
) -> Dict[str, Any]:
    """
    Streamlit 直接调用：
      resp = chat_once(token=token, message="...", session_id="...", history=[...])
    返回 dict（可 JSON 序列化）。
    """
    user = _require_user_by_token(token)
    _acquire_chat_slot()
    try:
        req = ChatRequest(
            message=message,
            session_id=session_id,
            history=history,
            debug=debug,
            max_turns=max_turns,
            enable_websearch=enable_websearch,
        )
        resp = _handle_chat(req, user)
        return resp.dict()
    finally:
        _release_chat_slot()


def chat_once_stream(
    *,
    token: str,
    message: str,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    debug: bool = False,
    max_turns: int = 5,
    enable_websearch: bool = True,
):
    """
    Streamlit 直接调用的流式版本。yield 字典：
      {"type": "meta", "session_id": ..., "message_id": ...}
      {"type": "token", "content": "..."}
      {"type": "done", ...}
    """
    user = _require_user_by_token(token)
    user_id = user["id"]
    _acquire_chat_slot()

    try:
        sid = session_id or f"s_{uuid4().hex[:10]}"
        message_id = f"m_{uuid4().hex[:12]}"

        session_snapshot = get_session(user_id, sid)
        history_dicts, _authoritative_history, summary = _choose_history_context(
            session_snapshot,
            history,
            message,
        )
        history_msgs = dicts_to_messages(history_dicts)
        state_dict = session_snapshot.get("state", None)

        yield {"type": "meta", "session_id": sid, "message_id": message_id}

        final_result = ""
        final_history = history_msgs
        final_tool_traces: List[Dict[str, Any]] = []
        final_state = state_dict or {}

        for event in query_stream(
            message,
            history=history_msgs,
            max_turns=max_turns,
            debug=debug,
            state=state_dict,
            user_id=user_id,
            enable_websearch=enable_websearch,
        ):
            if event["type"] == "token":
                yield {"type": "token", "content": event["content"]}
            elif event["type"] == "done":
                final_result = event["result"]
                final_history = event["history"]
                final_tool_traces = event["tool_traces"]
                final_state = event["state"]

        visible_reply, thinking = split_visible_and_thinking(final_result)
        full_history, last_turns = _finalize_history(messages_to_dicts(final_history))
        _persist_chat_async(
            prev_summary=summary,
            user_msg=message,
            assistant_msg=visible_reply,
            user_id=user_id,
            session_id=sid,
            full_history=full_history,
            last_turns=last_turns,
            state=final_state or {},
            traces=final_tool_traces,
        )

        yield {
            "type": "done",
            "session_id": sid,
            "message_id": message_id,
            "reply": visible_reply,
            "thinking": thinking,
            "history": full_history,
            "tool_traces": final_tool_traces,
        }
    finally:
        _release_chat_slot()


def submit_feedback(*, token: str, session_id: str, message_id: str, feedback: str) -> Dict[str, Any]:
    user = _require_user_by_token(token)
    user_id = user["id"]
    if feedback in {"like", "dislike"}:
        upsert_message_feedback(user_id, session_id, message_id, feedback)
        append_log(user_id, "feedback", f"{feedback}:{session_id}:{message_id}")
        return {
            "ok": True,
            "session_id": session_id,
            "message_id": message_id,
            "feedback": get_message_feedback(user_id, session_id, message_id),
        }
    if feedback == "cancel":
        delete_message_feedback(user_id, session_id, message_id)
        append_log(user_id, "feedback", f"cancel:{session_id}:{message_id}")
        return {"ok": True, "session_id": session_id, "message_id": message_id, "feedback": None}
    raise HTTPException(status_code=400, detail="invalid feedback")


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
        user = _require_user(authorization)
        _acquire_chat_slot()
        try:
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
        user = _require_user(authorization)
        user_id = user["id"]
        _acquire_chat_slot()

        def event_stream():
            try:
                sid = req.session_id or f"s_{uuid4().hex[:10]}"
                message_id = f"m_{uuid4().hex[:12]}"

                session_snapshot = get_session(user_id, sid)
                history_dicts, _authoritative_history, summary = _choose_history_context(
                    session_snapshot,
                    req.history,
                    req.message,
                )
                history_msgs = dicts_to_messages(history_dicts)
                state_dict = session_snapshot.get("state", None)

                yield _sse_event("meta", {"session_id": sid, "message_id": message_id})

                final_result = ""
                final_history = history_msgs
                final_tool_traces: list = []
                final_state = state_dict or {}

                for event in query_stream(
                    req.message,
                    history=history_msgs,
                    max_turns=req.max_turns,
                    debug=req.debug,
                    state=state_dict,
                    user_id=user_id,
                    enable_websearch=req.enable_websearch,
                ):
                    if event["type"] == "token":
                        yield _sse_event("delta", {"content": event["content"]})
                    elif event["type"] == "done":
                        final_result = event["result"]
                        final_history = event["history"]
                        final_tool_traces = event["tool_traces"]
                        final_state = event["state"]

                visible_reply, thinking = split_visible_and_thinking(final_result)
                full_history, last_turns = _finalize_history(messages_to_dicts(final_history))
                _persist_chat_async(
                    prev_summary=summary,
                    user_msg=req.message,
                    assistant_msg=visible_reply,
                    user_id=user_id,
                    session_id=sid,
                    full_history=full_history,
                    last_turns=last_turns,
                    state=final_state or {},
                    traces=final_tool_traces,
                )

                yield _sse_event("done", {
                    "session_id": sid,
                    "message_id": message_id,
                    "reply": visible_reply,
                    "thinking": thinking,
                    "history": full_history,
                    "tool_traces": [{"tool": t["tool"], "input": t["input"], "output": t["output"]} for t in final_tool_traces],
                })
            except Exception as exc:
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
        user = _require_user(authorization)
        user_id = user["id"]
        delete_user_session(user_id, session_id)
        return {"ok": True}

    @app.get("/api/sessions")
    def list_sessions(authorization: Optional[str] = Header(None)):
        user = _require_user(authorization)
        user_id = user["id"]
        return {"sessions": list_user_sessions(user_id)}

    @app.get("/api/sessions/{session_id}")
    def get_session_detail(session_id: str, authorization: Optional[str] = Header(None)):
        user = _require_user(authorization)
        user_id = user["id"]
        snapshot = find_session(user_id, session_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "session_id": session_id,
            "summary": snapshot.get("summary", ""),
            "history": snapshot.get("history", []),
            "last_turns": snapshot.get("last_turns", []),
            "state": snapshot.get("state", {}),
            "updated_at": snapshot.get("updated_at"),
        }

    return app


# ✅ 保持兼容：如果未来你还想 uvicorn server:app 跑起来
app = create_app()
