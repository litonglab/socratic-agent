from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import List, Optional, Dict, Any, Iterable, Tuple
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
from agentic_rag.agent import query, dicts_to_messages, messages_to_dicts
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
    append_log,
)

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


class ToolTrace(BaseModel):
    tool: str
    input: str
    output: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[Dict[str, str]]
    tool_traces: List[ToolTrace]


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

    session_snapshot = get_session(user_id, session_id)
    history_dicts = req.history if req.history is not None else session_snapshot.get("last_turns", [])
    summary = session_snapshot.get("summary", "")
    if summary:
        history_dicts = [{"role": "system", "content": f"对话摘要：{summary}"}] + history_dicts

    history_msgs = dicts_to_messages(history_dicts)
    state_dict = session_snapshot.get("state", None)

    reply, new_history_msgs, tool_traces, new_state = query(
        req.message,
        history=history_msgs,
        max_turns=req.max_turns,
        debug=req.debug,
        state=state_dict,
    )

    new_history_dicts = messages_to_dicts(new_history_msgs)
    filtered = [m for m in new_history_dicts if m.get("role") in {"user", "assistant"}]
    last_turns = filtered[-6:]
    summary = _update_summary(summary, req.message, reply)
    update_session(user_id, session_id, summary, last_turns, new_state or {})
    append_log(user_id, "request")

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        history=new_history_dicts,
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
        )
        resp = _handle_chat(req, user)
        return resp.dict()
    finally:
        _release_chat_slot()


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

def create_app() -> FastAPI:
    app = FastAPI(title="Networking Lab Agent API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest, authorization: Optional[str] = Header(None)):
        user = _require_user(authorization)
        _acquire_chat_slot()

        def event_stream():
            result_queue: queue.Queue = queue.Queue(maxsize=1)

            def _worker():
                try:
                    response = _handle_chat(req, user)
                    result_queue.put(("ok", response))
                except Exception as exc:
                    result_queue.put(("error", str(exc)))

            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()

            last_ping = time.monotonic()
            try:
                while True:
                    try:
                        status, payload = result_queue.get_nowait()
                        if status == "error":
                            yield _sse_event("error", {"detail": payload})
                            yield _sse_event("done", {"ok": False})
                            return
                        response: ChatResponse = payload
                        reply = response.reply or ""
                        for chunk in _chunk_text(reply, CHAT_STREAM_CHUNK_SIZE):
                            yield _sse_event("delta", {"content": chunk})
                        yield _sse_event("done", response.dict())
                        return
                    except queue.Empty:
                        now = time.monotonic()
                        if now - last_ping >= CHAT_STREAM_PING_INTERVAL:
                            yield _sse_event("ping", {"ts": time.time()})
                            last_ping = now
                        time.sleep(0.05)
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
            "last_turns": snapshot.get("last_turns", []),
            "state": snapshot.get("state", {}),
        }

    return app


# ✅ 保持兼容：如果未来你还想 uvicorn server:app 跑起来
app = create_app()
