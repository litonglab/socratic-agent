from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import your agent entrypoints
from agentic_rag.agent import query, dicts_to_messages, messages_to_dicts

app = FastAPI(title="Networking Lab Agent API")

# CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: set to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores (dev only; restart will lose)
SESSIONS: Dict[str, List[Dict[str, str]]] = {}
SESSION_STATES: Dict[str, Dict[str, Any]] = {}  # <-- [ADD] store Socratic state per session


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None  # optional: frontend-controlled history
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1) determine session_id
    session_id = req.session_id or f"s_{uuid4().hex[:10]}"

    # 2) get history: prefer request.history; else server memory
    if req.history is not None:
        history_dicts = req.history
    else:
        history_dicts = SESSIONS.get(session_id, [])

    # 3) dict -> langchain messages
    history_msgs = dicts_to_messages(history_dicts)

    # 3.5) get per-session socratic state
    state_dict = SESSION_STATES.get(session_id, None)

    # 4) call agent (NOTE: query must return 4-tuple now)
    reply, new_history_msgs, tool_traces, new_state = query(
        req.message,
        history=history_msgs,
        max_turns=req.max_turns,
        debug=req.debug,
        state=state_dict,
    )

    # 5) messages -> dict and persist in memory
    new_history_dicts = messages_to_dicts(new_history_msgs)
    SESSIONS[session_id] = new_history_dicts

    # 5.5) persist state in memory
    SESSION_STATES[session_id] = new_state or {}

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        history=new_history_dicts,
        tool_traces=[ToolTrace(**t) for t in tool_traces],
    )


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    # delete history
    if session_id in SESSIONS:
        del SESSIONS[session_id]

    # delete socratic state
    if session_id in SESSION_STATES:
        del SESSION_STATES[session_id]

    return {"ok": True}


# (Optional) useful endpoints for debugging / session management
@app.get("/api/sessions")
def list_sessions():
    return {"sessions": list(SESSIONS.keys())}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session_id,
        "history": SESSIONS[session_id],
        "state": SESSION_STATES.get(session_id, {}),
    }
