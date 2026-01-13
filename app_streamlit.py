# app_streamlit.py
import json
import uuid
import os
import requests
import streamlit as st
from typing import Dict, Any

BACKEND_BASE_URL = "http://127.0.0.1:8000"
PERSIST_PATH = "sessions.json"  # 会话持久化文件路径（可改）


# -----------------------------
# Helpers
# -----------------------------
def _new_local_session_id() -> str:
    return f"local_{uuid.uuid4().hex[:8]}"

def _default_title_from_message(text: str, max_len: int = 12) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t if t else "新会话"
    return t[:max_len].rstrip() + "…"

def _load_persisted() -> Dict[str, Any]:
    if not os.path.exists(PERSIST_PATH):
        return {}
    try:
        with open(PERSIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_persisted():
    data = {
        "sessions": st.session_state.sessions,
        "active_session_id": st.session_state.active_session_id,
    }
    try:
        with open(PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # 不中断 UI，只提示
        st.sidebar.warning(f"会话持久化写入失败：{repr(e)}")

def _ensure_one_session():
    """确保至少存在一个会话，并设置 active_session_id。"""
    if not st.session_state.sessions:
        sid = _new_local_session_id()
        st.session_state.sessions[sid] = {"title": "新会话", "messages": []}
        st.session_state.active_session_id = sid

def _switch_session(session_id: str):
    st.session_state.active_session_id = session_id
    _save_persisted()

def _create_new_session():
    sid = _new_local_session_id()
    st.session_state.sessions[sid] = {"title": "新会话", "messages": []}
    st.session_state.active_session_id = sid
    _save_persisted()

def _rename_key(old_key: str, new_key: str):
    """把 sessions 字典里的 key 从 old_key 改成 new_key（保持内容不变）。"""
    st.session_state.sessions[new_key] = st.session_state.sessions.pop(old_key)
    st.session_state.active_session_id = new_key
    _save_persisted()

def _delete_backend_session(session_id: str):
    """
    可选：如果后端提供 DELETE /api/sessions/{session_id}，则同步删除后端会话。
    没有该接口也没关系，失败就忽略。
    """
    try:
        requests.delete(f"{BACKEND_BASE_URL}/api/sessions/{session_id}", timeout=10)
    except Exception:
        pass

def _delete_session(session_id: str):
    # 先尝试同步删除后端（可选）
    if session_id.startswith("s_"):
        _delete_backend_session(session_id)

    # 删除前端会话
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]

    # 如果删的是当前会话，重置 active
    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = None

    # 保证至少一个会话存在
    _ensure_one_session()
    _save_persisted()

def _clear_all_sessions():
    # 可选：清空全部（不会逐个请求后端删除，避免慢）
    st.session_state.sessions = {}
    st.session_state.active_session_id = None
    _ensure_one_session()
    _save_persisted()


# -----------------------------
# Streamlit state init (with persistence)
# -----------------------------
st.set_page_config(page_title="计算机网络实验课 AI 助教", layout="wide")
st.title("Networking Lab Agent")

if "debug" not in st.session_state:
    st.session_state.debug = False

if "sessions" not in st.session_state or "active_session_id" not in st.session_state:
    persisted = _load_persisted()
    st.session_state.sessions = persisted.get("sessions", {}) if isinstance(persisted.get("sessions"), dict) else {}
    st.session_state.active_session_id = persisted.get("active_session_id")
    _ensure_one_session()
    _save_persisted()

# 确保 active_session_id 合法
if st.session_state.active_session_id not in st.session_state.sessions:
    st.session_state.active_session_id = next(iter(st.session_state.sessions.keys()), None)
    _ensure_one_session()
    _save_persisted()


# -----------------------------
# Sidebar: Debug + New + Delete + Session list
# -----------------------------
with st.sidebar:
    st.session_state.debug = st.toggle("Debug（显示工具调用）", value=st.session_state.debug)
    _save_persisted()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("新建会话"):
            _create_new_session()
            st.rerun()

    with col2:
        if st.button("删除当前"):
            _delete_session(st.session_state.active_session_id)
            st.rerun()

    with st.expander("更多操作", expanded=False):
        if st.button("清空所有会话"):
            _clear_all_sessions()
            st.rerun()

        st.caption(f"持久化文件：{PERSIST_PATH}")

    st.markdown("---")
    st.markdown("### 会话列表")

    # 显示会话列表：点击切换
    for sid, sess in st.session_state.sessions.items():
        title = sess.get("title", sid)
        is_active = (sid == st.session_state.active_session_id)
        label = f"▶ {title}" if is_active else title
        if st.button(label, key=f"switch_{sid}"):
            _switch_session(sid)
            st.rerun()


# -----------------------------
# Main: render current session messages
# -----------------------------
active_id = st.session_state.active_session_id
active_session = st.session_state.sessions.get(active_id, {"title": "新会话", "messages": []})
active_messages = active_session.get("messages", [])

st.caption(f"当前会话：{active_session.get('title', active_id)}")

for m in active_messages:
    role = m.get("role", "assistant")
    content = m.get("content", "")
    tool_traces = m.get("tool_traces", None)

    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

        if st.session_state.debug and role == "assistant" and tool_traces:
            with st.expander("工具调用详情", expanded=False):
                for t in tool_traces:
                    st.markdown(f"**Tool**: {t.get('tool')}")
                    st.code(t.get("input", ""))
                    st.text_area("output", t.get("output", ""), height=150)
                    st.divider()


# -----------------------------
# Input + send
# -----------------------------
user_input = st.chat_input("输入你的问题，例如：我对子网划分感到困惑")
if user_input:
    # 如果没有 active 会话，兜底新建
    if not st.session_state.active_session_id:
        _create_new_session()
        active_id = st.session_state.active_session_id

    active_id = st.session_state.active_session_id
    sess = st.session_state.sessions[active_id]

    # 自动命名：默认“新会话”则用第一条用户消息命名
    if sess.get("title") in (None, "", "新会话"):
        sess["title"] = _default_title_from_message(user_input)

    # 1) 写入用户消息
    sess["messages"].append({"role": "user", "content": user_input})
    _save_persisted()

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) 调后端
    payload_session_id = None if active_id.startswith("local_") else active_id

    payload = {
        "message": user_input,
        "session_id": payload_session_id,
        "history": None,  # 让后端按 session_id 记忆
        "debug": st.session_state.debug,
        "max_turns": 5,
    }

    try:
        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown("思考中…")  # 立即给用户反馈

            with st.spinner("正在生成回复..."):
                resp = requests.post(f"{BACKEND_BASE_URL}/api/chat", json=payload, timeout=180)
                resp.raise_for_status()
                data = resp.json()

            thinking.empty()  # 清掉“思考中…”
            st.markdown(data["reply"])

    except Exception as e:
        err_text = f"请求后端失败：{repr(e)}"
        sess["messages"].append({"role": "assistant", "content": err_text})
        _save_persisted()
        with st.chat_message("assistant"):
            st.error(err_text)
        st.stop()

    real_sid = data["session_id"]
    assistant_text = data.get("reply", "")
    tool_traces = data.get("tool_traces", [])

    # 3) 如果当前是 local 会话，替换为真实 session_id
    if active_id.startswith("local_"):
        _rename_key(active_id, real_sid)
        active_id = real_sid
        sess = st.session_state.sessions[active_id]

    # 4) 写入 assistant 消息（带 tool_traces）
    sess["messages"].append({
        "role": "assistant",
        "content": assistant_text,
        "tool_traces": tool_traces
    })
    _save_persisted()

    # 5) 渲染 assistant
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
        if st.session_state.debug and tool_traces:
            with st.expander("工具调用详情", expanded=False):
                for t in tool_traces:
                    st.markdown(f"**Tool**: {t.get('tool')}")
                    st.code(t.get("input", ""))
                    st.text_area("output", t.get("output", ""), height=150)
                    st.divider()

    st.rerun()
