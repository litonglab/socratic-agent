# app_streamlit.py
import json
import uuid
import os
import time
import streamlit as st
from typing import Dict, Any, Optional, Iterable, Tuple, List
from dotenv import load_dotenv

load_dotenv()

# ✅ 直接本地调用 server.py（不走 HTTP）
from server import (
    chat_once,
    register_user,
    login_user,
    RegisterRequest,
    LoginRequest,
)

PERSIST_PATH = os.getenv("PERSIST_PATH", os.path.join(os.path.dirname(__file__), "sessions.json"))
  # 会话持久化文件路径（可改）
USE_PSEUDO_STREAMING = True
PSEUDO_STREAM_DELAY = float(os.getenv("PSEUDO_STREAM_DELAY", "0.01"))
PSEUDO_STREAM_CHUNK = int(os.getenv("PSEUDO_STREAM_CHUNK", "20"))


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
    data = st.session_state.persisted_data
    try:
        with open(PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.sidebar.warning(f"会话持久化写入失败：{repr(e)}")


def _ensure_persisted_shape() -> Dict[str, Any]:
    data = _load_persisted()
    if not isinstance(data, dict):
        return {"users": {}}
    if "users" in data and isinstance(data["users"], dict):
        return data
    if "sessions" in data:
        return {"users": {"_legacy": {"sessions": data.get("sessions", {}), "active_session_id": data.get("active_session_id")}}}
    return {"users": {}}


def _get_user_store(username: str) -> Dict[str, Any]:
    data = st.session_state.persisted_data
    users = data.setdefault("users", {})
    return users.setdefault(username, {"sessions": {}, "active_session_id": None})


def _save_user_store(username: str, sessions: Dict[str, Any], active_session_id: Optional[str]) -> None:
    store = _get_user_store(username)
    store["sessions"] = sessions
    store["active_session_id"] = active_session_id
    _save_persisted()


def _normalize_sessions(sessions: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for sid, sess in (sessions or {}).items():
        if not isinstance(sess, dict):
            continue
        if "archived" not in sess:
            sess["archived"] = False
        normalized[sid] = sess
    return normalized


def _load_user_sessions(username: str) -> None:
    store = _get_user_store(username)
    st.session_state.sessions = _normalize_sessions(store.get("sessions", {}))
    st.session_state.active_session_id = store.get("active_session_id")
    _ensure_one_session()
    _save_persisted()


def _persist_current_user_sessions() -> None:
    if not st.session_state.current_user:
        return
    _save_user_store(
        st.session_state.current_user,
        st.session_state.sessions,
        st.session_state.active_session_id,
    )


def _ensure_one_session():
    if not st.session_state.sessions:
        sid = _new_local_session_id()
        st.session_state.sessions[sid] = {"title": "新会话", "messages": [], "archived": False}
        st.session_state.active_session_id = sid


def _switch_session(session_id: str):
    st.session_state.active_session_id = session_id
    _persist_current_user_sessions()


def _create_new_session():
    sid = _new_local_session_id()
    st.session_state.sessions[sid] = {"title": "新会话", "messages": [], "archived": False}
    st.session_state.active_session_id = sid
    _persist_current_user_sessions()


def _rename_key(old_key: str, new_key: str):
    st.session_state.sessions[new_key] = st.session_state.sessions.pop(old_key)
    st.session_state.active_session_id = new_key
    _persist_current_user_sessions()


def _set_session_archived(session_id: str, archived: bool) -> None:
    sess = st.session_state.sessions.get(session_id)
    if not sess:
        return
    sess["archived"] = archived
    st.session_state.sessions[session_id] = sess
    if archived and st.session_state.active_session_id == session_id:
        for sid, s in st.session_state.sessions.items():
            if not s.get("archived"):
                st.session_state.active_session_id = sid
                break
        else:
            _create_new_session()
    _persist_current_user_sessions()


def _archive_session(session_id: str) -> None:
    _set_session_archived(session_id, True)


def _unarchive_session(session_id: str) -> None:
    _set_session_archived(session_id, False)


def _delete_session(session_id: str):
    # ✅ B方案：不再请求后端删除（因为没有后端 HTTP）
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]

    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = None

    _ensure_one_session()
    _persist_current_user_sessions()


def _clear_all_sessions():
    st.session_state.sessions = {}
    st.session_state.active_session_id = None
    _ensure_one_session()
    _persist_current_user_sessions()


def _iter_chunks(text: str, chunk_size: int) -> Iterable[str]:
    if chunk_size <= 0:
        yield text
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# -----------------------------
# Streamlit state init (with persistence)
# -----------------------------
st.set_page_config(page_title="计算机网络实验课 AI 助教", layout="wide")
st.markdown("<h1 style='text-align:center;'>Networking Lab Agent</h1>", unsafe_allow_html=True)

if "persisted_data" not in st.session_state:
    st.session_state.persisted_data = _ensure_persisted_shape()

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

if "sync_history" not in st.session_state:
    st.session_state.sync_history = True

if "debug" not in st.session_state:
    st.session_state.debug = False

if "sessions" not in st.session_state or "active_session_id" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session_id = None

# 确保 active_session_id 合法
if st.session_state.current_user:
    if st.session_state.active_session_id not in st.session_state.sessions:
        st.session_state.active_session_id = next(iter(st.session_state.sessions.keys()), None)
        _ensure_one_session()
        _persist_current_user_sessions()


# -----------------------------
# Auth UI (local calls)
# -----------------------------
if not st.session_state.current_user:
    st.markdown(
        """
        <style>
        .auth-wrapper [data-baseweb="tab-list"] {justify-content: center;}
        .auth-wrapper [data-baseweb="tab"] {margin: 0 8px;}
        .auth-wrapper [data-testid="stForm"] {display: flex; flex-direction: column; align-items: center;}
        .auth-wrapper [data-testid="stForm"] label {text-align: left; width: 100%;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["登录", "注册"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("用户名")
                login_password = st.text_input("密码", type="password")
                login_submit = st.form_submit_button("登录")

            if login_submit:
                try:
                    data = login_user(LoginRequest(username=login_username, password=login_password))
                    st.session_state.auth_token = data.get("token")
                    st.session_state.current_user = data.get("user", {}).get("username")
                    _load_user_sessions(st.session_state.current_user)
                    st.rerun()
                except Exception as e:
                    st.error(f"登录失败：{repr(e)}")

        with tab_register:
            with st.form("register_form", clear_on_submit=False):
                reg_username = st.text_input("用户名（最多10位字母数字）")
                reg_password = st.text_input("密码（字母+数字，≥8位）", type="password")
                reg_name = st.text_input("姓名")
                reg_student_id = st.text_input("学号")
                reg_nickname = st.text_input("昵称")
                reg_class = st.text_input("班级（如计算机网络1班）")
                reg_email = st.text_input("邮箱")
                reg_submit = st.form_submit_button("注册")

            if reg_submit:
                try:
                    payload = RegisterRequest(
                        username=reg_username,
                        password=reg_password,
                        name=reg_name,
                        student_id=reg_student_id,
                        nickname=reg_nickname,
                        class_name=reg_class,
                        email=reg_email,
                    )
                    data = register_user(payload)
                    st.session_state.auth_token = data.get("token")
                    st.session_state.current_user = data.get("user", {}).get("username")
                    _load_user_sessions(st.session_state.current_user)
                    st.success("注册成功，已登录")
                    st.rerun()
                except Exception as e:
                    st.error(f"注册失败：{repr(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 用户")
    st.caption(f"当前用户：{st.session_state.current_user}")
    st.session_state.sync_history = st.toggle("同步历史到后端（B方案下仅用于摘要/状态一致性）", value=st.session_state.sync_history)
    st.session_state.manage_mode = st.toggle("批量管理模式", value=st.session_state.get("manage_mode", False))
    if st.button("退出登录"):
        st.session_state.current_user = None
        st.session_state.auth_token = None
        st.session_state.sessions = {}
        st.session_state.active_session_id = None
        st.rerun()

    st.markdown("---")
    st.session_state.debug = st.toggle("Debug（显示工具调用）", value=st.session_state.debug)

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

    unarchived_sessions = {sid: s for sid, s in st.session_state.sessions.items() if not s.get("archived")}
    archived_sessions = {sid: s for sid, s in st.session_state.sessions.items() if s.get("archived")}

    if st.session_state.manage_mode and unarchived_sessions:
        options = [f"{s.get('title', sid)} ({sid})" for sid, s in unarchived_sessions.items()]
        selected = st.multiselect("选择要操作的会话", options)
        selected_ids = [opt.split("(")[-1].rstrip(")") for opt in selected]
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("批量归档"):
                for sid in selected_ids:
                    _archive_session(sid)
                st.rerun()
        with col_b:
            if st.button("批量删除"):
                for sid in selected_ids:
                    _delete_session(sid)
                st.rerun()

    for sid, sess in unarchived_sessions.items():
        title = sess.get("title", sid)
        is_active = (sid == st.session_state.active_session_id)
        label = f"▶ {title}" if is_active else title
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        with col1:
            if st.button(label, key=f"switch_{sid}"):
                _switch_session(sid)
                st.rerun()
        with col2:
            if st.button("归档", key=f"archive_{sid}"):
                _archive_session(sid)
                st.rerun()
        with col3:
            if st.button("删除", key=f"delete_{sid}"):
                _delete_session(sid)
                st.rerun()

    with st.expander("已归档"):
        if st.session_state.manage_mode and archived_sessions:
            options = [f"{s.get('title', sid)} ({sid})" for sid, s in archived_sessions.items()]
            selected = st.multiselect("选择要恢复/删除的会话", options, key="archived_select")
            selected_ids = [opt.split("(")[-1].rstrip(")") for opt in selected]
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("批量取消归档"):
                    for sid in selected_ids:
                        _unarchive_session(sid)
                    st.rerun()
            with col_b:
                if st.button("批量删除", key="archive_batch_delete"):
                    for sid in selected_ids:
                        _delete_session(sid)
                    st.rerun()

        for sid, sess in archived_sessions.items():
            title = sess.get("title", sid)
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col1:
                if st.button(title, key=f"archived_view_{sid}"):
                    _switch_session(sid)
                    st.rerun()
            with col2:
                if st.button("恢复", key=f"unarchive_{sid}"):
                    _unarchive_session(sid)
                    st.rerun()
            with col3:
                if st.button("删除", key=f"archived_delete_{sid}"):
                    _delete_session(sid)
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
# Input + send (local chat_once)
# -----------------------------
user_input = st.chat_input("输入你的问题，例如：我对子网划分感到困惑")
if user_input:
    if not st.session_state.active_session_id:
        _create_new_session()
        active_id = st.session_state.active_session_id

    active_id = st.session_state.active_session_id
    sess = st.session_state.sessions[active_id]

    if sess.get("title") in (None, "", "新会话"):
        sess["title"] = _default_title_from_message(user_input)

    sess["messages"].append({"role": "user", "content": user_input})
    _persist_current_user_sessions()

    with st.chat_message("user"):
        st.markdown(user_input)

    payload_session_id = None if active_id.startswith("local_") else active_id

    history_payload = None
    if st.session_state.sync_history:
        history_payload = [{"role": m.get("role"), "content": m.get("content", "")} for m in sess.get("messages", [])]

    try:
        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown("思考中…")

            data = None
            assistant_text = ""
            tool_traces = []

            # ✅ 本地调用（不走 HTTP）
            with st.spinner("正在生成回复..."):
                data = chat_once(
                    token=st.session_state.auth_token or "",
                    message=user_input,
                    session_id=payload_session_id,
                    history=history_payload,
                    debug=st.session_state.debug,
                    max_turns=5,
                )

            assistant_text = (data or {}).get("reply", "") or ""
            tool_traces = (data or {}).get("tool_traces", []) or []

            thinking.empty()

            if USE_PSEUDO_STREAMING and assistant_text:
                placeholder = st.empty()
                buf = ""
                for chunk in _iter_chunks(assistant_text, PSEUDO_STREAM_CHUNK):
                    buf += chunk
                    placeholder.markdown(buf)
                    time.sleep(PSEUDO_STREAM_DELAY)
            else:
                st.markdown(assistant_text)

    except Exception as e:
        err_text = f"生成失败：{repr(e)}"
        sess["messages"].append({"role": "assistant", "content": err_text})
        _persist_current_user_sessions()
        with st.chat_message("assistant"):
            st.error(err_text)
        st.stop()

    if not data:
        raise RuntimeError("服务未返回有效数据")

    real_sid = data["session_id"]

    if active_id.startswith("local_"):
        _rename_key(active_id, real_sid)
        active_id = real_sid
        sess = st.session_state.sessions[active_id]

    sess["messages"].append({
        "role": "assistant",
        "content": assistant_text,
        "tool_traces": tool_traces
    })
    _persist_current_user_sessions()

    # 额外展示（避免上面的 streaming placeholder 被 rerun 覆盖）
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
