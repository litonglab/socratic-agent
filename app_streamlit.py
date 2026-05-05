# app_streamlit.py
import base64
import json
import uuid
import os
import time
import inspect
import html
import re
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, Optional, Iterable, Tuple, List
from dotenv import load_dotenv

load_dotenv()

# ✅ 直接本地调用 server.py（不走 HTTP）
from server import (
    chat_once_stream,
    delete_session_for_user,
    submit_feedback,
    register_user,
    login_user,
    load_user_session_cache,
    set_session_archive_state,
    RegisterRequest,
    LoginRequest,
)
from agentic_rag.chat_format import split_assistant_content
from agentic_rag.vision import describe_image
from components.chat_input_images import chat_input_images

PERSIST_PATH = os.getenv("PERSIST_PATH", os.path.join(os.path.dirname(__file__), "sessions.json"))
  # 仅持久化轻量 UI 状态；会话内容以后端数据库为准
USE_PSEUDO_STREAMING = True
PSEUDO_STREAM_DELAY = float(os.getenv("PSEUDO_STREAM_DELAY", "0.01"))
PSEUDO_STREAM_CHUNK = int(os.getenv("PSEUDO_STREAM_CHUNK", "20"))
LOGO_ICON_PATH = os.path.join(os.path.dirname(__file__), "netruc_agent_logo_icon.png")
REGISTER_LABELS = {
    "username": "用户名\u200b",
    "password": "密码\u200b",
    "name": "姓名\u200b",
    "student_id": "学号\u200b",
    "nickname": "昵称\u200b",
    "class_name": "班级\u200b",
    "email": "邮箱\u200b",
}


def _load_logo_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except OSError:
        return ""


LOGO_DATA_URI = _load_logo_data_uri(LOGO_ICON_PATH)


def _logo_html(container_class: str) -> str:
    if LOGO_DATA_URI:
        return (
            f'<div class="{container_class} has-logo">'
            f'<img src="{LOGO_DATA_URI}" class="brand-logo-img" alt="NetRUC Agent Logo" />'
            "</div>"
        )
    return f'<div class="{container_class}">N</div>'


def _render_websearch_chip(scope: str) -> None:
    """在输入框上方右对齐渲染一个"联网搜索" chip 风格 toggle。

    scope 用于区分 widget key（empty / bottom）。
    """
    with st.container(key=f"ws_chip_wrap_{scope}"):
        cols = st.columns([4, 1.4])
        with cols[1]:
            st.session_state.enable_websearch = st.toggle(
                "🌐 联网搜索",
                value=st.session_state.enable_websearch,
                key=f"ws_chip_toggle_{scope}",
            )


def _register_username_error(username: str) -> Optional[str]:
    value = (username or "").strip()
    if not value:
        return "请输入用户名"
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return "用户名仅支持字母和数字"
    if len(value) > 10:
        return "用户名最多10位字母数字"
    return None


def _register_password_error(password: str) -> Optional[str]:
    value = password or ""
    if not value:
        return "请输入密码"
    if len(value) < 8:
        return "密码需至少8位"
    if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
        return "密码需同时包含字母和数字"
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return "密码仅支持字母和数字"
    return None


def _live_register_username_error(username: str) -> Optional[str]:
    value = (username or "").strip()
    if not value:
        return None
    if len(value) > 10:
        return "用户名最多10位字母数字"
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return "用户名仅支持字母和数字"
    return None


def _live_register_password_error(password: str) -> Optional[str]:
    value = password or ""
    if not value:
        return None
    if len(value) < 8:
        return "密码需至少8位"
    if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
        return "密码需同时包含字母和数字"
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return "密码仅支持字母和数字"
    return None


def _register_name_error(name: str) -> Optional[str]:
    if not (name or "").strip():
        return "请输入姓名"
    return None


def _live_register_name_error(name: str) -> Optional[str]:
    if not (name or "").strip():
        return None
    return None


def _register_student_id_error(student_id: str) -> Optional[str]:
    value = (student_id or "").strip()
    if not value:
        return "请输入学号"
    if len(value) > 32:
        return "学号长度不能超过32位"
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return "学号仅支持字母和数字"
    return None


def _live_register_student_id_error(student_id: str) -> Optional[str]:
    value = (student_id or "").strip()
    if not value:
        return None
    if len(value) > 32:
        return "学号长度不能超过32位"
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return "学号仅支持字母和数字"
    return None


def _register_nickname_error(nickname: str) -> Optional[str]:
    if not (nickname or "").strip():
        return "请输入昵称"
    return None


def _live_register_nickname_error(nickname: str) -> Optional[str]:
    if not (nickname or "").strip():
        return None
    return None


def _register_class_error(class_name: str) -> Optional[str]:
    if not (class_name or "").strip():
        return "请输入班级"
    return None


def _live_register_class_error(class_name: str) -> Optional[str]:
    if not (class_name or "").strip():
        return None
    return None


def _register_email_error(email: str) -> Optional[str]:
    value = (email or "").strip()
    if not value:
        return "请输入邮箱"
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
        return "请输入有效邮箱地址"
    return None


def _live_register_email_error(email: str) -> Optional[str]:
    value = (email or "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
        return "请输入有效邮箱地址"
    return None


def _friendly_login_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    detail = str(getattr(exc, "detail", "") or "").lower()

    if status_code == 401 or "invalid credentials" in detail:
        return "用户名或密码错误，请重新输入。"
    if status_code == 429:
        return "登录尝试过于频繁，请稍后再试。"
    return "登录失败，请稍后重试。"


def _render_register_field_state_css(*, invalid_labels: List[str], valid_labels: List[str]) -> None:
    if not invalid_labels and not valid_labels:
        return

    invalid_blocks: List[str] = []
    valid_blocks: List[str] = []

    for label in invalid_labels:
        safe_label = html.escape(label, quote=True)
        invalid_blocks.append(
            f"""
            [data-testid="stTextInputRootElement"]:has(input[aria-label="{safe_label}"]),
            [data-testid="stTextInputRootElement"]:has(input[aria-label="{safe_label}"]):focus-within {{
                border-color: #D93025 !important;
                box-shadow: 0 0 0 1px rgba(217, 48, 37, 0.32) !important;
            }}
            """
        )
    for label in valid_labels:
        safe_label = html.escape(label, quote=True)
        valid_blocks.append(
            f"""
            [data-testid="stTextInputRootElement"]:has(input[aria-label="{safe_label}"]),
            [data-testid="stTextInputRootElement"]:has(input[aria-label="{safe_label}"]):focus-within {{
                border-color: #1E8E3E !important;
                box-shadow: 0 0 0 1px rgba(30, 142, 62, 0.28) !important;
            }}
            """
        )

    invalid_css = "\n".join(invalid_blocks)
    valid_css = "\n".join(valid_blocks)

    st.markdown(
        f"""
        <style>
        {invalid_css}
        {valid_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Helpers
# -----------------------------
def _new_local_session_id() -> str:
    return f"local_{uuid.uuid4().hex[:8]}"


def _now_ts() -> float:
    return time.time()


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
    try:
        with open(PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.persisted_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.sidebar.warning(f"会话持久化写入失败：{repr(e)}")


def _ensure_persisted_shape() -> Dict[str, Any]:
    data = _load_persisted()
    if not isinstance(data, dict):
        return {"users": {}}
    normalized = {"users": {}}
    if "users" in data and isinstance(data["users"], dict):
        for username, store in data["users"].items():
            if not isinstance(store, dict):
                continue
            normalized["users"][username] = {
                "active_session_id": store.get("active_session_id"),
            }
        return normalized
    if "sessions" in data:
        normalized["users"]["_legacy"] = {"active_session_id": data.get("active_session_id")}
    return normalized


def _get_user_store(username: str) -> Dict[str, Any]:
    data = st.session_state.persisted_data
    users = data.setdefault("users", {})
    return users.setdefault(username, {"active_session_id": None})


def _save_user_store(username: str, active_session_id: Optional[str]) -> None:
    store = _get_user_store(username)
    store["active_session_id"] = active_session_id
    _save_persisted()


def _normalize_sessions(sessions: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for idx, (sid, sess) in enumerate((sessions or {}).items(), start=1):
        if not isinstance(sess, dict):
            continue
        if "archived" not in sess:
            sess["archived"] = False
        if "updated_at" not in sess:
            sess["updated_at"] = float(idx)
        normalized[sid] = sess
    return normalized


def _session_updated_at(sess: Dict[str, Any], fallback: float = 0.0) -> float:
    raw = sess.get("updated_at", fallback)
    try:
        return float(raw)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(raw)).timestamp()
        except (TypeError, ValueError):
            return fallback


def _session_title_from_messages(messages: List[Dict[str, Any]], fallback: str = "新会话") -> str:
    for message in messages or []:
        if message.get("role") == "user":
            title = _default_title_from_message(message.get("content", ""))
            if title:
                return title
    return fallback


def _merge_backend_sessions(
    local_sessions: Dict[str, Any],
    backend_sessions: Dict[str, Any],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for sid, remote in (backend_sessions or {}).items():
        local = local_sessions.get(sid, {})
        remote_messages = remote.get("messages", []) or []
        local_messages = local.get("messages", []) or []
        use_remote_messages = len(remote_messages) > len(local_messages)
        chosen_messages = remote_messages if use_remote_messages else local_messages
        merged[sid] = {
            "title": local.get("title") or _session_title_from_messages(chosen_messages, sid),
            "messages": chosen_messages,
            "archived": local.get("archived", False),
            "updated_at": max(
                _session_updated_at(local),
                _session_updated_at(remote),
            ),
        }
    for sid, local in local_sessions.items():
        if sid not in merged:
            merged[sid] = local
    return _normalize_sessions(merged)


def _sorted_session_items(
    sessions: Dict[str, Any],
    *,
    archived: Optional[bool] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any], float]] = []
    for idx, (sid, sess) in enumerate((sessions or {}).items(), start=1):
        if archived is not None and bool(sess.get("archived")) != archived:
            continue
        fallback = float(idx)
        items.append((sid, sess, _session_updated_at(sess, fallback)))
    items.sort(key=lambda item: item[2], reverse=True)
    return [(sid, sess) for sid, sess, _ in items]


def _pick_session_id(*, archived: Optional[bool] = None) -> Optional[str]:
    items = _sorted_session_items(st.session_state.sessions, archived=archived)
    if items:
        return items[0][0]
    if archived is not None:
        any_items = _sorted_session_items(st.session_state.sessions, archived=None)
        if any_items:
            return any_items[0][0]
    return None


def _touch_session(session_id: str) -> None:
    sess = st.session_state.sessions.get(session_id)
    if not sess:
        return
    sess["updated_at"] = _now_ts()
    st.session_state.sessions[session_id] = sess


def _load_user_sessions(username: str) -> None:
    store = _get_user_store(username)
    backend_sessions: Dict[str, Any] = {}
    if st.session_state.get("auth_token"):
        try:
            backend_sessions = load_user_session_cache(token=st.session_state.auth_token).get("sessions", {})
        except Exception:
            backend_sessions = {}
    existing_local = {
        sid: sess for sid, sess in (st.session_state.get("sessions") or {}).items()
        if str(sid).startswith("local_") and sess.get("messages")
    }
    merged = _normalize_sessions(backend_sessions)
    for sid, sess in existing_local.items():
        if sid not in merged:
            merged[sid] = sess
    st.session_state.sessions = merged
    preferred_session_id = store.get("active_session_id")
    if preferred_session_id in st.session_state.sessions:
        st.session_state.active_session_id = preferred_session_id
    else:
        st.session_state.active_session_id = _pick_session_id(archived=False)
    _ensure_one_session()
    _save_persisted()


def _persist_current_user_sessions() -> None:
    if not st.session_state.current_user:
        return
    _save_user_store(
        st.session_state.current_user,
        st.session_state.active_session_id,
    )


def _ensure_one_session():
    if not st.session_state.sessions:
        sid = _new_local_session_id()
        st.session_state.sessions[sid] = {
            "title": "新会话",
            "messages": [],
            "archived": False,
            "updated_at": _now_ts(),
        }
        st.session_state.active_session_id = sid
    elif not st.session_state.active_session_id or st.session_state.active_session_id not in st.session_state.sessions:
        st.session_state.active_session_id = _pick_session_id(archived=False)


def _switch_session(session_id: str):
    st.session_state.active_session_id = session_id
    _persist_current_user_sessions()


def _create_new_session():
    sid = _new_local_session_id()
    st.session_state.sessions[sid] = {
        "title": "新会话",
        "messages": [],
        "archived": False,
        "updated_at": _now_ts(),
    }
    st.session_state.active_session_id = sid
    _persist_current_user_sessions()


def _rename_key(old_key: str, new_key: str):
    st.session_state.sessions[new_key] = st.session_state.sessions.pop(old_key)
    st.session_state.active_session_id = new_key
    _persist_current_user_sessions()


def _is_backend_session(session_id: Optional[str]) -> bool:
    return bool(session_id) and not str(session_id).startswith("local_")


def _set_session_archived(session_id: str, archived: bool) -> None:
    sess = st.session_state.sessions.get(session_id)
    if not sess:
        return
    if _is_backend_session(session_id):
        set_session_archive_state(
            token=st.session_state.auth_token or "",
            session_id=session_id,
            archived=archived,
        )
    sess["archived"] = archived
    sess["updated_at"] = _now_ts()
    st.session_state.sessions[session_id] = sess
    if archived and st.session_state.active_session_id == session_id:
        next_sid = _pick_session_id(archived=False)
        if next_sid:
            st.session_state.active_session_id = next_sid
        else:
            _create_new_session()
    _persist_current_user_sessions()


def _archive_session(session_id: str) -> None:
    _set_session_archived(session_id, True)


def _unarchive_session(session_id: str) -> None:
    _set_session_archived(session_id, False)


def _delete_session(session_id: str):
    if _is_backend_session(session_id):
        delete_session_for_user(
            token=st.session_state.auth_token or "",
            session_id=session_id,
        )
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]

    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = _pick_session_id(archived=False)

    _ensure_one_session()
    _persist_current_user_sessions()


def _iter_chunks(text: str, chunk_size: int) -> Iterable[str]:
    if chunk_size <= 0:
        yield text
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _assistant_visible_content(message: Dict[str, Any]) -> str:
    if message.get("role") != "assistant":
        return message.get("content", "")
    if "thinking" in message:
        return message.get("content", "")
    parsed = split_assistant_content(message.get("content", ""))
    if parsed["has_thinking"]:
        return parsed["visible"]
    return message.get("content", "")


def _message_for_history(message: Dict[str, Any]) -> Optional[Dict[str, str]]:
    role = message.get("role")
    if role not in {"user", "assistant", "system"}:
        return None
    if role == "assistant":
        return {"role": role, "content": _assistant_visible_content(message)}
    return {"role": role, "content": message.get("content", "")}


def _render_assistant_content(content: str, thinking: Optional[str] = None) -> None:
    parsed = split_assistant_content(content)
    visible = content if thinking is not None else (parsed["visible"] if parsed["has_thinking"] else content)
    hidden = thinking if thinking is not None else parsed["thinking"]
    has_hidden = bool(hidden) or (thinking is None and parsed["has_thinking"])

    if has_hidden:
        with st.expander("Thinking", expanded=False):
            if hidden:
                st.markdown(hidden)
            else:
                st.caption("思考过程已隐藏")

    if (visible or "").strip():
        st.markdown(visible)


# -----------------------------
# Streamlit state init (with persistence)
# -----------------------------
st.set_page_config(page_title="人大计算机网络实验课 AI 助教", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --brand-red: #861A11;
        --brand-red-dark: #6E150E;
        --ink-900: #2A1F1D;
        --ink-700: #5E4F4D;
        --line-soft: #E9E2E0;
        --line-mid: #D8CECB;
        --shadow-soft: 0 8px 24px rgba(45, 21, 17, 0.08);
    }

    .stApp {
        color: var(--ink-900);
        background:
            radial-gradient(circle at 12% 26%, rgba(134, 26, 17, 0.06) 0, rgba(134, 26, 17, 0.0) 24%),
            radial-gradient(circle at 88% 18%, rgba(134, 26, 17, 0.05) 0, rgba(134, 26, 17, 0.0) 22%),
            linear-gradient(180deg, #FCFBFB 0%, #F7F5F4 100%);
        background-attachment: fixed;
    }

    [data-testid="stToolbar"],
    [data-testid="stHeaderActionElements"],
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    @supports selector(body:has(*)) {
        body:has([data-testid="stSidebar"]) .main .block-container {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line-soft);
            border-radius: 18px;
            box-shadow: var(--shadow-soft);
            min-height: calc(100vh - 1.8rem);
            margin-top: 0.45rem;
            margin-bottom: 0.7rem;
            padding-top: 1rem;
            padding-bottom: 0.85rem;
        }
    }

    .main .block-container {
        max-width: none;
        padding-left: 0.95rem;
        padding-right: 0.95rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.94);
        border-right: 1px solid var(--line-mid);
        box-shadow: inset -1px 0 0 rgba(134, 26, 17, 0.03);
    }

    [data-testid="stSidebarHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebarHeader"] > * {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 0.08rem;
        /* 给 user card fixed 定位留出 sidebar 底部空间，避免最后一条会话被遮挡 */
        padding-bottom: 5.5rem !important;
    }

    /* user card 的视觉样式（位置由 JS 设为 fixed 跟随 sidebar 底部） */
    [data-testid="stSidebar"] .st-key-sidebar_user_card {
        background: rgba(255, 255, 255, 0.97);
        border-top: 1px solid rgba(216, 206, 203, 0.65);
        backdrop-filter: blur(2px);
        padding: 0.55rem 1rem 0.5rem !important;
    }

    hr {
        border-color: var(--line-soft) !important;
    }

    h1, h2, h3, h4, h5, h6, .stMarkdown p {
        color: var(--ink-900);
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        position: sticky;
        top: 0;
        z-index: 30;
        margin: 0 0 0.95rem;
        padding: 0.35rem 0.2rem 0.55rem;
        background: rgba(255, 255, 255, 0.96);
        border-bottom: 1px solid rgba(216, 206, 203, 0.72);
        backdrop-filter: blur(2px);
    }

    .sidebar-brand-badge {
        width: 2.45rem;
        height: 2.45rem;
        border-radius: 0.72rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-weight: 800;
        letter-spacing: 0.02em;
        font-size: 1.18rem;
        background: linear-gradient(180deg, #9B2419 0%, #861A11 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 2px 6px rgba(88, 21, 14, 0.28);
    }

    .sidebar-brand-title {
        font-weight: 700;
        color: var(--brand-red);
        font-size: 1.52rem;
        line-height: 1.05;
        white-space: nowrap;
    }

    /* === 侧边栏底部用户卡片：核心定位规则在上面的 sidebar 选择器内 === */

    /* 隐藏 toggle button：仍可被 JS click，但不可见、不占空间 */
    .st-key-sidebar_user_card_toggle_btn {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        clip: rect(0 0 0 0);
    }

    /* 用户卡片可视部分 */
    .user-card-display {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        padding: 0.55rem 0.65rem;
        border-radius: 0.7rem;
        border: 1px solid #E6DBD8;
        background: linear-gradient(180deg, #FFFFFF 0%, #FAF5F3 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.92), 0 2px 8px rgba(38, 22, 19, 0.10);
        cursor: pointer;
        transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        user-select: none;
    }
    .user-card-display:hover {
        background: linear-gradient(180deg, #FFF9F7 0%, #F4E7E3 100%);
        border-color: #D0BFBB;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.92), 0 3px 12px rgba(38, 22, 19, 0.14);
    }

    .user-card-avatar {
        flex: 0 0 auto;
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, #A42A1E 0%, #861A11 100%);
        color: #fff;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 2px 6px rgba(102, 28, 20, 0.22);
    }

    .user-card-text {
        flex: 1 1 auto;
        min-width: 0;
        display: flex;
        flex-direction: column;
        line-height: 1.15;
    }

    .user-card-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--brand-red);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .user-card-class {
        margin-top: 0.12rem;
        font-size: 0.78rem;
        color: #8A7B78;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .user-card-arrow {
        flex: 0 0 auto;
        font-size: 1.1rem;
        color: #B5A5A2;
        line-height: 1;
        margin-left: 0.2rem;
    }

    /* 向上弹出的菜单：absolute 定位到卡片上方 */
    .st-key-sidebar_user_card .st-key-sidebar_user_panel {
        position: absolute !important;
        bottom: calc(100% + 0.45rem) !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 10000;
        background: #FFFFFF !important;
        border: 1px solid #E6DBD8 !important;
        border-radius: 0.86rem !important;
        box-shadow: 0 -8px 24px rgba(38, 22, 19, 0.16);
    }

    .st-key-sidebar_user_card .st-key-sidebar_user_panel [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        border-radius: 0.86rem !important;
        background: #FFFFFF !important;
        box-shadow: none !important;
    }

    .st-key-sidebar_user_card .st-key-sidebar_user_panel [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.5rem 0.6rem 0.3rem !important;
        background: #FFFFFF !important;
        border-radius: 0.86rem !important;
    }

    /* outside-click iframe 完全 0 占位 */
    .st-key-sidebar_user_card iframe {
        height: 0 !important;
        min-height: 0 !important;
        border: none !important;
        display: block !important;
        margin: 0 !important;
    }
    .st-key-sidebar_user_card [data-testid="stElementContainer"]:has(iframe) {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden;
    }

    /* === Websearch chip toggle (输入框上方右对齐) === */
    [class*="st-key-ws_chip_wrap_"] {
        margin-top: -0.1rem !important;
        margin-bottom: -0.35rem !important;
    }

    [class*="st-key-ws_chip_wrap_"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0 !important;
    }

    [class*="st-key-ws_chip_wrap_"] [data-testid="stColumn"]:last-child {
        display: flex;
        justify-content: flex-end;
    }

    [class*="st-key-ws_chip_toggle_"] label {
        font-size: 0.82rem !important;
        color: #6F4D45 !important;
        gap: 0.4rem !important;
    }

    [class*="st-key-ws_chip_toggle_"] label p {
        font-size: 0.82rem !important;
        color: #6F4D45 !important;
    }

    /* === 侧边栏"新建会话"长条按钮（图标 + 文字，位于两条分隔线之间） === */
    .st-key-sidebar_new_session_btn button {
        width: 100% !important;
        height: auto !important;
        min-height: 2.5rem;
        border-radius: 0.7rem !important;
        padding: 0.5rem 0.85rem !important;
        border: 1px solid #D0C2BE !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #F7F3F2 100%) !important;
        color: var(--brand-red) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.92), 0 2px 6px rgba(41, 23, 19, 0.10);
        font-weight: 600 !important;
        text-align: left !important;
    }

    .st-key-sidebar_new_session_btn button:hover {
        border-color: #CDBFBB !important;
        background: linear-gradient(180deg, #FFF9F7 0%, #F6EBE8 100%) !important;
        color: var(--brand-red-dark) !important;
    }

    .st-key-sidebar_new_session_btn button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 0.55rem !important;
        width: 100% !important;
    }

    .st-key-sidebar_new_session_btn button [data-testid="stIconMaterial"] {
        font-size: 1.15rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        flex: 0 0 auto;
    }

    .st-key-sidebar_new_session_btn button p {
        margin: 0 !important;
        font-weight: 600 !important;
    }

    /* === 设置浮层内"已归档" expander 紧凑样式 === */
    .st-key-sidebar_user_card .st-key-settings_archived_wrap [data-testid="stExpander"] {
        border: 1px solid #ECE0DD !important;
        border-radius: 0.6rem !important;
        background: #FAF6F4 !important;
        margin-top: 0.2rem;
    }

    .st-key-sidebar_user_card .st-key-settings_archived_wrap [data-testid="stExpander"] summary {
        padding: 0.35rem 0.55rem !important;
        font-size: 0.85rem !important;
    }

    .st-key-sidebar_user_card .st-key-settings_archived_wrap [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        max-height: 40vh;
        overflow-y: auto;
        padding: 0.3rem 0.45rem 0.45rem !important;
    }

    /* === 会话列表行：ChatGPT 风格（无边框 / 左对齐 / 活跃高亮 / hover 灰底） === */
    /* CATCH-ALL：sidebar 内所有 tertiary button（含 switch_* 与 popover trigger）全部去边框 */
    [data-testid="stSidebar"] button[kind="tertiary"],
    [data-testid="stSidebar"] button[kind="tertiary"]:hover,
    [data-testid="stSidebar"] button[kind="tertiary"]:focus,
    [data-testid="stSidebar"] button[kind="tertiary"]:focus-visible,
    [data-testid="stSidebar"] button[kind="tertiary"]:active {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
        -webkit-tap-highlight-color: transparent !important;
    }

    [class*="st-key-session_row_"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    [class*="st-key-session_row_"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.1rem !important;
        margin: 0 !important;
    }

    /* hover 整行显示浅灰背景（包括三点列） */
    [class*="st-key-session_row_"]:hover {
        background: rgba(120, 100, 95, 0.08);
        border-radius: 0.55rem;
    }

    /* 活跃会话整行：浅红色背景 */
    [class*="st-key-session_row_"]:has(.session-row-active-marker) {
        background: rgba(150, 30, 30, 0.10);
        border-radius: 0.55rem;
    }
    [class*="st-key-session_row_"]:has(.session-row-active-marker):hover {
        background: rgba(150, 30, 30, 0.14);
    }

    /* 标题按钮：透明、无边框、左对齐+垂直居中 */
    [class*="st-key-session_row_"] [class*="st-key-switch_"] button,
    [class*="st-key-session_row_"] [class*="st-key-switch_"] button:hover,
    [class*="st-key-session_row_"] [class*="st-key-switch_"] button:focus,
    [class*="st-key-session_row_"] [class*="st-key-switch_"] button:focus-visible,
    [class*="st-key-session_row_"] [class*="st-key-switch_"] button:active {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [class*="st-key-session_row_"] [class*="st-key-switch_"] button {
        padding: 0.42rem 0.6rem !important;
        min-height: auto !important;
        height: auto !important;
        text-align: left !important;
        color: var(--ink-900, #2A1F1D) !important;
        font-weight: 400 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }

    [class*="st-key-session_row_"] [class*="st-key-switch_"] button:hover {
        color: var(--brand-red) !important;
    }

    [class*="st-key-session_row_"] [class*="st-key-switch_"] button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 0 !important;
        width: 100%;
    }

    [class*="st-key-session_row_"] [class*="st-key-switch_"] button p {
        margin: 0 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.92rem !important;
        line-height: 1.2;
    }

    /* 活跃会话标题加粗 + 深红 */
    [class*="st-key-session_row_"]:has(.session-row-active-marker) [class*="st-key-switch_"] button p {
        font-weight: 700 !important;
        color: var(--brand-red) !important;
    }

    /* 隐藏 popover trigger 的下拉箭头（chevron），只保留 ⋯ 图标 */
    [class*="st-key-session_row_"] [data-testid="stPopover"] button svg {
        display: none !important;
    }

    /* 三点 popover trigger：默认隐藏，hover 行时显示 */
    [class*="st-key-session_row_"] [data-testid="stPopover"] button {
        opacity: 0;
        transition: opacity 0.15s ease, background 0.15s ease;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.25rem !important;
        min-height: auto !important;
        height: 1.85rem !important;
        width: 1.85rem !important;
        min-width: 1.85rem !important;
        border-radius: 0.5rem !important;
        color: #7B6A68 !important;
    }

    [class*="st-key-session_row_"]:hover [data-testid="stPopover"] button,
    [class*="st-key-session_row_"] [data-testid="stPopover"][open] button,
    [class*="st-key-session_row_"] [data-testid="stPopover"] button[aria-expanded="true"] {
        opacity: 1;
    }

    [class*="st-key-session_row_"] [data-testid="stPopover"] button:hover {
        background: rgba(150, 30, 30, 0.08) !important;
        color: var(--brand-red) !important;
    }

    [class*="st-key-session_row_"] [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        font-size: 1.1rem !important;
        margin: 0 !important;
    }

    /* 隐藏 popover trigger 的占位文字（label 是空格） */
    [class*="st-key-session_row_"] [data-testid="stPopover"] button p {
        display: none !important;
    }

    /* popover 弹出内容：紧凑列表风格 */
    [data-testid="stPopoverBody"] {
        padding: 0.35rem !important;
        min-width: 9rem;
    }

    [data-testid="stPopoverBody"] button {
        text-align: left !important;
        padding: 0.4rem 0.6rem !important;
        font-size: 0.88rem !important;
        border-radius: 0.4rem !important;
    }

    [data-testid="stPopoverBody"] button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 0.5rem !important;
    }

    [data-testid="stPopoverBody"] button [data-testid="stIconMaterial"] {
        font-size: 1rem !important;
        margin: 0 !important;
    }

    .sb-section-title {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        margin: 0.35rem 0 0.15rem;
        color: var(--brand-red);
        font-size: 0.95rem;
        font-weight: 700;
    }

    .stCaption {
        color: var(--ink-700) !important;
    }

    .brand-logo-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }

    .auth-brand-badge.has-logo,
    .sidebar-brand-badge.has-logo,
    .empty-hero-logo.has-logo {
        background: transparent;
        box-shadow: none;
        padding: 0;
    }

    .stButton > button {
        white-space: nowrap !important;
        border-radius: 0.62rem;
        line-height: 1.15;
        font-weight: 600;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        border: 1px solid var(--line-mid) !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F2F1 100%);
        color: var(--brand-red);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.95), 0 1px 2px rgba(41, 23, 19, 0.1);
        transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.05s ease;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        border-color: #CABAB6 !important;
        background: linear-gradient(180deg, #FFF9F7 0%, #F6EBE8 100%);
        color: var(--brand-red-dark);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.96), 0 2px 4px rgba(40, 21, 18, 0.15);
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:active {
        transform: translateY(1px);
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.16);
    }

    [data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
        border: 1px solid #E5DCDA !important;
        background: #FFFFFF;
        color: #73504B;
        box-shadow: none;
    }

    [data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
        border-color: #D3C7C4 !important;
        background: #FAF7F6;
        color: #5F3C37;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #9D2419 0%, var(--brand-red) 100%);
        border: 1px solid #762018;
        color: #fff;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18), 0 2px 5px rgba(90, 23, 17, 0.22);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #892015 0%, var(--brand-red-dark) 100%);
        border-color: #651810;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16), 0 3px 6px rgba(75, 18, 13, 0.25);
    }

    .auth-shell {
        max-width: 820px;
        margin: 1.8rem auto 0.9rem;
    }

    .auth-header {
        text-align: center;
        margin-bottom: 0.95rem;
    }

    .auth-brand-badge {
        width: 3.65rem;
        height: 3.65rem;
        margin: 0 auto 0.45rem;
        border-radius: 1.05rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-size: 1.78rem;
        font-weight: 800;
        background: linear-gradient(180deg, #A42A1E 0%, #861A11 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.2), 0 6px 16px rgba(102, 28, 20, 0.24);
    }

    .auth-header h1 {
        margin: 0;
        color: var(--brand-red) !important;
        font-size: 2.25rem;
        line-height: 1.15;
        letter-spacing: 0.01em;
    }

    .auth-subtitle {
        margin-top: 0.22rem;
        color: #8A7B78;
        font-size: 0.98rem;
    }

    .auth-wrapper {
        background: #fff;
        border: 1px solid #ece5e3;
        border-top: none;
        border-radius: 1.1rem;
        box-shadow: 0 12px 32px rgba(53, 29, 24, 0.12);
        overflow: hidden;
        padding: 0;
    }

    .auth-wrapper [data-baseweb="tab-list"] {
        justify-content: flex-start;
        border-bottom: 1px solid #efe8e7;
        gap: 0.4rem;
        padding: 0 1.6rem;
    }

    .auth-wrapper [data-baseweb="tab"] {
        margin: 0;
        font-weight: 700;
        color: #7B6A68;
    }

    .auth-wrapper [aria-selected="true"] {
        color: var(--brand-red) !important;
        border-bottom-color: var(--brand-red) !important;
    }

    .auth-wrapper [data-testid="stForm"] {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        padding: 1.5rem 1.7rem 1.2rem;
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    .auth-wrapper [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1.5rem 1.7rem 1.2rem;
        border: 1px solid #ece5e3 !important;
        border-radius: 0.95rem !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .auth-wrapper [data-testid="stTextInputRootElement"] {
        border: 1px solid var(--line-mid);
        border-radius: 0.62rem;
        background: #fff;
        box-shadow: none;
    }

    .auth-wrapper [data-testid="stTextInputRootElement"]:focus-within {
        border-color: #B54F44;
        box-shadow: 0 0 0 1px rgba(134, 26, 17, 0.24);
    }

    .auth-wrapper [data-testid="stTextInputRootElement"] input {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 0.62rem !important;
        background: transparent !important;
    }

    [data-testid="InputInstructions"] {
        display: none !important;
    }

    [data-testid="stForm"] .stButton > button {
        margin-top: 0.2rem;
        min-height: 2.55rem;
        font-size: 1.04rem;
        letter-spacing: 0.02em;
    }

    [data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
        margin-top: 0.2rem;
        min-height: 2.55rem;
        font-size: 1.04rem;
        letter-spacing: 0.02em;
    }

    .register-field-error {
        margin: 0.22rem 0 0.62rem 0.1rem;
        color: #D93025;
        font-size: 0.86rem;
        line-height: 1.25;
    }

    .empty-hero {
        text-align: center;
        margin-top: 2.7rem;
        color: #8B7E7B;
    }

    .empty-hero-logo {
        width: 3.1rem;
        height: 3.1rem;
        margin: 0 auto 0.42rem;
        border-radius: 0.95rem;
        background: linear-gradient(180deg, #A42A1E 0%, #861A11 100%);
        color: #fff;
        font-size: 1.55rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.2), 0 6px 15px rgba(102, 28, 20, 0.2);
    }

    .empty-hero h2 {
        margin: 0.2rem 0 0;
        color: var(--brand-red) !important;
        font-size: 3.05rem;
        letter-spacing: 0.01em;
    }

    .empty-hero-subtitle {
        margin-top: 0.28rem;
        font-size: 1rem;
        color: #8C7D7A;
    }

    .empty-hero-divider {
        width: 2.3rem;
        height: 0.16rem;
        border-radius: 999px;
        margin: 0.95rem auto;
        background: linear-gradient(90deg, transparent 0%, var(--brand-red) 35%, var(--brand-red) 65%, transparent 100%);
        opacity: 0.82;
    }

    .empty-hero-tip {
        color: #8A7B78;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: 1px solid var(--line-soft);
        border-radius: 0.62rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "persisted_data" not in st.session_state:
    st.session_state.persisted_data = _ensure_persisted_shape()

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "current_user_profile" not in st.session_state:
    st.session_state.current_user_profile = {}

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

if "enable_websearch" not in st.session_state:
    st.session_state.enable_websearch = True

if "show_user_menu" not in st.session_state:
    st.session_state.show_user_menu = False

if "register_submit_attempted" not in st.session_state:
    st.session_state.register_submit_attempted = False

if "sessions" not in st.session_state or "active_session_id" not in st.session_state:
    st.session_state.sessions = {}
    st.session_state.active_session_id = None

# 确保 active_session_id 合法
if st.session_state.current_user:
    if st.session_state.active_session_id not in st.session_state.sessions:
        st.session_state.active_session_id = _pick_session_id(archived=False)
        _ensure_one_session()
        _persist_current_user_sessions()


# -----------------------------
# Auth UI (local calls)
# -----------------------------
if not st.session_state.current_user:
    st.markdown(
        """
        <style>
        /* 登录/注册页输入框：只保留一层焦点边框 */
        .auth-wrapper [data-testid="stTextInputRootElement"] {
            border: 1px solid #D8CECB;
            border-radius: 0.62rem;
            background: #fff;
            box-shadow: none;
        }

        .auth-wrapper [data-testid="stTextInputRootElement"]:focus-within {
            border-color: #B54F44;
            box-shadow: 0 0 0 1px rgba(134, 26, 17, 0.24);
        }

        .auth-wrapper [data-testid="stTextInputRootElement"] input,
        .auth-wrapper [data-testid="stTextInputRootElement"] input:focus {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }

        /* 去掉英文提示：Press Enter to submit form */
        [data-testid="InputInstructions"],
        [data-testid="stTextInput"] [aria-live="polite"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_center, col_right = st.columns([0.9, 1.5, 0.9])
    with col_center:
        auth_logo_html = _logo_html("auth-brand-badge")
        st.markdown(
            f"""
            <div class="auth-shell">
              <div class="auth-header">
                {auth_logo_html}
                <h1>NetRUC Agent</h1>
              </div>
              <div class="auth-wrapper">
            """,
            unsafe_allow_html=True,
        )
        tab_login, tab_register = st.tabs(["登录", "注册"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input(
                    "用户名",
                    placeholder="请输入用户名",
                    icon=":material/person:",
                )
                login_password = st.text_input(
                    "密码",
                    type="password",
                    placeholder="请输入密码",
                    icon=":material/lock:",
                )
                login_submit = st.form_submit_button("登录", type="primary", use_container_width=True)

            if login_submit:
                try:
                    data = login_user(LoginRequest(username=login_username, password=login_password))
                    st.session_state.auth_token = data.get("token")
                    user_obj = data.get("user", {}) or {}
                    st.session_state.current_user = user_obj.get("username")
                    st.session_state.current_user_profile = user_obj.get("profile", {}) or {}
                    _load_user_sessions(st.session_state.current_user)
                    st.rerun()
                except Exception as e:
                    st.error(_friendly_login_error(e))

        with tab_register:
            submit_attempted = bool(st.session_state.get("register_submit_attempted", False))

            with st.container(border=True):
                reg_username = st.text_input(
                    REGISTER_LABELS["username"],
                    placeholder="请输入用户名",
                    icon=":material/person:",
                    key="register_username_input",
                )
                username_error = _live_register_username_error(reg_username)
                if submit_attempted and username_error is None:
                    username_error = _register_username_error(reg_username)
                if username_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(username_error)}</div>",
                        unsafe_allow_html=True,
                    )

                reg_password = st.text_input(
                    REGISTER_LABELS["password"],
                    type="password",
                    placeholder="请输入密码",
                    icon=":material/lock:",
                    key="register_password_input",
                )
                password_error = _live_register_password_error(reg_password)
                if submit_attempted and password_error is None:
                    password_error = _register_password_error(reg_password)
                if password_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(password_error)}</div>",
                        unsafe_allow_html=True,
                    )

                reg_name = st.text_input(
                    REGISTER_LABELS["name"],
                    placeholder="请输入姓名",
                    icon=":material/badge:",
                    key="register_name_input",
                )
                name_error = _live_register_name_error(reg_name)
                if submit_attempted and name_error is None:
                    name_error = _register_name_error(reg_name)
                if name_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(name_error)}</div>",
                        unsafe_allow_html=True,
                    )

                reg_student_id = st.text_input(
                    REGISTER_LABELS["student_id"],
                    placeholder="请输入学号",
                    icon=":material/tag:",
                    key="register_student_id_input",
                )
                student_id_error = _live_register_student_id_error(reg_student_id)
                if submit_attempted and student_id_error is None:
                    student_id_error = _register_student_id_error(reg_student_id)
                if student_id_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(student_id_error)}</div>",
                        unsafe_allow_html=True,
                    )

                reg_nickname = st.text_input(
                    REGISTER_LABELS["nickname"],
                    placeholder="请输入昵称",
                    icon=":material/alternate_email:",
                    key="register_nickname_input",
                )
                nickname_error = _live_register_nickname_error(reg_nickname)
                if submit_attempted and nickname_error is None:
                    nickname_error = _register_nickname_error(reg_nickname)
                if nickname_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(nickname_error)}</div>",
                        unsafe_allow_html=True,
                    )

                reg_class = st.text_input(
                    REGISTER_LABELS["class_name"],
                    placeholder="如计算机网络1班",
                    icon=":material/groups:",
                    key="register_class_input",
                )
                class_error = _live_register_class_error(reg_class)
                if submit_attempted and class_error is None:
                    class_error = _register_class_error(reg_class)
                if class_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(class_error)}</div>",
                        unsafe_allow_html=True,
                    )

                reg_email = st.text_input(
                    REGISTER_LABELS["email"],
                    placeholder="请输入邮箱",
                    icon=":material/mail:",
                    key="register_email_input",
                )
                email_error = _live_register_email_error(reg_email)
                if submit_attempted and email_error is None:
                    email_error = _register_email_error(reg_email)
                if email_error:
                    st.markdown(
                        f"<div class='register-field-error'>{html.escape(email_error)}</div>",
                        unsafe_allow_html=True,
                    )

                invalid_labels: List[str] = []
                valid_labels: List[str] = []
                if username_error:
                    invalid_labels.append(REGISTER_LABELS["username"])
                elif (reg_username or "").strip():
                    valid_labels.append(REGISTER_LABELS["username"])
                if password_error:
                    invalid_labels.append(REGISTER_LABELS["password"])
                elif reg_password:
                    valid_labels.append(REGISTER_LABELS["password"])
                if name_error:
                    invalid_labels.append(REGISTER_LABELS["name"])
                elif (reg_name or "").strip():
                    valid_labels.append(REGISTER_LABELS["name"])
                if student_id_error:
                    invalid_labels.append(REGISTER_LABELS["student_id"])
                elif (reg_student_id or "").strip():
                    valid_labels.append(REGISTER_LABELS["student_id"])
                if nickname_error:
                    invalid_labels.append(REGISTER_LABELS["nickname"])
                elif (reg_nickname or "").strip():
                    valid_labels.append(REGISTER_LABELS["nickname"])
                if class_error:
                    invalid_labels.append(REGISTER_LABELS["class_name"])
                elif (reg_class or "").strip():
                    valid_labels.append(REGISTER_LABELS["class_name"])
                if email_error:
                    invalid_labels.append(REGISTER_LABELS["email"])
                elif (reg_email or "").strip():
                    valid_labels.append(REGISTER_LABELS["email"])
                _render_register_field_state_css(
                    invalid_labels=invalid_labels,
                    valid_labels=valid_labels,
                )

                reg_submit = st.button("注册", type="primary", use_container_width=True, key="register_submit_btn")

            if reg_submit:
                submit_errors: Dict[str, str] = {}
                strict_username_error = _register_username_error(reg_username)
                strict_password_error = _register_password_error(reg_password)
                strict_name_error = _register_name_error(reg_name)
                strict_student_id_error = _register_student_id_error(reg_student_id)
                strict_nickname_error = _register_nickname_error(reg_nickname)
                strict_class_error = _register_class_error(reg_class)
                strict_email_error = _register_email_error(reg_email)
                if strict_username_error:
                    submit_errors[REGISTER_LABELS["username"]] = strict_username_error
                if strict_password_error:
                    submit_errors[REGISTER_LABELS["password"]] = strict_password_error
                if strict_name_error:
                    submit_errors[REGISTER_LABELS["name"]] = strict_name_error
                if strict_student_id_error:
                    submit_errors[REGISTER_LABELS["student_id"]] = strict_student_id_error
                if strict_nickname_error:
                    submit_errors[REGISTER_LABELS["nickname"]] = strict_nickname_error
                if strict_class_error:
                    submit_errors[REGISTER_LABELS["class_name"]] = strict_class_error
                if strict_email_error:
                    submit_errors[REGISTER_LABELS["email"]] = strict_email_error

                if submit_errors:
                    st.session_state.register_submit_attempted = True
                    st.rerun()

                st.session_state.register_submit_attempted = False
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
                    user_obj = data.get("user", {}) or {}
                    st.session_state.current_user = user_obj.get("username")
                    st.session_state.current_user_profile = user_obj.get("profile", {}) or {}
                    _load_user_sessions(st.session_state.current_user)
                    st.success("注册成功，已登录")
                    st.rerun()
                except Exception as e:
                    st.error(f"注册失败：{repr(e)}")

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
# 选取 fragment 装饰器（兼容旧版本 streamlit）
_st_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def _render_sidebar_user_card() -> None:
    """侧边栏底部"用户卡片"+ 向上弹出菜单。

    用 @st.fragment 包裹后，卡片 click 与 outside-click 触发的开关只会
    重跑本函数而非整个脚本，主区域内容不抖动。
    """
    profile = st.session_state.get("current_user_profile") or {}
    username = st.session_state.current_user or ""
    class_name = (profile.get("class_name") or "").strip() or "未填写班级"
    initial = (username[:1] or "U").upper()

    with st.container(key="sidebar_user_card"):
        # 1. 隐藏 button：用于接收"点击卡片"事件（被 JS 触发）
        if st.button(
            " ",
            key="sidebar_user_card_toggle_btn",
            type="secondary",
        ):
            st.session_state.show_user_menu = not bool(
                st.session_state.get("show_user_menu", False)
            )

        # 2. 可视卡片（markdown），点击会被 JS 转发到上面的 button
        st.markdown(
            f"""
            <div class="user-card-display">
                <div class="user-card-avatar">{html.escape(initial)}</div>
                <div class="user-card-text">
                    <div class="user-card-name">{html.escape(username)}</div>
                    <div class="user-card-class">{html.escape(class_name)}</div>
                </div>
                <div class="user-card-arrow">⌃</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. 向上弹出的菜单
        if st.session_state.get("show_user_menu", False):
            with st.container(key="sidebar_user_panel", border=True):
                with st.container(key="settings_archived_wrap"):
                    archived_in_panel = _sorted_session_items(
                        st.session_state.sessions, archived=True
                    )
                    with st.expander(f"已归档（{len(archived_in_panel)}）", expanded=False):
                        if not archived_in_panel:
                            st.caption("暂无归档会话")
                        for sid, sess in archived_in_panel:
                            title = sess.get("title", sid)
                            col_t, col_r, col_d = st.columns([0.52, 0.24, 0.24])
                            with col_t:
                                if st.button(
                                    title,
                                    key=f"panel_archived_view_{sid}",
                                    use_container_width=True,
                                    type="secondary",
                                ):
                                    _switch_session(sid)
                                    st.session_state.show_user_menu = False
                                    st.rerun()
                            with col_r:
                                if st.button(
                                    "恢复",
                                    key=f"panel_unarchive_{sid}",
                                    use_container_width=True,
                                    type="tertiary",
                                ):
                                    _unarchive_session(sid)
                                    st.rerun()
                            with col_d:
                                if st.button(
                                    "删除",
                                    key=f"panel_archived_delete_{sid}",
                                    use_container_width=True,
                                    type="tertiary",
                                ):
                                    _delete_session(sid)
                                    st.rerun()

                if st.button(
                    "退出登录",
                    key="top_user_logout",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.current_user = None
                    st.session_state.current_user_profile = {}
                    st.session_state.auth_token = None
                    st.session_state.sessions = {}
                    st.session_state.active_session_id = None
                    st.session_state.show_user_menu = False
                    st.rerun()

        # 4. JS：固定 user card 到 sidebar 底部 + click 转发 + outside-click 关闭
        components.html(
            """
            <script>
            (function(){
                try {
                    const parentWin = window.parent;
                    const parentDoc = parentWin.document;

                    // === 1. 把 user card 用 fixed 对齐 sidebar 底部 ===
                    const positionCard = function() {
                        const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
                        const card = parentDoc.querySelector('.st-key-sidebar_user_card');
                        if (!sidebar || !card) return;
                        const r = sidebar.getBoundingClientRect();
                        // sidebar 折叠时 width 为 0，隐藏卡片
                        if (r.width < 60) {
                            card.style.display = 'none';
                            return;
                        }
                        card.style.display = '';
                        card.style.position = 'fixed';
                        card.style.bottom = '0px';
                        card.style.left = r.left + 'px';
                        card.style.width = r.width + 'px';
                        card.style.zIndex = '999';
                        card.style.boxSizing = 'border-box';
                    };
                    positionCard();

                    // 监听 sidebar resize / window resize
                    if (parentWin.__sidebarUserCardRO) {
                        try { parentWin.__sidebarUserCardRO.disconnect(); } catch(_) {}
                    }
                    const sidebarEl = parentDoc.querySelector('[data-testid="stSidebar"]');
                    if (sidebarEl && parentWin.ResizeObserver) {
                        const ro = new ResizeObserver(positionCard);
                        ro.observe(sidebarEl);
                        ro.observe(parentDoc.body);
                        parentWin.__sidebarUserCardRO = ro;
                    }
                    if (parentWin.__sidebarUserCardWindowResize) {
                        parentWin.removeEventListener(
                            'resize',
                            parentWin.__sidebarUserCardWindowResize
                        );
                    }
                    parentWin.__sidebarUserCardWindowResize = positionCard;
                    parentWin.addEventListener('resize', positionCard);

                    // 偶尔轮询纠偏（处理 sidebar 折叠 transition 等情况）
                    if (parentWin.__sidebarUserCardInterval) {
                        clearInterval(parentWin.__sidebarUserCardInterval);
                    }
                    parentWin.__sidebarUserCardInterval = setInterval(positionCard, 800);

                    // === 2. 卡片 click 转发 + outside-click 关闭 ===
                    if (parentWin.__sidebarUserCardHandler) {
                        parentDoc.removeEventListener(
                            'mousedown',
                            parentWin.__sidebarUserCardHandler,
                            true
                        );
                    }
                    const handler = function(e) {
                        const root = parentDoc.querySelector('.st-key-sidebar_user_card');
                        if (!root) return;
                        const card = root.querySelector('.user-card-display');
                        const panel = root.querySelector('.st-key-sidebar_user_panel');
                        const toggleBtn = root.querySelector(
                            '.st-key-sidebar_user_card_toggle_btn button'
                        );
                        if (!toggleBtn) return;

                        if (panel && panel.contains(e.target)) return;

                        if (card && card.contains(e.target)) {
                            e.preventDefault();
                            e.stopPropagation();
                            toggleBtn.click();
                            return;
                        }

                        if (!root.contains(e.target) && panel) {
                            toggleBtn.click();
                        }
                    };
                    parentWin.__sidebarUserCardHandler = handler;
                    parentDoc.addEventListener('mousedown', handler, true);
                } catch (err) {
                    console.warn('[user card] init failed:', err);
                }
            })();
            </script>
            """,
            height=0,
        )


if _st_fragment is not None:
    _render_sidebar_user_card = _st_fragment(_render_sidebar_user_card)

with st.sidebar:
    sidebar_logo_html = _logo_html("sidebar-brand-badge")
    with st.container(key="sidebar_top_bar"):
        st.markdown(
            f"""
            <div class="sidebar-brand">
                {sidebar_logo_html}
                <div class="sidebar-brand-title">NetRUC Agent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button(
        "新建会话",
        key="sidebar_new_session_btn",
        icon=":material/edit_square:",
        type="secondary",
        use_container_width=True,
    ):
        _create_new_session()
        st.rerun()

    with st.container(key="sidebar_session_list"):
        st.markdown("---")
        st.markdown('<div class="sb-section-title">◌ 会话列表</div>', unsafe_allow_html=True)

        unarchived_sessions = _sorted_session_items(st.session_state.sessions, archived=False)

        for sid, sess in unarchived_sessions:
            title = sess.get("title", sid)
            is_active = (sid == st.session_state.active_session_id)
            row_class = "session-row-active" if is_active else "session-row"
            with st.container(key=f"session_row_{sid}"):
                # 注入 marker class 让 CSS 区分活跃 / 非活跃
                st.markdown(
                    f'<div class="{row_class}-marker" style="display:none"></div>',
                    unsafe_allow_html=True,
                )
                col_t, col_m = st.columns([0.84, 0.16])
                with col_t:
                    if st.button(
                        title,
                        key=f"switch_{sid}",
                        use_container_width=True,
                        type="tertiary",
                    ):
                        _switch_session(sid)
                        st.rerun()
                with col_m:
                    with st.popover(
                        " ",
                        icon=":material/more_horiz:",
                        use_container_width=True,
                    ):
                        if st.button(
                            "归档",
                            key=f"archive_pop_{sid}",
                            icon=":material/archive:",
                            use_container_width=True,
                            type="tertiary",
                        ):
                            _archive_session(sid)
                            st.rerun()
                        if st.button(
                            "删除",
                            key=f"delete_pop_{sid}",
                            icon=":material/delete:",
                            use_container_width=True,
                            type="tertiary",
                        ):
                            _delete_session(sid)
                            st.rerun()

    _render_sidebar_user_card()


# -----------------------------
# Main: render current session messages
# -----------------------------
active_id = st.session_state.active_session_id
active_session = st.session_state.sessions.get(active_id, {"title": "新会话", "messages": []})
active_messages = active_session.get("messages", [])
_input_v = st.session_state.get("_input_v", 0)
raw_input = None
input_widget_rendered = False

session_title = str(active_session.get("title", active_id))
if active_messages:
    st.caption(f"当前会话：{session_title}")
else:
    empty_logo_html = _logo_html("empty-hero-logo")
    st.markdown(
        f"""
        <div class="empty-hero">
            {empty_logo_html}
            <h2>NetRUC Agent</h2>
            <div class="empty-hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    empty_left, empty_center, empty_right = st.columns([0.6, 4.8, 0.6])
    with empty_center:
        _render_websearch_chip("empty")
        raw_input = chat_input_images(key=f"chat_img_input_{_input_v}")
        input_widget_rendered = True

for idx, m in enumerate(active_messages):
    role = m.get("role", "assistant")
    content = m.get("content", "")
    thinking = m.get("thinking")
    tool_traces = m.get("tool_traces", None)
    message_id = m.get("message_id")
    current_feedback = m.get("feedback")

    with st.chat_message("user" if role == "user" else "assistant"):
        # 显示用户消息中附带的图片
        if role == "user" and m.get("image_b64"):
            for img_data in m["image_b64"]:
                st.image(base64.b64decode(img_data), width=300)
        if role == "assistant":
            _render_assistant_content(content, thinking=thinking)
        else:
            st.markdown(content)

        if role == "assistant" and message_id and active_id:
            supports_icon = "icon" in inspect.signature(st.button).parameters
            col_like, col_dislike, _ = st.columns([1, 1, 12], gap="small")
            with col_like:
                like_clicked = False
                if supports_icon:
                    like_clicked = st.button(
                        " ",
                        key=f"fb_like_{active_id}_{message_id}_{idx}",
                        icon=":material/thumb_up:" if current_feedback == "like" else ":material/thumb_up_off_alt:",
                        type="primary" if current_feedback == "like" else "tertiary",
                    )
                else:
                    like_label = "👍🏻" if current_feedback == "like" else "👍︎"
                    like_clicked = st.button(
                        like_label,
                        key=f"fb_like_{active_id}_{message_id}_{idx}",
                        type="primary" if current_feedback == "like" else "tertiary",
                    )

                if like_clicked:
                    next_feedback = "cancel" if current_feedback == "like" else "like"
                    try:
                        submit_feedback(
                            token=st.session_state.auth_token or "",
                            session_id=active_id,
                            message_id=message_id,
                            feedback=next_feedback,
                        )
                        m["feedback"] = None if next_feedback == "cancel" else "like"
                        _persist_current_user_sessions()
                        st.rerun()
                    except Exception as e:
                        st.error(f"反馈提交失败：{repr(e)}")
            with col_dislike:
                dislike_clicked = False
                if supports_icon:
                    dislike_clicked = st.button(
                        " ",
                        key=f"fb_dislike_{active_id}_{message_id}_{idx}",
                        icon=":material/thumb_down:" if current_feedback == "dislike" else ":material/thumb_down_off_alt:",
                        type="primary" if current_feedback == "dislike" else "tertiary",
                    )
                else:
                    dislike_label = "👎🏻" if current_feedback == "dislike" else "👎︎"
                    dislike_clicked = st.button(
                        dislike_label,
                        key=f"fb_dislike_{active_id}_{message_id}_{idx}",
                        type="primary" if current_feedback == "dislike" else "tertiary",
                    )

                if dislike_clicked:
                    next_feedback = "cancel" if current_feedback == "dislike" else "dislike"
                    try:
                        submit_feedback(
                            token=st.session_state.auth_token or "",
                            session_id=active_id,
                            message_id=message_id,
                            feedback=next_feedback,
                        )
                        m["feedback"] = None if next_feedback == "cancel" else "dislike"
                        _persist_current_user_sessions()
                        st.rerun()
                    except Exception as e:
                        st.error(f"反馈提交失败：{repr(e)}")


# -----------------------------
# Input + send (custom chat input with paste/upload/thumbnails)
# -----------------------------

user_text = ""
if not input_widget_rendered:
    # 非空会话保持底部固定输入；空会话已在主区中部渲染
    bottom_container = getattr(st, "bottom", None) or getattr(st, "_bottom", None)
    if bottom_container is None:
        _render_websearch_chip("bottom_inline")
        raw_input = chat_input_images(key=f"chat_img_input_{_input_v}")
    else:
        with bottom_container:
            _render_websearch_chip("bottom")
            raw_input = chat_input_images(key=f"chat_img_input_{_input_v}")

if raw_input:
    # 解析组件返回的 JSON: {text: "...", images: [{base64, mime}, ...]}
    input_data = json.loads(raw_input)
    user_text = (input_data.get("text") or "").strip()
    input_images = input_data.get("images") or []

    # OCR 提取图片文字
    image_descriptions: List[str] = []
    image_b64_list: List[str] = []

    if input_images:
        with st.spinner("正在识别图片内容…"):
            for i, img_info in enumerate(input_images):
                try:
                    img_bytes = base64.b64decode(img_info["base64"])
                    image_b64_list.append(img_info["base64"])
                    desc = describe_image(img_bytes, filename=f"image_{i}.png")
                    image_descriptions.append(f"[图片 {i+1}]: {desc}")
                except Exception as e:
                    image_descriptions.append(f"[图片 {i+1}]: (识别失败: {repr(e)})")

    # 拼接最终消息
    if image_descriptions:
        desc_block = "\n".join(image_descriptions)
        if user_text:
            combined_message = f"以下是用户上传的图片内容：\n{desc_block}\n\n用户的问题：{user_text}"
        else:
            combined_message = f"以下是用户上传的图片内容：\n{desc_block}\n\n请根据图片内容回答。"
    else:
        combined_message = user_text

    if not combined_message.strip():
        st.stop()

    # 重置输入组件（清除已提交的图片和文字）
    st.session_state["_input_v"] = _input_v + 1

    if not st.session_state.active_session_id:
        _create_new_session()
        active_id = st.session_state.active_session_id

    active_id = st.session_state.active_session_id
    sess = st.session_state.sessions[active_id]

    if sess.get("title") in (None, "", "新会话"):
        sess["title"] = _default_title_from_message(user_text or "图片问题")

    user_msg_record = {"role": "user", "content": combined_message}
    if image_b64_list:
        user_msg_record["image_b64"] = image_b64_list
    sess["messages"].append(user_msg_record)
    _touch_session(active_id)
    _persist_current_user_sessions()

    with st.chat_message("user"):
        if image_b64_list:
            for img_data in image_b64_list:
                st.image(base64.b64decode(img_data), width=300)
        st.markdown(user_text or "(仅上传了图片)")

    payload_session_id = None if active_id.startswith("local_") else active_id

    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("思考中…")

            data = None
            assistant_text = ""
            assistant_thinking = ""
            tool_traces = []
            response_message_id = None

            buf = ""
            stream_kwargs = {
                "token": st.session_state.auth_token or "",
                "message": combined_message,
                "session_id": payload_session_id,
                "history": None,
                "max_turns": 5,
            }
            try:
                supports_websearch_toggle = "enable_websearch" in inspect.signature(chat_once_stream).parameters
            except (TypeError, ValueError):
                supports_websearch_toggle = False
            if supports_websearch_toggle:
                stream_kwargs["enable_websearch"] = st.session_state.enable_websearch

            for event in chat_once_stream(**stream_kwargs):
                if event["type"] == "meta":
                    response_message_id = event.get("message_id")
                    placeholder.markdown("正在检索与生成…")
                elif event["type"] == "token":
                    buf += event["content"]
                    parsed_stream = split_assistant_content(buf)
                    if parsed_stream["visible"]:
                        placeholder.markdown(parsed_stream["visible"])
                    elif parsed_stream["has_thinking"] or parsed_stream["in_thinking"]:
                        placeholder.markdown("思考中…")
                    else:
                        placeholder.markdown(buf)
                elif event["type"] == "done":
                    data = event
                    tool_traces = event.get("tool_traces", []) or []
                    final_reply = event.get("reply", "")
                    assistant_thinking = event.get("thinking", "") or ""
                    if not final_reply and buf:
                        parsed_final = split_assistant_content(buf)
                        final_reply = parsed_final["visible"]
                        assistant_thinking = assistant_thinking or parsed_final["thinking"]
                    if final_reply:
                        placeholder.markdown(final_reply)
                    elif assistant_thinking:
                        placeholder.empty()
                    assistant_text = final_reply or ""

            if assistant_thinking:
                with st.expander("Thinking", expanded=False):
                    st.markdown(assistant_thinking)

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
        "thinking": assistant_thinking,
        "tool_traces": tool_traces,
        "message_id": response_message_id,
        "feedback": None,
    })
    _touch_session(active_id)
    _persist_current_user_sessions()

    st.rerun()
