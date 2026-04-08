# app_streamlit.py
import base64
import json
import uuid
import os
import time
import inspect
from datetime import datetime
import streamlit as st
from typing import Dict, Any, Optional, Iterable, Tuple, List
from dotenv import load_dotenv

load_dotenv()

# ✅ 直接本地调用 server.py（不走 HTTP）
from server import (
    chat_once,
    chat_once_stream,
    submit_feedback,
    register_user,
    login_user,
    load_user_session_cache,
    RegisterRequest,
    LoginRequest,
)
from agentic_rag.chat_format import split_assistant_content
from agentic_rag.vision import describe_image
from components.chat_input_images import chat_input_images

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
    import copy
    data = copy.deepcopy(st.session_state.persisted_data)
    # 剥离 image_b64 避免 JSON 文件膨胀
    for _uname, udata in data.get("users", {}).items():
        for _sid, sess in udata.get("sessions", {}).items():
            for m in sess.get("messages", []):
                m.pop("image_b64", None)
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
    local_sessions = _normalize_sessions(store.get("sessions", {}))
    backend_sessions: Dict[str, Any] = {}
    if st.session_state.get("auth_token"):
        try:
            backend_sessions = load_user_session_cache(token=st.session_state.auth_token).get("sessions", {})
        except Exception:
            backend_sessions = {}
    st.session_state.sessions = _merge_backend_sessions(local_sessions, backend_sessions)
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


def _set_session_archived(session_id: str, archived: bool) -> None:
    sess = st.session_state.sessions.get(session_id)
    if not sess:
        return
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
    # ✅ B方案：不再请求后端删除（因为没有后端 HTTP）
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]

    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = _pick_session_id(archived=False)

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
st.markdown("<h1 style='text-align:center;'>NetRUC Agent</h1>", unsafe_allow_html=True)

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

if "enable_websearch" not in st.session_state:
    st.session_state.enable_websearch = True

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
    st.session_state.enable_websearch = st.toggle("启用联网搜索", value=st.session_state.enable_websearch)

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

    unarchived_sessions = _sorted_session_items(st.session_state.sessions, archived=False)
    archived_sessions = _sorted_session_items(st.session_state.sessions, archived=True)

    if st.session_state.manage_mode and unarchived_sessions:
        options = [f"{s.get('title', sid)} ({sid})" for sid, s in unarchived_sessions]
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

    for sid, sess in unarchived_sessions:
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
            options = [f"{s.get('title', sid)} ({sid})" for sid, s in archived_sessions]
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

        for sid, sess in archived_sessions:
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

        if st.session_state.debug and role == "assistant" and tool_traces:
            with st.expander("工具调用详情", expanded=False):
                for t in tool_traces:
                    st.markdown(f"**Tool**: {t.get('tool')}")
                    st.code(t.get("input", ""))
                    st.text_area("output", t.get("output", ""), height=150)
                    st.divider()


# -----------------------------
# Input + send (custom chat input with paste/upload/thumbnails)
# -----------------------------

# 渲染自定义输入组件到底部固定栏（类似 ChatGPT）
_input_v = st.session_state.get("_input_v", 0)
with st._bottom:
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

    history_payload = None
    if st.session_state.sync_history:
        previous_messages = sess.get("messages", [])[:-1]
        history_payload = [
            normalized
            for normalized in (_message_for_history(message) for message in previous_messages)
            if normalized is not None
        ]

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
                "history": history_payload,
                "debug": st.session_state.debug,
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

    # 额外展示（避免上面的 streaming placeholder 被 rerun 覆盖）
    with st.chat_message("assistant"):
        _render_assistant_content(assistant_text, thinking=assistant_thinking)
        if st.session_state.debug and tool_traces:
            with st.expander("工具调用详情", expanded=False):
                for t in tool_traces:
                    st.markdown(f"**Tool**: {t.get('tool')}")
                    st.code(t.get("input", ""))
                    st.text_area("output", t.get("output", ""), height=150)
                    st.divider()

    st.rerun()
