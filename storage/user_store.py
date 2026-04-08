from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .auth import create_token, is_token_expired, token_expires_at, verify_password

_LOCK = threading.Lock()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data_store"
DB_FILE = DATA_DIR / "app.db"
USERS_FILE = DATA_DIR / "users.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"
LOGS_FILE = DATA_DIR / "logs.json"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _init_db() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    preferences_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                );
                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    history_json TEXT NOT NULL DEFAULT '[]',
                    last_turns_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, session_id),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    detail TEXT,
                    ts TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL CHECK(feedback_type IN ('like', 'dislike')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, session_id, message_id),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS interaction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    question_category TEXT NOT NULL,
                    hint_level_start INTEGER NOT NULL,
                    hint_level_end INTEGER NOT NULL,
                    hint_decision TEXT NOT NULL,
                    was_failsafe INTEGER NOT NULL DEFAULT 0,
                    relevance INTEGER NOT NULL DEFAULT 1,
                    tool_count INTEGER NOT NULL DEFAULT 0,
                    response_length INTEGER NOT NULL DEFAULT 0,
                    ts TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_im_user_ts
                    ON interaction_metrics(user_id, ts);
                CREATE TABLE IF NOT EXISTS proficiency_scores (
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    interaction_count INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL,
                    PRIMARY KEY(user_id, category),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            _ensure_column(conn, "sessions", "history_json", "history_json TEXT NOT NULL DEFAULT '[]'")
            conn.execute(
                """
                UPDATE sessions
                SET history_json = COALESCE(NULLIF(last_turns_json, ''), '[]')
                WHERE history_json IS NULL OR history_json = '' OR history_json = '[]'
                """
            )
            conn.commit()
        finally:
            conn.close()

    _maybe_migrate_from_json()


def _maybe_migrate_from_json() -> None:
    if not USERS_FILE.exists() and not SESSIONS_FILE.exists() and not LOGS_FILE.exists():
        return
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute("SELECT COUNT(1) AS cnt FROM users")
            if cur.fetchone()["cnt"] > 0:
                return
            _migrate_users(conn)
            _migrate_sessions(conn)
            _migrate_logs(conn)
            conn.commit()
        finally:
            conn.close()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _migrate_users(conn: sqlite3.Connection) -> None:
    data = _read_json(USERS_FILE, {"users": []})
    for user in data.get("users", []):
        profile = user.get("profile", {})
        preferences = user.get("preferences", {})
        conn.execute(
            """
            INSERT OR IGNORE INTO users (
                id, username, password_salt, password_hash, profile_json,
                preferences_json, created_at, last_login_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.get("id"),
                user.get("username"),
                user.get("password_salt"),
                user.get("password_hash"),
                json.dumps(profile, ensure_ascii=False),
                json.dumps(preferences, ensure_ascii=False),
                user.get("created_at") or _utc_now(),
                user.get("last_login_at"),
            ),
        )
        for token in user.get("tokens", []):
            conn.execute(
                "INSERT OR IGNORE INTO tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
                (user.get("id"), token.get("token"), token.get("expires_at")),
            )


def _migrate_sessions(conn: sqlite3.Connection) -> None:
    data = _read_json(SESSIONS_FILE, {"users": {}})
    users = data.get("users", {})
    for user_id, user_sessions in users.items():
        sessions = user_sessions.get("sessions", {})
        if not isinstance(sessions, dict):
            continue
        for session_id, sess in sessions.items():
            full_history = sess.get("messages") or sess.get("history") or sess.get("last_turns", [])
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (user_id, session_id, summary, history_json, last_turns_json, state_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    sess.get("summary", ""),
                    json.dumps(full_history, ensure_ascii=False),
                    json.dumps(sess.get("last_turns", []), ensure_ascii=False),
                    json.dumps(sess.get("state", {}), ensure_ascii=False),
                    _utc_now(),
                ),
            )


def _migrate_logs(conn: sqlite3.Connection) -> None:
    data = _read_json(LOGS_FILE, {"events": []})
    for event in data.get("events", []):
        conn.execute(
            "INSERT INTO logs (user_id, type, detail, ts) VALUES (?, ?, ?, ?)",
            (
                event.get("user_id"),
                event.get("type"),
                event.get("detail"),
                event.get("ts") or _utc_now(),
            ),
        )


def load_users() -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM users")
        users = []
        for row in cur.fetchall():
            users.append(_row_to_user_dict(conn, row, include_tokens=True))
        return {"users": users}
    finally:
        conn.close()


def save_users(data: Dict[str, Any]) -> None:
    users = data.get("users", [])
    for user in users:
        create_user(user)


def _row_to_user_dict(conn: sqlite3.Connection, row: sqlite3.Row, include_tokens: bool) -> Dict[str, Any]:
    profile = json.loads(row["profile_json"] or "{}")
    preferences = json.loads(row["preferences_json"] or "{}")
    user = {
        "id": row["id"],
        "username": row["username"],
        "password_salt": row["password_salt"],
        "password_hash": row["password_hash"],
        "profile": profile,
        "preferences": preferences,
        "tokens": [],
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }
    if include_tokens:
        tokens = conn.execute(
            "SELECT token, expires_at FROM tokens WHERE user_id = ?",
            (row["id"],),
        ).fetchall()
        user["tokens"] = [{"token": t["token"], "expires_at": t["expires_at"]} for t in tokens]
    return user


def find_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        return _row_to_user_dict(conn, row, include_tokens=True)
    finally:
        conn.close()


def find_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return _row_to_user_dict(conn, row, include_tokens=True)
    finally:
        conn.close()


def update_user(updated: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE users
            SET username = ?, password_salt = ?, password_hash = ?,
                profile_json = ?, preferences_json = ?, last_login_at = ?
            WHERE id = ?
            """,
            (
                updated.get("username"),
                updated.get("password_salt"),
                updated.get("password_hash"),
                json.dumps(updated.get("profile", {}), ensure_ascii=False),
                json.dumps(updated.get("preferences", {}), ensure_ascii=False),
                updated.get("last_login_at"),
                updated.get("id"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def create_user(user: Dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO users (
                id, username, password_salt, password_hash, profile_json,
                preferences_json, created_at, last_login_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user.get("id"),
                user.get("username"),
                user.get("password_salt"),
                user.get("password_hash"),
                json.dumps(user.get("profile", {}), ensure_ascii=False),
                json.dumps(user.get("preferences", {}), ensure_ascii=False),
                user.get("created_at") or _utc_now(),
                user.get("last_login_at"),
            ),
        )
        for token in user.get("tokens", []):
            conn.execute(
                "INSERT OR IGNORE INTO tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
                (user.get("id"), token.get("token"), token.get("expires_at")),
            )
        conn.commit()
    finally:
        conn.close()


def issue_token_for_user(user: Dict[str, Any], days: int = 30) -> str:
    token = create_token()
    expires_at = token_expires_at(days=days)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user.get("id"), token, expires_at),
        )
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (_utc_now(), user.get("id")),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def authenticate_user(username: str, password: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    user = find_user_by_username(username)
    if not user:
        return None, False
    ok = verify_password(password, user.get("password_salt", ""), user.get("password_hash", ""))
    return user, ok


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM tokens WHERE expires_at < ?",
            (_utc_now(),),
        )
        row = conn.execute(
            """
            SELECT u.* FROM users u
            JOIN tokens t ON t.user_id = u.id
            WHERE t.token = ?
            """,
            (token,),
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.commit()
        return _row_to_user_dict(conn, row, include_tokens=False)
    finally:
        conn.close()


def get_session(user_id: str, session_id: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT summary, history_json, last_turns_json, state_json, updated_at
            FROM sessions
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ).fetchone()
        if not row:
            return {"summary": "", "history": [], "last_turns": [], "state": {}, "updated_at": None}
        return {
            "summary": row["summary"] or "",
            "history": json.loads(row["history_json"] or "[]"),
            "last_turns": json.loads(row["last_turns_json"] or "[]"),
            "state": json.loads(row["state_json"] or "{}"),
            "updated_at": row["updated_at"],
        }
    finally:
        conn.close()


def find_session(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT summary, history_json, last_turns_json, state_json, updated_at
            FROM sessions
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ).fetchone()
        if not row:
            return None
        return {
            "summary": row["summary"] or "",
            "history": json.loads(row["history_json"] or "[]"),
            "last_turns": json.loads(row["last_turns_json"] or "[]"),
            "state": json.loads(row["state_json"] or "{}"),
            "updated_at": row["updated_at"],
        }
    finally:
        conn.close()


def update_session(
    user_id: str,
    session_id: str,
    summary: str,
    last_turns: list,
    state: dict,
    history: Optional[list] = None,
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO sessions (user_id, session_id, summary, history_json, last_turns_json, state_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, session_id) DO UPDATE SET
                summary=excluded.summary,
                history_json=excluded.history_json,
                last_turns_json=excluded.last_turns_json,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                session_id,
                summary or "",
                json.dumps(history or [], ensure_ascii=False),
                json.dumps(last_turns or [], ensure_ascii=False),
                json.dumps(state or {}, ensure_ascii=False),
                _utc_now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_session(user_id: str, session_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_user_sessions(user_id: str) -> list:
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [row["session_id"] for row in cur.fetchall()]
    finally:
        conn.close()


def list_user_session_snapshots(user_id: str) -> list:
    conn = _connect()
    try:
        cur = conn.execute(
            """
            SELECT session_id, summary, history_json, last_turns_json, state_json, updated_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        snapshots = []
        for row in cur.fetchall():
            history = json.loads(row["history_json"] or "[]")
            last_turns = json.loads(row["last_turns_json"] or "[]")
            snapshots.append(
                {
                    "session_id": row["session_id"],
                    "summary": row["summary"] or "",
                    "history": history,
                    "last_turns": last_turns,
                    "state": json.loads(row["state_json"] or "{}"),
                    "updated_at": row["updated_at"],
                }
            )
        return snapshots
    finally:
        conn.close()


def append_log(user_id: str, event_type: str, detail: Optional[str] = None) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO logs (user_id, type, detail, ts) VALUES (?, ?, ?, ?)",
            (user_id, event_type, detail, _utc_now()),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_message_feedback(user_id: str, session_id: str, message_id: str, feedback_type: str) -> None:
    if feedback_type not in {"like", "dislike"}:
        raise ValueError("feedback_type must be 'like' or 'dislike'")
    conn = _connect()
    now = _utc_now()
    try:
        conn.execute(
            """
            INSERT INTO message_feedback (
                user_id, session_id, message_id, feedback_type, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, session_id, message_id) DO UPDATE SET
                feedback_type=excluded.feedback_type,
                updated_at=excluded.updated_at
            """,
            (user_id, session_id, message_id, feedback_type, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def delete_message_feedback(user_id: str, session_id: str, message_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM message_feedback WHERE user_id = ? AND session_id = ? AND message_id = ?",
            (user_id, session_id, message_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_message_feedback(user_id: str, session_id: str, message_id: str) -> Optional[str]:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT feedback_type
            FROM message_feedback
            WHERE user_id = ? AND session_id = ? AND message_id = ?
            """,
            (user_id, session_id, message_id),
        ).fetchone()
        if not row:
            return None
        return row["feedback_type"]
    finally:
        conn.close()


def record_interaction_metric(
    user_id: str,
    session_id: str,
    state: Dict[str, Any],
    tool_traces: list,
    response_length: int,
) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO interaction_metrics
            (user_id, session_id, turn_index, question_category,
             hint_level_start, hint_level_end, hint_decision, was_failsafe,
             relevance, tool_count, response_length, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                session_id,
                state.get("user_turn_count", 1),
                state.get("question_category", "UNKNOWN"),
                state.get("_hint_level_start", 0),
                state.get("hint_level", 0),
                state.get("_hint_decision", "MAINTAIN"),
                int(state.get("_was_failsafe", False)),
                1,
                len(tool_traces),
                response_length,
                _utc_now(),
            ),
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def upsert_proficiency_score(
    user_id: str,
    category: str,
    score: float,
    confidence: float,
    interaction_count: int,
) -> None:
    conn = _connect()
    now = _utc_now()
    try:
        conn.execute(
            """
            INSERT INTO proficiency_scores
            (user_id, category, score, confidence, interaction_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, category) DO UPDATE SET
                score=excluded.score,
                confidence=excluded.confidence,
                interaction_count=excluded.interaction_count,
                last_updated=excluded.last_updated
            """,
            (user_id, category, score, confidence, interaction_count, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_proficiency_scores(user_id: str) -> Dict[str, Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT category, score, confidence, interaction_count, last_updated "
            "FROM proficiency_scores WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return {
            row["category"]: {
                "score": row["score"],
                "confidence": row["confidence"],
                "interaction_count": row["interaction_count"],
                "last_updated": row["last_updated"],
            }
            for row in rows
        }
    finally:
        conn.close()


_init_db()

