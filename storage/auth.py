from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

USERNAME_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")
PASSWORD_RE = re.compile(r"^[A-Za-z0-9]{8,}$")


def validate_username(username: str) -> bool:
    return bool(USERNAME_RE.fullmatch(username or ""))


def validate_password(password: str) -> bool:
    if not PASSWORD_RE.fullmatch(password or ""):
        return False
    has_alpha = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_alpha and has_digit


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt_bytes = bytes.fromhex(salt) if salt else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return salt_bytes.hex(), digest.hex()


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, digest = hash_password(password, salt=salt)
    return secrets.compare_digest(digest, password_hash)


def create_token() -> str:
    return secrets.token_urlsafe(32)


def token_expires_at(days: int = 30) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def is_token_expired(expires_at: str) -> bool:
    try:
        expires = datetime.fromisoformat(expires_at)
    except ValueError:
        return True
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires

