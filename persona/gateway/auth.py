"""Accounts and bearer sessions.

Ported from companion-app/app/auth.py (per README.md
Phase 1 A1). Design unchanged:

- bcrypt-hashed passwords (cost 12)
- opaque random bearer tokens; only a SHA-256 hash is stored, so a DB leak
  doesn't hand over live credentials
- a login against a missing account burns a bcrypt round so timing can't
  reveal whether an email exists

The companion-app's auth module used a module-level settings import; this
port reads the same values from env vars directly (SAWII_SESSION_TTL,
SAWII_RATE_LIMIT_PER_MINUTE, SAWII_ALLOW_REGISTRATION), so the gateway
module is self-contained and testable.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from collections import defaultdict, deque

try:
    import bcrypt  # type: ignore
except ImportError:  # pragma: no cover
    bcrypt = None  # type: ignore[assignment]

from .db import get_db, now

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MIN_PASSWORD = 10


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SESSION_TTL = _env_int("SAWII_SESSION_TTL", 60 * 60 * 24 * 30)  # 30 days
RATE_LIMIT_PER_MINUTE = _env_int("SAWII_RATE_LIMIT_PER_MINUTE", 30)
ALLOW_REGISTRATION = _env_bool("SAWII_ALLOW_REGISTRATION", True)


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


# ──────────────────────────────────────────────────────────────────────────
# Password handling


def _require_bcrypt() -> None:
    if bcrypt is None:
        raise RuntimeError(
            "bcrypt is required for gateway auth. "
            "Install sawii-persona[gateway] or add `bcrypt` to the environment."
        )


def hash_password(password: str) -> str:
    _require_bcrypt()
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    _require_bcrypt()
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def constant_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")


# ──────────────────────────────────────────────────────────────────────────
# Credential validation


def validate_credentials(email: str, password: str) -> tuple[str, str]:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise AuthError("Enter a valid email address.")
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
    return email, password


# ──────────────────────────────────────────────────────────────────────────
# Session lifecycle


def create_session(user_id: int) -> tuple[str, float]:
    token = secrets.token_urlsafe(32)
    expires = now() + SESSION_TTL
    get_db().write(
        "INSERT INTO sessions (user_id, token_hash, created_at, expires_at) VALUES (?,?,?,?)",
        (user_id, _hash_token(token), now(), expires),
    )
    return token, expires


def user_for_token(token: str) -> dict | None:
    if not token:
        return None
    row = get_db().one(
        """SELECT u.*, s.expires_at, s.token_hash FROM sessions s
           JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = ? AND s.revoke_reason IS NULL""",
        (_hash_token(token),),
    )
    if not row:
        return None
    if row["expires_at"] < now():
        revoke_token(token, "expired")
        return None
    return row


def revoke_token(token: str, reason: str = "logout") -> None:
    get_db().write(
        "UPDATE sessions SET revoke_reason=? WHERE token_hash=? AND revoke_reason IS NULL",
        (reason, _hash_token(token)),
    )


def revoke_all(user_id: int, reason: str = "password change") -> None:
    get_db().write(
        "UPDATE sessions SET revoke_reason=? WHERE user_id=? AND revoke_reason IS NULL",
        (reason, user_id),
    )


# ──────────────────────────────────────────────────────────────────────────
# Register / login


def register(email: str, password: str, name: str = "") -> dict:
    email, password = validate_credentials(email, password)
    if not ALLOW_REGISTRATION:
        raise AuthError("Registration is closed.", 403)
    db = get_db()
    if db.one("SELECT id FROM users WHERE email=?", (email,)):
        raise AuthError("That email already has an account. Try signing in.", 409)
    uid = db.write(
        "INSERT INTO users (email, name, pass_hash, created_at) VALUES (?,?,?,?)",
        (email, (name or "")[:80], hash_password(password), now()),
    )
    return {"id": uid, "email": email, "name": name}


def login(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    db = get_db()
    row = db.one("SELECT * FROM users WHERE email=?", (email,))
    # Always burn a bcrypt round so a missing account and a wrong password
    # take the same wall-clock time and cannot be told apart.
    target = row["pass_hash"] if row else hash_password(secrets.token_urlsafe(12))
    ok = verify_password(password or "", target)
    if not row or not ok:
        raise AuthError("Email or password is wrong.", 401)
    db.write("UPDATE users SET last_login_at=? WHERE id=?", (now(), row["id"]))
    return row


# ──────────────────────────────────────────────────────────────────────────
# Rate limiting (sliding window, in-memory)


_buckets: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(bucket: str, limit: int | None = None, window: float = 60.0) -> bool:
    """Sliding-window limiter. Returns False when the caller is over budget."""
    limit = limit or RATE_LIMIT_PER_MINUTE
    t = time.monotonic()
    q = _buckets[bucket]
    while q and t - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(t)
    return True


def reset_rate_limits_for_tests() -> None:
    _buckets.clear()


# ──────────────────────────────────────────────────────────────────────────
# Request helpers


def bearer_from_request(request) -> str:
    """Pull the bearer token from a Starlette Request's Authorization header."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def require_user(request) -> dict:
    """Resolve the current user from the request, or raise AuthError(401)."""
    user = user_for_token(bearer_from_request(request))
    if not user:
        raise AuthError("Sign in to continue.", 401)
    return user
