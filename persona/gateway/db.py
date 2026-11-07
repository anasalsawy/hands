"""SQLite persistence for the gateway (home layer).

Owns the *app* side: who exists, which persona key belongs to them. The
mind's own data (memory, journal, verdicts) lives in sawii-persona's own
store modules; the gateway store never touches those tables.

Schema is deliberately small for the A1 landing — just users and sessions.
Later issues extend it:
    A3 (#20) adds files + workspace quotas
    (threads come later; the companion-app's thread model may change)

Single-writer SQLite under a lock, same pattern as the companion-app's db.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL DEFAULT '',
  pass_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'user',
  created_at REAL NOT NULL,
  last_login_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token_hash TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  revoke_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- One persona per user. upstream_id is kept for future compat with the
-- companion-app upstream flow when A2 lands.
CREATE TABLE IF NOT EXISTS personas (
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  persona_key TEXT NOT NULL,
  upstream_id INTEGER,
  provisioned_at REAL NOT NULL,
  source TEXT NOT NULL DEFAULT 'self'
);
"""


def now() -> float:
    return time.time()


class DB:
    """Thin SQLite wrapper. Single connection, serialized via a lock."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def write(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.lastrowid or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_db: DB | None = None
_db_lock = threading.Lock()


def get_db(path: str = "") -> DB:
    """Lazily initialize the module-level DB.

    Path precedence: explicit arg > SAWII_GATEWAY_DB env > default under
    the SAWII_DATA_DIR / data / gateway.sqlite3.
    """
    global _db
    with _db_lock:
        if _db is None:
            resolved = path or os.environ.get("SAWII_GATEWAY_DB", "")
            if not resolved:
                root = os.environ.get("SAWII_DATA_DIR", "./data")
                resolved = os.path.join(root, "gateway.sqlite3")
            _db = DB(resolved)
        return _db


def reset_db_for_tests(path: str = ":memory:") -> DB:
    """Reset the module-level DB. Tests use this to get a fresh in-memory DB."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = DB(path)
        return _db


def set_db_for_tests(db: DB) -> None:
    """Inject a specific DB (used by tests that spin up their own)."""
    global _db
    with _db_lock:
        _db = db
