"""Gateway — the home layer around Sawii's mind.

Per README.md, the gateway holds everything the companion
app needs that isn't the mind itself: accounts, sessions, workspace, plugin
catalogue, user-facing routes. The mind (/v1/*) is separate; the gateway
sits around it under /api/*.

What's here as of 2026-10-21:
    auth.py        — accounts + bearer sessions (A1)
    personas.py    — per-user persona minting on register (A2)
    middleware.py  — request-id, structured JSON logs
    db.py          — SQLite users + sessions + personas tables
    routes.py      — /api/auth/* + /api/persona endpoints
    browser/       — Shape B wrap of browser-use (ships in #27)

What's coming (per README.md):
    A3 (#20) — workspace (per-user sandboxed files)
    A4 (#21) — plugin catalogue
"""
from .auth import AuthError, require_user, user_for_token
from .db import DB, get_db
from .middleware import RequestIdMiddleware
from .personas import get_persona_for_user, provision_persona_for_user
from .routes import register_gateway_routes

__all__ = [
    "AuthError",
    "DB",
    "RequestIdMiddleware",
    "get_db",
    "get_persona_for_user",
    "provision_persona_for_user",
    "register_gateway_routes",
    "require_user",
    "user_for_token",
]
