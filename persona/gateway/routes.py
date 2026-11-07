"""Gateway routes — the /api/* surface the companion app's UI talks to.

Per README.md, the gateway holds the *home* layer (accounts,
workspace, plugins), distinct from the mind (/v1/*). This module registers
the routes onto a Starlette app. A1 lands auth; A3 and A4 will extend it.
"""
from __future__ import annotations

import logging

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .auth import (
    AuthError,
    bearer_from_request,
    create_session,
    login,
    rate_limit,
    register,
    require_user,
    revoke_token,
)
from .middleware import RequestIdMiddleware
from .personas import get_persona_for_user, provision_persona_for_user

log = logging.getLogger(__name__)


def _ip_bucket(request: Request, kind: str) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"{kind}:{ip}"


# ──────────────────────────────────────────────────────────────────────────
# Handlers


def _mind_store(request: Request):
    """Dig the mind's store out of app.state.runtime, if present."""
    rt = getattr(request.app.state, "runtime", None)
    mind = getattr(rt, "mind", None) if rt else None
    return getattr(mind, "store", None) if mind else None


async def auth_register(request: Request) -> JSONResponse:
    if not rate_limit(_ip_bucket(request, "register"), limit=10):
        return JSONResponse({"error": "Too many attempts. Try again shortly."}, 429)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected JSON body."}, 400)
    try:
        user = register(
            email=str(body.get("email", "")),
            password=str(body.get("password", "")),
            name=str(body.get("name", "")),
        )
    except AuthError as exc:
        return JSONResponse({"error": exc.message}, exc.status)
    token, expires = create_session(user["id"])

    # A2: provision a persona for the new account, in-process.
    persona_info: dict | None = None
    store = _mind_store(request)
    if store is not None:
        try:
            persona = await provision_persona_for_user(
                store,
                user_id=user["id"],
                user_name=user.get("name") or "",
            )
            persona_info = {
                "persona_key": persona["persona_key"],
                "upstream_id": persona["upstream_id"],
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("persona provisioning failed for user %s: %s", user["id"], exc)
            persona_info = {"error": "persona provisioning failed; retry /api/persona"}

    return JSONResponse({
        "user": {"id": user["id"], "email": user["email"], "name": user.get("name", "")},
        "token": token,
        "expires_at": expires,
        "persona": persona_info,
    })


async def auth_login(request: Request) -> JSONResponse:
    if not rate_limit(_ip_bucket(request, "login"), limit=10):
        return JSONResponse({"error": "Too many attempts. Try again shortly."}, 429)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected JSON body."}, 400)
    try:
        user = login(
            email=str(body.get("email", "")),
            password=str(body.get("password", "")),
        )
    except AuthError as exc:
        return JSONResponse({"error": exc.message}, exc.status)
    token, expires = create_session(user["id"])
    return JSONResponse({
        "user": {"id": user["id"], "email": user["email"], "name": user.get("name", "")},
        "token": token,
        "expires_at": expires,
    })


async def auth_me(request: Request) -> JSONResponse:
    try:
        user = require_user(request)
    except AuthError as exc:
        return JSONResponse({"error": exc.message}, exc.status)
    return JSONResponse({
        "user": {"id": user["id"], "email": user["email"], "name": user.get("name", "")},
    })


async def auth_logout(request: Request) -> JSONResponse:
    token = bearer_from_request(request)
    if token:
        revoke_token(token, reason="logout")
    return JSONResponse({"ok": True})


async def persona_me(request: Request) -> JSONResponse:
    """Return the current user's persona info. If they have one already,
    return it. If not, provision one (idempotent catch-up for users who
    registered before A2 landed or whose first provisioning failed).
    """
    try:
        user = require_user(request)
    except AuthError as exc:
        return JSONResponse({"error": exc.message}, exc.status)
    existing = get_persona_for_user(user["id"])
    if existing:
        return JSONResponse({
            "persona_key": existing["persona_key"],
            "upstream_id": existing["upstream_id"],
            "provisioned_at": existing["provisioned_at"],
            "source": existing["source"],
        })
    # Catch-up provisioning
    store = _mind_store(request)
    if store is None:
        return JSONResponse(
            {"error": "mind not available; persona cannot be provisioned here"},
            503,
        )
    try:
        persona = await provision_persona_for_user(
            store, user_id=user["id"], user_name=user.get("name") or "",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("persona catch-up failed for user %s: %s", user["id"], exc)
        return JSONResponse({"error": "persona provisioning failed"}, 500)
    return JSONResponse({
        "persona_key": persona["persona_key"],
        "upstream_id": persona["upstream_id"],
        "provisioned_at": persona["provisioned_at"],
        "source": persona["source"],
    })


# ──────────────────────────────────────────────────────────────────────────
# Mount


GATEWAY_MIDDLEWARE: list[Middleware] = [
    Middleware(RequestIdMiddleware),
]


GATEWAY_ROUTES: list[Route] = [
    Route("/api/auth/register", auth_register, methods=["POST"]),
    Route("/api/auth/login", auth_login, methods=["POST"]),
    Route("/api/auth/me", auth_me, methods=["GET"]),
    Route("/api/auth/logout", auth_logout, methods=["POST"]),
    Route("/api/persona", persona_me, methods=["GET"]),
]


def register_gateway_routes(app: Starlette) -> None:
    """Mount the /api/* gateway routes onto the main app.

    Call this once at app startup. The gateway's SQLite is lazy-init
    (first query opens the file). Env vars that matter:
        SAWII_DATA_DIR         — base dir for gateway.sqlite3 (default ./data)
        SAWII_GATEWAY_DB       — explicit DB path (overrides SAWII_DATA_DIR)
        SAWII_SESSION_TTL      — session lifetime in seconds (default 30 days)
        SAWII_ALLOW_REGISTRATION — "0" closes registration
        SAWII_RATE_LIMIT_PER_MINUTE — default 30
    """
    app.router.routes.extend(GATEWAY_ROUTES)
    # NOTE: adding a middleware onto an already-built Starlette app isn't
    # supported. For now, the gateway's RequestIdMiddleware is applied only
    # if the host app has not disabled it. The companion-app deployment can
    # add Middleware(RequestIdMiddleware) at app construction time if it
    # wants per-request logging; this is deliberately not forced here, to
    # keep the opt-in surface clean.
