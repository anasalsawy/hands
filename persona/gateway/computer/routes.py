"""HTTP + WebSocket routes for live computer capture cards.

Endpoints:
    GET  /v1/computer/windows   — enumerable top-level windows
    GET  /v1/computer/capture   — capability probe
    WS   /v1/computer/stream    — continuous JPEG frames (binary) for one card

Query / first-message subscribe fields:
    kind=desktop|window|region
    hwnd / pid / title          (window)
    x,y,w,h                     (region)
    fps=2..15 (default 8)
    exclude_self=1|0
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from .capture import capture_available, list_windows
from .stream import ComputerStreamSession, StreamSpec

log = logging.getLogger(__name__)


async def _pid(request_or_ws) -> tuple[bool, object]:
    runtime = getattr(request_or_ws.app.state, "runtime", None)
    if runtime is None:
        return True, None
    auth = request_or_ws.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else request_or_ws.query_params.get("key", "")
    from ...identity import persona_by_key
    persona = await persona_by_key(runtime.mind.store, key) if key else None
    return (persona is not None), (persona["id"] if persona else None)


def _denied() -> JSONResponse:
    return JSONResponse({"error": "invalid persona key"}, status_code=401)


def _spec_from_mapping(data: dict | None, qp) -> StreamSpec:
    src = dict(data or {})
    # Query params fill gaps.
    for k in ("kind", "title"):
        if k not in src and qp.get(k) is not None:
            src[k] = qp.get(k)
    for k in ("hwnd", "pid", "x", "y", "w", "h", "fps", "quality"):
        if k not in src and qp.get(k) not in (None, ""):
            try:
                src[k] = float(qp.get(k)) if k == "fps" else int(qp.get(k))
            except (TypeError, ValueError):
                pass
    if "exclude_self" not in src and qp.get("exclude_self") is not None:
        src["exclude_self"] = str(qp.get("exclude_self")).lower() not in {"0", "false", "no"}
    kind = str(src.get("kind") or src.get("type") or src.get("session") or "desktop")
    return StreamSpec(
        kind=kind,
        hwnd=int(src["hwnd"]) if src.get("hwnd") not in (None, "") else None,
        pid=int(src["pid"]) if src.get("pid") not in (None, "") else None,
        title=str(src["title"]) if src.get("title") else None,
        x=int(src["x"]) if src.get("x") not in (None, "") else None,
        y=int(src["y"]) if src.get("y") not in (None, "") else None,
        w=int(src["w"]) if src.get("w") not in (None, "") else None,
        h=int(src["h"]) if src.get("h") not in (None, "") else None,
        fps=float(src.get("fps") or 8),
        exclude_self=bool(src.get("exclude_self", True)),
        quality=int(src.get("quality") or 55),
    ).clamp()


async def computer_windows(request: Request) -> JSONResponse:
    ok, _pid_val = await _pid(request)
    if not ok:
        return _denied()
    wins = await asyncio.to_thread(list_windows)
    return JSONResponse({
        "ok": True,
        "windows": [
            {
                "hwnd": w.hwnd,
                "title": w.title,
                "pid": w.pid,
                "left": w.left,
                "top": w.top,
                "width": w.width,
                "height": w.height,
                "visible": w.visible,
            }
            for w in wins
        ],
        "exclude_self": "best-effort blackout; not OS-level WGC exclude",
    })


async def computer_capture_status(request: Request) -> JSONResponse:
    ok, _pid_val = await _pid(request)
    if not ok:
        return _denied()
    return JSONResponse({"ok": True, **capture_available()})


class ComputerStream(WebSocketEndpoint):
    """One WebSocket = one live card. Binary JPEG frames after a JSON ready."""

    encoding = None

    async def on_connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        ok, _pid_val = await _pid(websocket)
        if not ok:
            await websocket.send_text(json.dumps({"type": "error", "detail": "invalid persona key"}))
            await websocket.close(code=4401)
            return

        caps = capture_available()
        if not caps.get("ok"):
            await websocket.send_text(json.dumps({
                "type": "error",
                "detail": caps.get("error") or "capture unavailable",
                **caps,
            }))
            await websocket.close(code=1013)
            return

        spec = _spec_from_mapping(None, websocket.query_params)
        websocket.state.spec = spec
        websocket.state.session = None

        async def on_frame(jpeg: bytes, meta: dict) -> None:
            # Prefer binary frames (UI LiveComputerStream paints ArrayBuffer).
            try:
                await websocket.send_bytes(jpeg)
            except Exception:  # noqa: BLE001
                raise

        session = ComputerStreamSession(spec=spec, on_frame=on_frame)
        websocket.state.session = session

        await websocket.send_text(json.dumps({
            "type": "ready",
            "kind": spec.kind,
            "fps": spec.fps,
            "exclude_self": spec.exclude_self,
            "backend": caps.get("backend"),
            "note": caps.get("exclude_self"),
            "hwnd": spec.hwnd,
            "pid": spec.pid,
            "title": spec.title,
        }))
        session.start()

    async def on_receive(self, websocket: WebSocket, data: Any) -> None:
        # Late subscribe / retarget messages arrive through the endpoint's own
        # receive loop; a second reader would race it for the same socket.
        if not isinstance(data, str):
            return
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        if not (msg.get("type") in {"subscribe", "configure", "retarget"} or msg.get("kind") or msg.get("session")):
            return
        new_spec = _spec_from_mapping(msg, websocket.query_params)
        old: ComputerStreamSession | None = getattr(websocket.state, "session", None)
        if old is not None:
            await old.stop()

        async def on_frame(jpeg: bytes, meta: dict) -> None:
            await websocket.send_bytes(jpeg)

        session = ComputerStreamSession(spec=new_spec, on_frame=on_frame)
        websocket.state.session = session
        websocket.state.spec = new_spec
        await websocket.send_text(json.dumps({
            "type": "ready",
            "kind": new_spec.kind,
            "fps": new_spec.fps,
            "exclude_self": new_spec.exclude_self,
            "hwnd": new_spec.hwnd,
            "pid": new_spec.pid,
            "title": new_spec.title,
        }))
        session.start()

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        session: ComputerStreamSession | None = getattr(websocket.state, "session", None)
        if session is not None:
            with contextlib.suppress(Exception):
                await session.stop()


def register_computer_routes(app: Starlette) -> None:
    """Mount live computer capture routes. Safe on non-Windows (WS reports error)."""
    app.router.routes.extend([
        Route("/v1/computer/windows", computer_windows, methods=["GET"]),
        Route("/v1/computer/capture", computer_capture_status, methods=["GET"]),
        WebSocketRoute("/v1/computer/stream", ComputerStream),
    ])
