"""HTTP + WebSocket routes for the browser pane.

Mounted onto the main FastAPI app via `register_browser_routes(app, state)`.

Endpoints:
    POST   /v1/browser/action       — fire a single tool call (debugging / UI nudges)
    POST   /v1/browser/snapshot     — force a snapshot broadcast to the stream
    WS     /v1/browser/stream       — subscribe to screenshot + action frames
    GET    /v1/browser/state        — one-shot current snapshot as JSON

These are gateway routes, so they live under /v1/browser/* alongside the mind's
/v1/* namespace, not under /api/* (which per architecture-split.md is reserved
for home layer like accounts and workspace). They talk to the browser session
and its broadcaster.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .session import BrowserSessionProto, create_browser_session
from .stream import BrowserBroadcast
from .tools import BrowserTools

log = logging.getLogger(__name__)


class _One:
    """One persona's browser: the session, its broadcast, its tools."""

    def __init__(self, session: BrowserSessionProto):
        self.session = session
        self.broadcast = BrowserBroadcast(session=session)
        self.tools = BrowserTools(session)


class BrowserHub:
    """One browser per persona (2026-10-21: it used to be one per deployment,
    which with accounts on meant every user shared - and could drive - the
    same browser). Lazy: nothing opens until a route or Sawii asks for it.

    ``pid=None`` is the single-user/dev slot, kept for old callers and tests.
    """

    def __init__(self, *, dry_run: bool = False, cdp_url: str | None = None):
        self.dry_run = dry_run
        self.cdp_url = cdp_url
        self._ones: dict = {}
        self._lock = asyncio.Lock()
        self.tasks = None

    async def _get(self, pid=None) -> _One:
        async with self._lock:
            one = self._ones.get(pid)
            if one is None:
                s = create_browser_session(dry_run=self.dry_run, cdp_url=self.cdp_url)
                try:
                    await s.start()
                except Exception:
                    # A failed Chromium launch used to escape this lock without
                    # an explicit diagnosis and could strand its Playwright
                    # driver. Allow a clean retry on the next call.
                    log.exception("browser launch failed (dry_run=%s)", self.dry_run)
                    with contextlib.suppress(Exception):
                        await s.close()
                    raise
                one = self._ones[pid] = _One(s)
            return one

    async def session(self, pid=None) -> BrowserSessionProto:
        return (await self._get(pid)).session

    async def tools(self, pid=None) -> BrowserTools:
        return (await self._get(pid)).tools

    async def broadcast(self, pid=None) -> BrowserBroadcast:
        return (await self._get(pid)).broadcast

    async def run(self, pid, name: str, arguments: dict, lane: str | None = None) -> str:
        """Sawii (or the UI) presses one button. The pane sees it happen.

        ``lane`` gives an execution lane (R1, R2...) its own browser, so
        independent lanes browse at the same time instead of colliding on
        the persona's one page. No lane is the persona's pane browser.
        """
        if name in {'browser_run_task','browser_task_status','browser_stop_task'}:
            if self.tasks is None:
                return json.dumps({'error':'Browser task runtime unavailable'})
            try:
                if name == 'browser_run_task':
                    result = self.tasks.start(pid, arguments.get('goal'))
                elif name == 'browser_stop_task':
                    result = await self.tasks.stop(pid)
                else:
                    result = self.tasks.status(pid)
                return json.dumps(result)
            except ValueError as exc:
                return json.dumps({'error':str(exc)})
        if self.tasks and self.tasks.active(pid) and name != 'browser_snapshot':
            return json.dumps({'error':'Stop the active browser task before manually controlling its page'})
        one = await self._get(pid if lane is None else (pid, lane))
        if name == 'browser_snapshot' and self.tasks and self.tasks.active(pid):
            from .session import read_page
            from .tools import _render_snapshot
            return _render_snapshot(await read_page(one.session, screenshot=False, ground=False))
        async with getattr(one.session, 'action_lock', asyncio.Lock()):
            result = await one.tools.run(name, arguments or {})
        try:
            await one.broadcast.push_action("tool", {"name": name, "arguments": arguments,
                                                     "result": str(result)[:300]})
            # Never on the hand's clock: live, this screenshot doubled every read.
            one.broadcast.schedule()
        except Exception as exc:  # noqa: BLE001 - the pane is a viewer, never a blocker
            log.warning("browser broadcast failed: %s", exc)
        return result

    async def shutdown(self) -> None:
        if self.tasks:
            await self.tasks.close()
        async with self._lock:
            for one in self._ones.values():
                pending = one.broadcast._pending
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                try:
                    await one.session.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("browser hub shutdown: %s", exc)
            self._ones.clear()


async def _pid(request_or_ws) -> tuple[bool, object]:
    """(allowed, pid). With the mind runtime mounted, the persona key is
    required and picks the browser; a bare app (tests/dev) uses the shared slot."""
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


# ──────────────────────────────────────────────────────────────────────
# Route handlers.


async def browser_state(request: Request) -> JSONResponse:
    ok, pid = await _pid(request)
    if not ok:
        return _denied()
    hub: BrowserHub = request.app.state.browser_hub
    sess = await hub.session(pid)
    from .session import read_page
    snap = await read_page(sess, ground=not (hub.tasks and hub.tasks.active(pid)))
    return JSONResponse({
        "url": snap.url,
        "title": snap.title,
        "elements": snap.elements,
        "visible_text": snap.visible_text,
        # omit screenshot bytes here — too big for a JSON probe
        "has_screenshot": snap.screenshot_png is not None,
    })


async def browser_action(request: Request) -> JSONResponse:
    """Debug/UI poke: fire one tool call from an authenticated client.

    This bypasses the kernel loop. It is NOT how the kernel invokes browser
    tools during a turn (that goes through ToolsCap.run, driven by the
    model's function_call stream). It is here so a developer can nudge the
    session from the UI or from curl.
    """
    ok, pid = await _pid(request)
    if not ok:
        return _denied()
    body = await request.json()
    name = str(body.get("name", ""))
    arguments = body.get("arguments", {}) or {}
    hub: BrowserHub = request.app.state.browser_hub
    try:
        result = await hub.run(pid, name, arguments)
    except Exception as exc:
        log.warning("browser action unavailable: %s: %s", type(exc).__name__, exc)
        return JSONResponse({"ok": False, "error": "browser session unavailable"}, status_code=503)
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if error:
        return JSONResponse({"ok": False, "error": str(error)[:500]}, status_code=502)
    return JSONResponse({"ok": True, "result": result})


async def browser_snapshot(request: Request) -> JSONResponse:
    ok, pid = await _pid(request)
    if not ok:
        return _denied()
    hub: BrowserHub = request.app.state.browser_hub
    bc = await hub.broadcast(pid)
    await bc.push_snapshot()
    return JSONResponse({"ok": True})


class BrowserStream(WebSocketEndpoint):
    """Subscribe to screenshot + action frames."""

    encoding = "json"

    async def on_connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        ok, pid = await _pid(websocket)
        if not ok:
            await websocket.send_text(json.dumps({"type": "error", "detail": "invalid persona key"}))
            await websocket.close(code=4401)
            return
        hub: BrowserHub = websocket.app.state.browser_hub
        lane = str(websocket.query_params.get("lane") or "")[:64]
        bc = await hub.broadcast((pid, lane) if lane else pid)
        queue = bc.attach()
        websocket.state.queue = queue
        websocket.state.bc = bc
        # Fire one snapshot immediately so a new viewer sees something.
        asyncio.create_task(bc.push_snapshot())
        # Pump task
        websocket.state.pump_task = asyncio.create_task(self._pump(websocket, queue))

    async def _pump(self, websocket: WebSocket, queue: asyncio.Queue) -> None:
        try:
            while True:
                frame = await queue.get()
                await websocket.send_text(json.dumps(frame))
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("browser stream pump error: %s", exc)
            with contextlib.suppress(Exception):
                await websocket.close()

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        pump = getattr(websocket.state, "pump_task", None)
        if pump is not None:
            pump.cancel()
        bc = getattr(websocket.state, "bc", None)
        q = getattr(websocket.state, "queue", None)
        if bc is not None and q is not None:
            bc.detach(q)


async def browser_task(request: Request) -> JSONResponse:
    ok, pid = await _pid(request)
    if not ok:
        return _denied()
    tasks = request.app.state.browser_hub.tasks
    if tasks is None:
        return JSONResponse({'error':'Browser task runtime unavailable'}, status_code=503)
    if request.method == 'GET':
        return JSONResponse(tasks.status(pid))
    if request.method == 'DELETE':
        return JSONResponse(await tasks.stop(pid))
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError('Expected an object')
        return JSONResponse(tasks.start(pid, data.get('goal')), status_code=202)
    except (ValueError, TypeError) as exc:
        return JSONResponse({'error':str(exc)}, status_code=400)


def register_browser_routes(app: Starlette, *, dry_run: bool = False, cdp_url: str | None = None) -> None:
    """Mount the browser hub + routes onto the main app.

    Call this once at app startup. The hub is lazy — nothing opens a
    browser until a route asks for the session.
    """
    app.state.browser_hub = BrowserHub(dry_run=dry_run, cdp_url=cdp_url)
    # Sawii's own hands: the mind presses the same buttons the pane shows.
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        runtime.mind.browser_hub = app.state.browser_hub
        from .tasks import BrowserTasks
        app.state.browser_hub.tasks = BrowserTasks(app.state.browser_hub, runtime.mind)
    app.router.routes.extend([
        Route("/v1/browser/task", browser_task, methods=["GET","POST","DELETE"]),
        Route("/v1/browser/state", browser_state, methods=["GET"]),
        Route("/v1/browser/action", browser_action, methods=["POST"]),
        Route("/v1/browser/snapshot", browser_snapshot, methods=["POST"]),
        WebSocketRoute("/v1/browser/stream", BrowserStream),
    ])

    async def _shutdown() -> None:
        await app.state.browser_hub.shutdown()

    # Starlette 1.x removed router.on_shutdown (this line crashed startup with
    # SAWII_BROWSER_PANE=1). The app's lifespan runs app.state.shutdown_hooks.
    hooks = getattr(app.state, "shutdown_hooks", None)
    if hooks is None:
        hooks = app.state.shutdown_hooks = []
    hooks.append(_shutdown)

