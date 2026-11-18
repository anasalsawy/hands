"""Her hands on the person's own computer (owner, 2026-10-22).

    "companion app should have access to system shell and files like any agent app"

The shared server never runs a shell for anyone: every account's memories and
the operator's keys live in that container. Instead, like Claude Code or Cline,
the hands run on the person's machine. A small runner (``scripts/sawii_hands.py``)
connects here with the persona key and offers tools (shell, files, later the
screen/mouse for computer use). During a home turn the mind offers those tools
to the kernel; a call is sent down this socket, the runner executes it in the
folders the person allowed (asking first if they chose that), and the result
comes back as the tool result.

Two places, like ChatGPT/Codex (owner, 2026-10-22):

    computer - the person's own machine (Codex-CLI / Cursor mode)
    cloud    - her own sandbox: the same runner in a separate container with
               no app secrets and no database (Codex-cloud mode)

Both may be connected at once; their tools reach the model as
``computer_<tool>`` and ``cloud_<tool>`` so she always knows which machine
she is acting on.

Wire protocol (JSON frames over WebSocket ``/v1/hands``):

    runner -> server  {"type": "hello", "key": "...", "place": "computer"|"cloud",
                       "tools": [<openai function schemas>], "host": "..."}
    server -> runner  {"type": "ready", "persona": "<name>"}
    server -> runner  {"type": "call", "id": "...", "name": "...", "arguments": {...}}
    runner -> server  {"type": "result", "id": "...", "ok": true, "output": "..."}
    runner -> server  {"type": "status", "kind": "approval", "text": "..."}  (waiting on the person)
    server -> runner  {"type": "cancel", "id": "..."}   (the turn let go of that call)
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Names the runner may not take: the mind's own tools and the browser's.
RESERVED_PREFIXES = ("browser_",)


PLACES = {
    "computer": "on the person's own computer",
    "cloud": "in your own cloud sandbox (not the person's computer)",
    "host": "on the device where the app is open",
}


@dataclass
class Runner:
    pid: int
    send: object                     # async callable(dict)
    tools: list[dict] = field(default_factory=list)   # as offered to the model (prefixed)
    host: str = ""
    place: str = "computer"
    lane: str = ""
    pending: dict = field(default_factory=dict)   # call id -> Future
    observation: dict | None = None
    last_app: str = ""
    names: dict = field(default_factory=dict)     # prefixed name -> runner's own tool name


class HandsHub:
    """Connected runners, keyed by (persona, place, lane).

    One runner per (place, lane): a newer connection on the *same* lane replaces
    the older one, while different lanes coexist. That is what lets several
    R-lines drive several engine processes on one machine at the same time - each
    ``sawii_hands.py --lane <lane>`` is its own hand, and every one of them is
    still the same mind thinking through its own line. A runner with no lane keeps
    the old single-hand-per-place behaviour.
    """

    def __init__(self, timeout: float = 120.0):
        self.runners: dict[tuple[int, str, str], Runner] = {}
        self.timeout = timeout

    @staticmethod
    def _key(pid: int, place: str, lane: str) -> tuple[int, str, str]:
        return (pid, place, str(lane or "")[:32])

    def connect(self, pid: int, send, tools: list[dict], host: str = "", place: str = "computer",
                lane: str = "") -> Runner:
        place = place if place in PLACES else "computer"
        lane = str(lane or "")[:32]
        offered, names = [], {}
        for t in tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if not (isinstance(fn, dict) and fn.get("name")) or str(fn["name"]).startswith(RESERVED_PREFIXES):
                continue
            pname = f"{place}_{fn['name']}"
            names[pname] = fn["name"]
            offered.append({"type": "function", "function": {
                **fn, "name": pname,
                "description": f"[{PLACES[place]}] " + str(fn.get("description", ""))}})
        old = self.runners.get(self._key(pid, place, lane))
        if old is not None:
            self._fail_all(old, "replaced by a newer connection")
        r = self.runners[self._key(pid, place, lane)] = Runner(
            pid=pid, send=send, tools=offered, host=host[:80],
            place=place, lane=lane, names=names)
        return r

    def disconnect(self, runner: Runner) -> None:
        if self.runners.get(self._key(runner.pid, runner.place, runner.lane)) is runner:
            self.runners.pop(self._key(runner.pid, runner.place, runner.lane), None)
        self._fail_all(runner, f"the hands {PLACES[runner.place]} disconnected")

    def connected(self, pid: int) -> list[Runner]:
        """Every runner of this persona, across places and lanes."""
        return [r for (p, _, _), r in self.runners.items() if p == pid]

    def schemas(self, pid: int) -> list[dict]:
        return [t for r in self.connected(pid) for t in r.tools]

    def names(self, pid: int) -> set[str]:
        return {t["function"]["name"] for t in self.schemas(pid)}

    async def call(self, pid: int, name: str, arguments: dict, lane: str = "") -> str:
        """Send one tool call and await its result.

        ``lane`` routes to the hand of one specific line of work (the value the
        runner put in its ``hello`` frame). With no lane, the call goes to the
        first hand that advertises the tool and is not already busy - so one
        persona can drive several engine processes without their results mixing.
        """
        lane = str(lane or "")[:32]
        candidates = [x for x in self.connected(pid) if name in x.names
                      and (not lane or x.lane == lane)]
        r = next((x for x in candidates if not x.pending), None) or (candidates[0] if candidates else None)
        if r is None:
            return json.dumps({"error": f"no hands connected for {name} (start sawii_hands)"})
        cid = uuid.uuid4().hex[:12]
        fut = asyncio.get_running_loop().create_future()
        r.pending[cid] = fut
        if r.names[name].startswith('ocu_') and isinstance(arguments.get('app'), str):
            r.last_app = arguments['app']
        try:
            await r.send({"type": "call", "id": cid, "name": r.names[name], "arguments": arguments or {}})
            out = await asyncio.wait_for(fut, timeout=self.timeout)
        except asyncio.TimeoutError:
            await self._cancel(r, cid)
            return json.dumps({"error": f"{name} took longer than {int(self.timeout)} s on your computer"})
        except asyncio.CancelledError:
            # The turn let go of this call (the person stopped her): tell the
            # runner, so typing halts and a shell command is killed.
            await self._cancel(r, cid)
            raise
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"{name} failed: {type(exc).__name__}"})
        finally:
            r.pending.pop(cid, None)
        return out

    async def observe(self, pid: int, place: str, lane: str = "") -> dict | None:
        runner = self.runners.get(self._key(pid, place, lane))
        if runner is None and not lane:
            # No lane given (or no lane-bound hand): the first hand on this place.
            runner = next((r for (p, pl, _), r in self.runners.items()
                           if p == pid and pl == place), None)
        if runner is None:
            return None
        tool = place + '_ocu_get_app_state'
        if runner.last_app and tool in runner.names:
            await self.call(pid, tool, {'app':runner.last_app}, lane=runner.lane)
        return runner.observation

    @staticmethod
    async def _cancel(runner: Runner, cid: str) -> None:
        try:
            await runner.send({"type": "cancel", "id": cid})
        except Exception:  # noqa: BLE001 - a gone runner has nothing to cancel
            pass

    def result(self, runner: Runner, frame: dict) -> None:
        fut = runner.pending.get(str(frame.get("id", "")))
        if fut is not None and not fut.done():
            out = frame.get("output", "")
            try:
                raw = json.loads(out)
                if isinstance(raw, dict) and isinstance(raw.get('content'), list) and not raw.get('isError'):
                    texts = [b.get('text','') for b in raw['content'] if b.get('type') == 'text']
                    images = [b for b in raw['content'] if b.get('type') == 'image' and len(b.get('data','')) <= 4_000_000]
                    runner.observation = {'source':runner.place, 'text':'\n'.join(texts)[:14000]}
                    if images:
                        runner.observation['image_url'] = 'data:' + images[0].get('mimeType','image/png') + ';base64,' + images[0]['data']
                    # Model receipts stay bounded; screenshot bytes live in observation, not text tokens.
                    out = json.dumps({'isError':False,'text':'\n'.join(texts)[:14000], 'has_image':bool(images)})
                elif isinstance(raw, dict) and raw.get('isError'):
                    out = json.dumps({'error':raw.get('content',[])})
            except (TypeError, ValueError):
                pass
            if not frame.get("ok", True):
                out = json.dumps({"error": str(frame.get("error") or out)[:4000]})
            fut.set_result(str(out)[:16000])

    @staticmethod
    def _fail_all(runner: Runner, why: str) -> None:
        for fut in list(runner.pending.values()):
            if not fut.done():
                fut.set_result(json.dumps({"error": why}))
        runner.pending.clear()


async def hands_ws(ws, mind) -> None:
    """The /v1/hands socket. ``mind`` has ``hands`` (a HandsHub) and ``store``."""
    from .identity import persona_by_key
    await ws.accept()
    try:
        hello = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=15))
    except Exception:  # noqa: BLE001
        await ws.close(code=4400)
        return
    persona = await persona_by_key(mind.store, str(hello.get("key", ""))) if hello.get("type") == "hello" else None
    if persona is None:
        await ws.send_json({"type": "error", "detail": "invalid persona key"})
        await ws.close(code=4401)
        return

    async def send(obj: dict) -> None:
        await ws.send_json(obj)

    runner = mind.hands.connect(persona["id"], send, hello.get("tools") or [], str(hello.get("host", "")),
                                str(hello.get("place", "computer")), str(hello.get("lane", "")))
    await ws.send_json({"type": "ready", "persona": persona.get("name", ""), "place": runner.place,
                        "lane": runner.lane,
                        "tools": [t["function"]["name"] for t in runner.tools]})
    mind.bus.publish(persona["id"], {"type": "hands", **hands_status(mind, persona["id"])})
    try:
        while True:
            frame = json.loads(await ws.receive_text())
            if frame.get("type") == "result":
                mind.hands.result(runner, frame)
                _show_screen(mind, persona["id"], runner.place, frame)
            elif frame.get("type") == "status":
                _waiting_on_person(mind, persona["id"], runner.place, frame)
    except Exception:  # noqa: BLE001 - disconnect, bad frame: either way the hands are gone
        pass
    finally:
        mind.hands.disconnect(runner)
        mind.bus.publish(persona["id"], {"type": "hands", **hands_status(mind, persona["id"])})


def _show_screen(mind, pid: int, place: str, frame: dict) -> None:
    """C15. What the hand just saw goes to every open app for this persona,
    so a phone watches the desktop she is working on."""
    out = str(frame.get("output") or "")
    try:
        raw = json.loads(out)
    except ValueError:
        return
    image = raw.get('image_b64') or ''
    mime = 'image/png'
    for block in raw.get('content', []):
        if block.get('type') == 'image':
            image, mime = block.get('data',''), block.get('mimeType','image/png')
            break
    if image and len(image) <= 4_000_000:
        mind.bus.publish(pid, {'type':'screen','place':place,'image':image,'mime_type':mime})


def _waiting_on_person(mind, pid: int, place: str, frame: dict) -> None:
    """C13. The hand is waiting on an approval: the app shows it, and the
    lobe that is talking is told, so she asks for the OK instead of going quiet."""
    text = str(frame.get("text") or "")[:300]
    kind = str(frame.get("kind") or "status")
    mind.bus.publish(pid, {"type": "kernel", "event": {"type": "hands", "kind": kind, "phase": "start",
                                                       "name": text, "place": place}})
    loop = getattr(mind, "_active_loops", {}).get(pid)
    if loop is not None and kind == "approval":
        loop.hear_inner(f"Their computer is asking them to approve: {text}. Tell them in a few "
                        "words that you need their OK on the screen, then carry on.")


def hands_status(mind, pid: int) -> dict:
    rs = mind.hands.connected(pid)
    # Keyed by place, or "place:lane" when a second hand works that place for a
    # specific R-line, so concurrent per-lane engines are all visible.
    places = {(r.place if not r.lane else f"{r.place}:{r.lane}"): {
        "host": r.host, "lane": r.lane, "place": r.place,
        "tools": [t["function"]["name"] for t in r.tools]} for r in rs}
    return {"connected": bool(rs),
            "places": places,
            "lanes": sorted({r.lane for r in rs if r.lane}),
            "host": ", ".join(f"{r.place}: {r.host}" for r in rs),
            "tools": [t["function"]["name"] for r in rs for t in r.tools]}
