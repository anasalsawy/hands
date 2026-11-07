"""Twilio phone-call surface — Sawii places and receives real phone calls.

Shape of the experience (companion-app's old plugins/phone_call.py set the
pattern): both lobes land into one phone call. L speaks the beats the caller
hears; R reads the caller's audio and feeds the chord into L's next beat.
The UI shows the live transcript interleaved with inner notes so the owner
can watch Maya take a call.

This module implements the TwiML Gather loop (one request/response per
turn, each Gather captures speech and POSTs transcript to the next webhook).
It is simpler than Media Streams and enough for the YTA use case.
Dry-run barge-in is the same interrupt as chat and voice: a second thing
the caller says while she is still replying is handed to the silent lobe,
which takes the mic after the current sentence. Twilio Media Streams
(continuous audio, not the Gather loop) is still a later step.

Twilio integration is lazy. When TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN +
TWILIO_FROM + SAWII_PUBLIC_URL are set, we drive real calls. Otherwise we
run in dry-run mode: /v1/calls/outbound returns a simulated call id and the
TwiML flow can be exercised against POST bodies for testing.

Env vars:
    TWILIO_ACCOUNT_SID      — your Twilio account id
    TWILIO_AUTH_TOKEN       — your Twilio auth token
    TWILIO_FROM             — the Twilio phone number to call from
    SAWII_PUBLIC_URL        — the public base URL for Twilio webhooks
                              (e.g. https://sawii-persona.railway.app)
    SAWII_CALLS_ENABLED     — "0" turns the call routes off. On otherwise.

When any of the first four is missing, the module runs in dry-run mode and
simulates the Twilio flow in-memory. Tests never need real Twilio.
"""
from __future__ import annotations

import asyncio
import json
import os
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Config


@dataclass
class TwilioConfig:
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""
    public_url: str = ""

    @classmethod
    def from_env(cls) -> "TwilioConfig":
        return cls(
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
            from_number=os.getenv("TWILIO_FROM", "").strip(),
            public_url=os.getenv("SAWII_PUBLIC_URL", "").strip().rstrip("/"),
        )

    @property
    def live(self) -> bool:
        """Whether we have everything to drive a real call."""
        return bool(self.account_sid and self.auth_token
                    and self.from_number and self.public_url)


# ──────────────────────────────────────────────────────────────────────────
# Call session state (one per active call)


@dataclass
class CallEvent:
    at: float
    kind: str  # "you" (Maya) | "them" (caller) | "coaching" (A↔B whisper) | "sys"
    text: str


@dataclass
class CallSession:
    """One phone call's state. Owns the transcript, the subscribers, and the
    'what should Maya say next?' handler.
    """
    sid: str
    to_number: str
    hint: str = ""  # the brief Maya was handed when she placed/answered
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    events: list[CallEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    # The current turn's next-beat text, set by compose_next_beat()
    _next_say: str = ""
    # Mode: "live" or "dry-run"
    mode: str = "dry-run"
    # Whose call this is. Set from the persona key (outbound) or
    # SAWII_CALLS_INBOUND_PERSONA_ID (inbound). None = no mind, dry ack only.
    persona_id: int | None = None
    persona_name: str = ""
    tz: str = ""
    # The human/remote endpoint on the other side of the Twilio number.
    peer_number: str = ""
    # The reply she is in the middle of. A new "heard" while this is running
    # is a barge-in, not a second turn.
    _reply_task: Any = None
    # Random capability used by the Twilio Media Streams websocket. It is
    # embedded only in TwiML returned to Twilio and prevents arbitrary clients
    # from attaching themselves to an active phone call.
    media_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def attach(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.subscribers.add(q)
        # Replay recent history
        for ev in self.events[-20:]:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                break
        return q

    def detach(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def push(self, kind: str, text: str) -> CallEvent:
        ev = CallEvent(at=time.time(), kind=kind, text=text)
        self.events.append(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    pass
        return ev


# ──────────────────────────────────────────────────────────────────────────
# Call registry


class CallHub:
    """Server-wide active-call registry. One active call per persona key
    (the Twilio integration ties a call to its persona at creation time;
    the registry just maps sid → session).
    """

    def __init__(self):
        self.sessions: dict[str, CallSession] = {}
        self.config = TwilioConfig.from_env()
        self._lock = asyncio.Lock()

    def new_sid(self, prefix: str = "CA") -> str:
        return f"{prefix}{secrets.token_hex(8)}"

    async def place_outbound(self, to_number: str, hint: str = "", *, mind: Any = None,
                             persona: dict | None = None) -> CallSession:
        """Start a new outbound call. In live mode, uses the Twilio REST API
        to initiate the call; the Twilio POST to /v1/calls/twilio/voice will
        then drive the gather loop. In dry-run, we create a session and
        simulate the first greeting locally."""
        async with self._lock:
            if self.config.live:
                sid = await self._place_outbound_live(to_number, hint)
                mode = "live"
            else:
                sid = self.new_sid()
                mode = "dry-run"
            session = CallSession(sid=sid, to_number=to_number, hint=hint, mode=mode,
                                  persona_id=(persona or {}).get("id"),
                                  persona_name=(persona or {}).get("name") or "",
                                  peer_number=to_number)
            session._mind = mind  # type: ignore[attr-defined]
            self.sessions[sid] = session
            session.push("sys", f"call placed to {to_number}  mode={mode}")

            if mode == "dry-run":
                # In dry-run, kick off a first beat synchronously so the UI
                # has something to show immediately.
                opener = compose_opening_beat(hint, session.persona_name)
                session._next_say = opener
                session.push("you", opener)
            return session

    async def _place_outbound_live(self, to_number: str, hint: str) -> str:
        """Hit Twilio's REST API to create the call. Returns the real SID."""
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise RuntimeError("httpx required to call Twilio REST API") from exc
        cfg = self.config
        voice_url = (
            f"{cfg.public_url}/v1/calls/twilio/voice"
            f"?{urlencode({'hint': hint})}"
        )
        status_url = f"{cfg.public_url}/v1/calls/twilio/status"
        body = urlencode({
            "To": to_number,
            "From": cfg.from_number,
            "Url": voice_url,
            "StatusCallback": status_url,
            "StatusCallbackEvent": "completed",
        })
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{cfg.account_sid}/Calls.json",
                content=body,
                headers={"content-type": "application/x-www-form-urlencoded"},
                auth=(cfg.account_sid, cfg.auth_token),
            )
        if res.status_code >= 300:
            raise RuntimeError(f"Twilio REST error {res.status_code}: {res.text[:300]}")
        return res.json()["sid"]

    def end(self, sid: str, reason: str = "ended") -> None:
        s = self.sessions.get(sid)
        if s and s.active:
            s.ended_at = time.time()
            s.push("sys", f"call ended: {reason}")


# ──────────────────────────────────────────────────────────────────────────
# The next-beat composer


def compose_opening_beat(hint: str, name: str = "") -> str:
    """Her first thing to say when the caller picks up. Keep it warm,
    one short sentence, no small-talk theater."""
    base = f"Hi, it's {name or 'Maya'}."
    if hint:
        return f"{base} {hint}".strip()
    return base


def _mind_of(session: CallSession) -> Any:
    """The Mind behind a call. The hub may hold the Mind or the Runtime."""
    mind = getattr(session, "_mind", None)
    return getattr(mind, "mind", mind)


async def hear(session: CallSession, heard: str) -> dict:
    """One thing the caller said.

    If she is mid-reply, the silent lobe takes the words and the reply
    already running changes direction. Otherwise this starts that reply.
    The first caller waits for the words she actually says; a barge-in
    returns at once so the person can keep talking.
    """
    heard = (heard or "").strip()
    mind = _mind_of(session)
    task = session._reply_task
    if task is not None and not task.done():
        session.push("them", heard)
        res: dict = {"accepted": False, "mode": None}
        if mind is not None and session.persona_id is not None and hasattr(mind, "interrupt"):
            res = mind.interrupt(session.persona_id, heard)
        if res.get("accepted"):
            session.push("barge", heard)
            return {"barged": True, "mode": res.get("mode"), "reply": ""}
        await task
    else:
        session.push("them", heard)

    async def _go() -> str:
        reply = await compose_next_beat(session, heard)
        session._next_say = reply
        session.push("you", reply)
        return reply

    task = asyncio.create_task(_go())
    session._reply_task = task
    reply = await task
    return {"barged": False, "mode": None, "reply": reply}


async def compose_next_beat(session: CallSession, heard: str) -> str:
    """What Maya should say next, given what the caller just said.

    This is where the mind integration lands when wired. For now, if the
    session carries a `._mind` attribute (set by the hub when the call is
    placed), we route the turn through the mind's chat method. Otherwise
    we return a dry-run acknowledgement so the gather loop still cycles.
    """
    mind = getattr(session, "_mind", None)
    # The hub may hold the Mind (outbound) or the Runtime (webhooks) - unwrap.
    mind = getattr(mind, "mind", mind)
    if mind is None or session.persona_id is None or not hasattr(mind, "phone_reply"):
        # Dry-run: tightly scoped acknowledgment so the loop keeps going.
        reply = f"got it — {heard[:60]}" if heard else "mm."
        return reply
    # Real path: ask the mind for a reply. Keep it tight — this is a phone
    # call, not a chat window.
    try:
        reply = await mind.phone_reply(heard, session=session)
    except Exception as exc:  # noqa: BLE001
        log.warning("mind.phone_reply failed: %s", exc)
        reply = "sorry — give me a moment and I'll come back to that."
    return (reply or "").strip() or "mm."


# ──────────────────────────────────────────────────────────────────────────
# TwiML generation


def twiml_say_and_gather(say_text: str, response_url: str, *, hangup: bool = False) -> str:
    """Return TwiML that says a line then gathers the caller's speech.

    If hangup=True, drop the Gather and emit a Say+Hangup — used to end
    the call cleanly on an explicit "goodbye".
    """
    say = _twiml_escape(say_text)
    if hangup:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Say voice="Polly.Joanna">{say}</Say>'
            '<Hangup/></Response>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>'
        f'<Say voice="Polly.Joanna">{say}</Say>'
        f'<Gather input="speech" speechTimeout="auto" '
        f'action="{_twiml_escape(response_url)}" method="POST"/>'
        f'<Say voice="Polly.Joanna">I didn\'t catch that. Call you back.</Say>'
        f'<Hangup/>'
        '</Response>'
    )


def _twiml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )


# ──────────────────────────────────────────────────────────────────────────
# Route handlers


async def _caller(request: Request) -> tuple[dict | None, JSONResponse | None]:
    """Owner-side routes need the persona key. A bare app with no runtime
    (unit tests) and no live Twilio is the only thing allowed through without
    one - it cannot reach a mind or place a real call."""
    runtime = getattr(request.app.state, "runtime", None)
    hub: CallHub = request.app.state.call_hub
    if runtime is None:
        if hub.config.live:
            return None, JSONResponse({"error": "calls need the mind runtime"}, status_code=503)
        return None, None
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    from ..identity import persona_by_key
    persona = await persona_by_key(runtime.mind.store, key) if key else None
    if persona is None:
        return None, JSONResponse({"error": "invalid persona key"}, status_code=401)
    return persona, None


def _owned(s: "CallSession | None", persona: dict | None) -> bool:
    return s is not None and (persona is None or s.persona_id == persona["id"])


def _twilio_signature_ok(request: Request, form, cfg: TwilioConfig) -> bool:
    """X-Twilio-Signature check (HMAC-SHA1 over the public URL + sorted POST
    params). Only enforced when Twilio is live; dry-run tests post by hand."""
    if not cfg.live:
        return True
    import base64
    import hashlib
    import hmac
    sig = request.headers.get("x-twilio-signature", "")
    url = cfg.public_url + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    data = url + "".join(f"{k}{form[k]}" for k in sorted(form.keys()))
    want = base64.b64encode(hmac.new(cfg.auth_token.encode(), data.encode(), hashlib.sha1).digest()).decode()
    return bool(sig) and hmac.compare_digest(sig, want)


def _inbound_persona_id() -> int | None:
    raw = os.getenv("SAWII_CALLS_INBOUND_PERSONA_ID", "").strip()
    return int(raw) if raw.isdigit() else None


async def calls_place(request: Request) -> JSONResponse:
    """Owner-side route: start an outbound call. Body: {"to": "+1...", "hint": "..."}."""
    hub: CallHub = request.app.state.call_hub
    persona, denied = await _caller(request)
    if denied:
        return denied
    body = await request.json()
    to = str(body.get("to", "")).strip()
    hint = str(body.get("hint", "")).strip()
    if not to:
        return JSONResponse({"error": "to is required"}, status_code=400)
    mind = getattr(request.app.state, "runtime", None)
    try:
        session = await hub.place_outbound(to, hint, mind=mind, persona=persona)
    except Exception as exc:  # noqa: BLE001
        log.warning("outbound call failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({"sid": session.sid, "to": to, "mode": session.mode})


async def calls_list(request: Request) -> JSONResponse:
    """Owner-side route: list active + recent calls."""
    hub: CallHub = request.app.state.call_hub
    persona, denied = await _caller(request)
    if denied:
        return denied
    out = []
    mine = [s for s in hub.sessions.values() if _owned(s, persona)]
    for s in sorted(mine, key=lambda x: x.started_at, reverse=True)[:30]:
        out.append({
            "sid": s.sid,
            "to": s.to_number,
            "started_at": s.started_at,
            "ended_at": s.ended_at,
            "mode": s.mode,
            "active": s.active,
            "events": len(s.events),
        })
    return JSONResponse({"calls": out, "twilio_live": hub.config.live})


async def calls_detail(request: Request) -> JSONResponse:
    hub: CallHub = request.app.state.call_hub
    persona, denied = await _caller(request)
    if denied:
        return denied
    sid = request.path_params["sid"]
    s = hub.sessions.get(sid)
    if not _owned(s, persona):
        return JSONResponse({"error": "unknown sid"}, status_code=404)
    return JSONResponse({
        "sid": s.sid,
        "to": s.to_number,
        "mode": s.mode,
        "started_at": s.started_at,
        "ended_at": s.ended_at,
        "active": s.active,
        "events": [{"at": e.at, "kind": e.kind, "text": e.text} for e in s.events],
    })


async def calls_end(request: Request) -> JSONResponse:
    hub: CallHub = request.app.state.call_hub
    persona, denied = await _caller(request)
    if denied:
        return denied
    sid = request.path_params["sid"]
    if not _owned(hub.sessions.get(sid), persona):
        return JSONResponse({"error": "unknown sid"}, status_code=404)
    hub.end(sid, reason="owner-ended")
    return JSONResponse({"ok": True})


async def calls_context(request: Request) -> JSONResponse:
    """Trusted worker result -> same live L phone session.

    R/Cline never speaks to the caller directly. It posts factual background
    results here; the existing L VoiceSession receives them as private system
    context and conveys them naturally.
    """
    hub: CallHub = request.app.state.call_hub
    persona, denied = await _caller(request)
    if denied:
        return denied
    sid = request.path_params["sid"]
    s = hub.sessions.get(sid)
    if not _owned(s, persona):
        return JSONResponse({"error": "unknown sid"}, status_code=404)
    body = await request.json()
    result = str(body.get("result", "")).strip()[:8000]
    if not result:
        return JSONResponse({"error": "result is required"}, status_code=400)
    s.push("worker", result)
    voice = getattr(s, "_voice", None)
    if voice is None or not s.active:
        return JSONResponse({"ok": True, "delivered": False, "reason": "live L voice session unavailable"})
    await voice.on_internal_context(result)
    return JSONResponse({"ok": True, "delivered": True})


async def calls_simulate_turn(request: Request) -> JSONResponse:
    """Dry-run-only: inject what the "caller" said. In live mode this comes
    from Twilio via the gather webhook; in dry-run the UI calls this to walk
    the loop forward.
    """
    hub: CallHub = request.app.state.call_hub
    persona, denied = await _caller(request)
    if denied:
        return denied
    body = await request.json()
    sid = request.path_params["sid"]
    s = hub.sessions.get(sid)
    if not _owned(s, persona):
        return JSONResponse({"error": "unknown sid"}, status_code=404)
    if s.mode != "dry-run":
        return JSONResponse({"error": "live call — Twilio drives turns"}, status_code=400)
    heard = str(body.get("heard", "")).strip()
    if not heard:
        return JSONResponse({"error": "heard is required"}, status_code=400)
    zone = (request.headers.get("x-sawii-timezone") or "").strip()
    if zone:
        s.tz = zone[:64]
    out = await hear(s, heard)
    return JSONResponse(out)


# --- Twilio webhooks ---------------------------------------------------


def _realtime_phone_requested() -> bool:
    return os.getenv("SAWII_CALLS_REALTIME", "1").strip().lower() not in ("0", "false", "no", "off")


def _realtime_phone_ready(request: Request) -> bool:
    """Realtime phone requires native-audio inference plus speech output.

    ASR is deliberately not a prerequisite and is never a substitute for a
    missing native-audio lobe model.
    """
    if not _realtime_phone_requested():
        return False
    runtime = getattr(request.app.state, "runtime", None)
    speech = getattr(runtime, "speech", None)
    mind = getattr(runtime, "mind", None)
    model = getattr(mind, "a", None)
    return bool(speech is not None and speech.has_tts and getattr(model, "supports_native_audio", False))


def _media_stream_twiml(session: CallSession, public_url: str) -> str:
    ws_base = public_url.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/v1/calls/twilio/media/{session.sid}/{session.media_token}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response><Connect><Stream url="' + _twiml_escape(url) + '" /></Connect></Response>'
    )


async def _companion_phone_event(request: Request, session: CallSession, event_type: str,
                                 *, text: str = "", actor: str = "", metadata: dict | None = None) -> None:
    """Forward phone lifecycle/native-worker events into the generic Companion
    integration ingress. Failure is intentionally non-fatal to the live call."""
    base = os.getenv("COMPANION_RUNTIME_URL", "").strip().rstrip("/")
    token = os.getenv("COMPANION_RUNTIME_TOKEN", "").strip()
    if not base or not token:
        return
    try:
        import httpx  # type: ignore
        owner_phone = "".join(ch for ch in os.getenv("SAWII_OWNER_PHONE", "") if ch.isdigit() or ch == "+")
        peer = "".join(ch for ch in (actor or session.peer_number or session.to_number) if ch.isdigit() or ch == "+")
        payload = {
            "id": f"{session.sid}:{event_type}:{len(session.events)}",
            "type": event_type,
            "account": session.persona_id,
            "actor": actor or session.peer_number or session.to_number,
            "text": text,
            "metadata": {"call_sid": session.sid, "active": session.active,
                         "owner_authorized": bool(owner_phone and peer and owner_phone == peer),
                         **(metadata or {})},
        }
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(base + "/api/integrations/phone/events", json=payload,
                              headers={"Authorization": "Bearer " + token})
    except Exception as exc:  # noqa: BLE001
        log.debug("companion phone event unavailable: %s", type(exc).__name__)


async def twilio_voice(request: Request) -> Response:
    """Twilio POSTs here when a call connects. Realtime production calls use
    bidirectional Media Streams and native-audio L/R inference. The Gather
    route remains only for explicit legacy mode and dry-run compatibility."""
    hub: CallHub = request.app.state.call_hub
    form = await request.form()
    if not _twilio_signature_ok(request, form, hub.config):
        return PlainTextResponse("forbidden", status_code=403)
    sid = str(form.get("CallSid", "")) or hub.new_sid("CA")
    hint = request.query_params.get("hint", "") or str(form.get("hint", "") or "")
    from_number = str(form.get("From", ""))
    to_number = str(form.get("To", ""))
    s = hub.sessions.get(sid)
    peer_number = from_number if from_number and from_number != hub.config.from_number else to_number
    if s is None:
        s = CallSession(sid=sid, to_number=to_number or from_number, hint=hint, mode="live",
                        persona_id=_inbound_persona_id(), peer_number=peer_number)
        s._mind = getattr(request.app.state, "runtime", None)  # type: ignore[attr-defined]
        if s.persona_id is not None and s._mind is not None:
            try:
                from ..identity import get_persona
                p = await get_persona(getattr(s._mind, "mind", s._mind).store, s.persona_id)
                s.persona_name = (p or {}).get("name") or ""
            except Exception:  # noqa: BLE001
                pass
        hub.sessions[sid] = s
    elif peer_number:
        s.peer_number = peer_number
    # Surface a real incoming ring on the existing Companion live bus as well
    # as the integration runtime. Current web/mobile clients already turn
    # "message" events into visible/OS notifications.
    inbound = bool(from_number and from_number != hub.config.from_number)
    runtime = getattr(request.app.state, "runtime", None)
    if inbound and runtime is not None and s.persona_id is not None:
        try:
            runtime.mind.bus.publish(s.persona_id, {
                "type": "message",
                "text": f"Incoming call from {s.peer_number or from_number}",
                "reason": "incoming_call",
                "channel": "app",
                "call_sid": sid,
            })
        except Exception:  # noqa: BLE001
            pass
    opener = compose_opening_beat(s.hint or hint, s.persona_name)
    s._next_say = opener
    # Ring/answer is an observation for L and a notification source for the
    # owner. It does not block Twilio's webhook response.
    asyncio.create_task(_companion_phone_event(request, s, "incoming_call", actor=s.peer_number or from_number))
    if _realtime_phone_ready(request):
        s.push("sys", "realtime native-audio media stream connected")
        return PlainTextResponse(_media_stream_twiml(s, hub.config.public_url), media_type="application/xml")
    if hub.config.live and _realtime_phone_requested():
        # Production realtime mode fails closed. Never silently replace native
        # L/R hearing with Twilio/ASR transcription.
        s.push("sys", "realtime call rejected: native-audio model or TTS unavailable")
        return PlainTextResponse(
            twiml_say_and_gather("The voice service is temporarily unavailable.", "", hangup=True),
            media_type="application/xml",
        )
    # Explicit legacy/dry-run mode only.
    s.push("you", opener)
    response_url = f"{hub.config.public_url}/v1/calls/twilio/response?sid={sid}"
    return PlainTextResponse(
        twiml_say_and_gather(opener, response_url),
        media_type="application/xml",
    )


async def twilio_response(request: Request) -> Response:
    """Each turn after the opening: Twilio hands us the ASR transcript of
    what the caller said; we respond with TwiML carrying Maya's next beat
    and another Gather."""
    hub: CallHub = request.app.state.call_hub
    form = await request.form()
    if not _twilio_signature_ok(request, form, hub.config):
        return PlainTextResponse("forbidden", status_code=403)
    sid = request.query_params.get("sid") or str(form.get("CallSid", ""))
    s = hub.sessions.get(sid)
    if not s:
        return PlainTextResponse(
            '<?xml version="1.0"?><Response><Hangup/></Response>',
            media_type="application/xml",
        )
    heard = str(form.get("SpeechResult", "")).strip()
    s.push("them", heard or "(unclear)")
    reply = await compose_next_beat(s, heard)
    s._next_say = reply
    s.push("you", reply)
    # Treat "bye", "goodbye", "we're done" as hang-up intent.
    low = heard.lower().strip()
    hangup = any(k in low for k in (" bye", "goodbye", "we're done", "that's all"))
    response_url = f"{hub.config.public_url}/v1/calls/twilio/response?sid={sid}"
    return PlainTextResponse(
        twiml_say_and_gather(reply, response_url, hangup=hangup),
        media_type="application/xml",
    )


async def twilio_status(request: Request) -> Response:
    hub: CallHub = request.app.state.call_hub
    form = await request.form()
    if not _twilio_signature_ok(request, form, hub.config):
        return PlainTextResponse("forbidden", status_code=403)
    sid = str(form.get("CallSid", ""))
    status = str(form.get("CallStatus", ""))
    if sid and status in ("completed", "failed", "no-answer", "busy", "canceled"):
        s = hub.sessions.get(sid)
        hub.end(sid, reason=status)
        if s is not None:
            asyncio.create_task(_companion_phone_event(request, s, "call_ended", text=status))
    return PlainTextResponse("")


# --- Live transcript stream for the UI --------------------------------


class CallStream(WebSocketEndpoint):
    encoding = "json"

    async def on_connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        sid = websocket.path_params.get("sid", "")
        hub: CallHub = websocket.app.state.call_hub
        s = hub.sessions.get(sid)
        runtime = getattr(websocket.app.state, "runtime", None)
        if s is not None and runtime is not None:
            # Same rule as the REST routes: only the persona that owns the call.
            from ..identity import persona_by_key
            key = websocket.query_params.get("key", "")
            persona = await persona_by_key(runtime.mind.store, key) if key else None
            if not _owned(s, persona) or persona is None:
                s = None
        if not s:
            await websocket.send_text(json.dumps({"error": "unknown call sid"}))
            await websocket.close()
            return
        queue = s.attach()
        websocket.state.queue = queue
        websocket.state.session = s
        websocket.state.pump = asyncio.create_task(self._pump(websocket, queue))

    async def _pump(self, websocket: WebSocket, queue: asyncio.Queue) -> None:
        try:
            while True:
                ev: CallEvent = await queue.get()
                await websocket.send_text(json.dumps({
                    "at": ev.at, "kind": ev.kind, "text": ev.text,
                }))
        except (WebSocketDisconnect, asyncio.CancelledError):
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("call stream pump error: %s", exc)

    async def on_disconnect(self, websocket: WebSocket, close_code: int) -> None:
        pump = getattr(websocket.state, "pump", None)
        if pump is not None:
            pump.cancel()
        s = getattr(websocket.state, "session", None)
        q = getattr(websocket.state, "queue", None)
        if s and q:
            s.detach(q)


# ──────────────────────────────────────────────────────────────────────────
# Mount


def register_call_routes(app: Starlette) -> None:
    """Mount the Twilio call surface + owner-side routes.

    Call this once at app startup. In dry-run (no Twilio env vars), the
    Twilio webhook routes still exist — they'll just be hit only by local
    tests. In live, Twilio needs the public_url pointing at this app.
    """
    app.state.call_hub = CallHub()
    from .phone_stream import TwilioMediaStream
    app.router.routes.extend([
        # Owner-side control
        Route("/v1/calls", calls_list, methods=["GET"]),
        Route("/v1/calls/outbound", calls_place, methods=["POST"]),
        Route("/v1/calls/{sid}", calls_detail, methods=["GET"]),
        Route("/v1/calls/{sid}/end", calls_end, methods=["POST"]),
        Route("/v1/calls/{sid}/simulate", calls_simulate_turn, methods=["POST"]),
        Route("/v1/calls/{sid}/context", calls_context, methods=["POST"]),
        # Twilio webhooks (TwiML gather loop)
        Route("/v1/calls/twilio/voice", twilio_voice, methods=["POST"]),
        Route("/v1/calls/twilio/response", twilio_response, methods=["POST"]),
        Route("/v1/calls/twilio/status", twilio_status, methods=["POST"]),
        WebSocketRoute("/v1/calls/twilio/media/{sid}/{token}", TwilioMediaStream),
        # Live transcript stream for the UI
        WebSocketRoute("/v1/calls/{sid}/stream", CallStream),
    ])
