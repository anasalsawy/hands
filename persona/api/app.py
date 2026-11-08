"""HTTP / WebSocket surface.

    /v1/chat/completions   OpenAI-compatible (any chat client works; streaming supported).
                           The persona owns the conversation: only the newest user message is
                           taken from the request - the history lives server-side, across devices.
    /v1/voice              realtime voice WebSocket
    /v1/events             live stream (SSE): running clock, pop-up messages, workshop results
    /v1/messages           outbox long-poll for clients without SSE
    /v1/files/{id}         download a document she made (make_document), own persona only
    /v1/persona/...        definition (preset / clone), clock, emotion, memory audit + wipe, loops, sleep
    /hooks/...             inbound SMS / WhatsApp (Twilio), Telegram, email
    /admin/personas        create personas (each gets its key = its identity)
    /app                   built-in web app (chat, voice, live timer, pop-ups)
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .. import emotion, identity
from ..channels.adapters import CHANNELS, Channels, twilio_signature_ok
from ..config import Settings, load_settings
from ..lanes.deep import attach as attach_deep
from ..life import anchor
from ..life.finetune import FineTuner
from ..life.initiative import Initiative
from ..life.sleep import Sleep
from ..mind import Mind
from ..profiles import PRESETS, build_clone, preset_list
from ..store import jdump, jload, now
from ..testing import run_test_turn, run_benchmark
from ..timesense import Clock
from ..voice.realtime import SpeechIO, VoiceSession, parse_frame

log = logging.getLogger(__name__)
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
# The companion client (ui/ in the repo) is built to ui/dist and copied here.
COMPANION_DIR = os.path.join(WEB_DIR, "companion")
MAX_BODY = 8_000_000
# Persona has one public model and one execution architecture.
# Internal L/R workers are never selected by a model alias.
HOST_MODELS = ("sawii-persona",)

class RateLimiter:
    def __init__(self, per_minute: int = 40):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        t = time.monotonic()
        q = self.hits[key]
        while q and t - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(t)
        return True


def _bearer(request_or_ws) -> str:
    auth = request_or_ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request_or_ws.query_params.get("key", "")


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": {"message": msg}}, status_code=status)


class Runtime:
    """Everything the API needs, built once."""

    def __init__(self, settings: Settings, mind: Mind | None = None, channels: Channels | None = None):
        self.s = settings
        self.mind = mind or Mind(settings)
        self.channels = channels or Channels(settings)
        self.finetuner = FineTuner(settings, self.mind.store, self.mind.subconscious, self.mind.a)
        self.sleep = Sleep(self.mind, self.finetuner)
        self.initiative = Initiative(self.mind, self.channels, self.sleep)
        self.mind.outbox = self.initiative
        attach_deep(self.mind, settings)
        self.speech = SpeechIO(settings.asr, settings.tts, settings.tts_voice)
        # Agents (Hermes, Cline) send one request per tool step; 40/min stalled
        # them mid-task. SAWII_CHAT_PER_MINUTE to tune.
        self.limiter = RateLimiter(per_minute=int(os.environ.get("SAWII_CHAT_PER_MINUTE", "120") or 120))


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> Starlette:
    settings = settings or (runtime.s if runtime else load_settings())
    rt = runtime or Runtime(settings)
    mind = rt.mind
    # persona id -> unix time of the last host-delegated job. A delegated turn
    # closes its stream (the host runs the tool), which used to mean minutes of
    # silence; while a job is outstanding the live channel keeps saying so.
    delegated: dict[int, float] = {}
    DELEGATED_TTL = 900.0

    async def persona_or_401(request) -> dict | None:
        key = _bearer(request)
        persona = await identity.persona_by_key(mind.store, key)
        if persona:
            return persona
        # Cursor uses its existing gateway bearer key; provision a dedicated
        # Persona identity for that key, rather than requiring a second secret.
        # This is opt-in and requires the Railway reference to the gateway key.
        configured = os.environ.get("SAWII_CURSOR_GATEWAY_KEY", "").strip()
        if not configured or not key or not hmac.compare_digest(key, configured):
            return None
        key_hash = identity.hash_key(key)
        def ensure(conn):
            conn.execute(
                "INSERT OR IGNORE INTO personas"
                " (key_hash, key_hint, name, definition, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (key_hash, key[:7] + "…" + key[-4:], "Cursor Persona",
                 "You are Persona, one unified coding assistant. Use the host's tools"
                 " through independent concurrent execution lanes. Record truthful"
                 " outcomes, preserve shared history and do not duplicate work.",
                 now()))
        await mind.store.tx(ensure)
        return await identity.persona_by_key(mind.store, key)

    async def body_json(request: Request) -> dict:
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise ValueError("body too large")
        if not raw:
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    # ------------------------------------------------------------------ basics
    async def health(request):
        return JSONResponse({"ok": True, "time": now(), "channels": rt.channels.available(),
                             "voice": {"asr": rt.speech.has_asr, "tts": rt.speech.has_tts},
                             "embedder": mind.embedder.model,
                             "workshop": "openhands" if getattr(mind.deep.worker_factory(), "name", "") == "openhands"
                             else "model"})

    async def command_post(request: Request):
        """Unblocked command ingress, even if all R tools are currently busy.

        202 means persisted/accepted, NOT executed. A client may retry using
        the same Idempotency-Key and then poll the command status.
        """
        persona = await persona_or_401(request)
        if not persona:
            return _err(401,"invalid persona key")
        # No app-defined per-minute limit for authenticated requests.
        from ..live_commands import CommandConflict, CommandValidation
        try:
            payload = await body_json(request)
            result = await mind.live_commands.submit(
                persona["id"], instruction=str(payload.get("instruction") or ""),
                action=payload.get("action","steer"),
                target_type=payload.get("target_type","auto"),
                target_id=str(payload.get("target_id") or ""),
                session_id=str(payload.get("session_id") or ""),
                idempotency_key=request.headers.get("idempotency-key")
                               or str(payload.get("idempotency_key") or ""))
        except CommandConflict as exc:
            return _err(409,str(exc))
        except (ValueError, CommandValidation, json.JSONDecodeError) as exc:
            return _err(400,str(exc))
        return JSONResponse(result,status_code=202,
            headers={"Cache-Control":"no-store"})

    async def timeline_get(request: Request):
        persona=await persona_or_401(request)
        if not persona:
            return _err(401,"invalid persona key")
        try:
            after=max(0,int(request.query_params.get("after","0")))
            limit=min(1000,max(1,int(request.query_params.get("limit","200"))))
            events=await mind.shared_timeline.read(persona["id"],after=after,limit=limit)
        except (ValueError, OverflowError):
            return _err(400,"invalid timeline cursor")
        return JSONResponse({"events":events,
                             "next_cursor":events[-1]["seq"] if events else after},
                            headers={"Cache-Control":"no-store"})

    async def command_list(request: Request):
        persona=await persona_or_401(request)
        if not persona:
            return _err(401,"invalid persona key")
        return JSONResponse({"commands":await mind.live_commands.list(persona["id"])},
                            headers={"Cache-Control":"no-store"})

    async def command_get(request: Request):
        persona=await persona_or_401(request)
        if not persona:
            return _err(401,"invalid persona key")
        row=await mind.live_commands.get(persona["id"],request.path_params["command_id"])
        return (JSONResponse(row,headers={"Cache-Control":"no-store"}) if row
                else _err(404,"unknown command"))

    async def agent_lanes(request: Request):
        persona=await persona_or_401(request)
        if not persona:
            return _err(401,"invalid persona key")
        return JSONResponse({"foreground":"L",
             "execution":await mind.live_commands.active_lanes(persona["id"])},
             headers={"Cache-Control":"no-store"})

    async def health_subsystems(request):
        """Per-subsystem alive/degraded/dead view. Reads the mind's health
        registry each subsystem writes to on success/failure, and augments
        with embedder's own tracking + a few live counts (lane active jobs,
        pending questions, etc.) that aren't event-driven."""
        subs = mind.health_registry.all()
        # Augment with live snapshots for things that aren't event-counted.
        # Embedder also has a richer snapshot in its own class.
        emb = mind.embedder.health()
        subs["embedder"] = {**subs.get("embedder", {"name": "embedder"}),
                            "live": emb}
        # The lines of work running now. Nothing queues: lines are concurrent, so
        # how many are running is the whole story - there is no queue depth.
        lane_live: dict = {}
        try:
            lane_live = {str(pid): {"active": count}
                         for pid, count in mind.lanes.counts().items()}
        except Exception:  # noqa: BLE001
            pass
        subs["lane"] = {**subs.get("lane", {"name": "lane"}), "live": lane_live}
        summary = mind.health_registry.summary()
        return JSONResponse({"ok": summary["ok"], "summary": summary, "subsystems": subs})

    async def execution_blueprint_view(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        pid = persona["id"]
        return JSONResponse({"revision": mind.execution_blueprint.revision(pid),
                             **mind.execution_blueprint.peek(pid)},
                            headers={"Cache-Control": "no-store"})

    async def conversation_board_view(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        pid = persona["id"]
        return JSONResponse({"revision": mind.conversation_board.revision(pid),
                             **mind.conversation_board.peek(pid)},
                            headers={"Cache-Control": "no-store"})

    async def execution_board_view(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        pid = persona["id"]
        if request.method == "GET":
            return JSONResponse(mind.picture_board.peek(pid),
                                headers={"Cache-Control": "no-store"})
        try:
            data = await body_json(request)
            revision = data.get("revision")
            if revision is not None and (not isinstance(revision, int) or revision < 0):
                return _err(400, "revision must be a nonnegative integer")
            view = await mind.picture_board.update(pid, data,
                                                  expected_revision=revision)
        except RuntimeError as exc:
            return _err(409, str(exc))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        return JSONResponse(view, headers={"Cache-Control": "no-store"})

    async def web_app(request):
        with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as fh:
            return HTMLResponse(fh.read())

    async def models(request):
        return JSONResponse({"object": "list", "data": [
            {"id": name, "object": "model", "owned_by": "sawii"}
            for name in HOST_MODELS
        ]})

    async def products(request):
        from ..products import describe
        return JSONResponse(describe())

    # ------------------------------------------------------------------ admin
    def admin_ok(request) -> bool:
        tok = settings.admin_token
        return bool(tok) and hmac.compare_digest(_bearer(request), tok)

    async def admin_personas(request: Request):
        if not admin_ok(request):
            return _err(401, "admin token required")
        if request.method == "GET":
            rows = await mind.store.all("SELECT * FROM personas ORDER BY id")
            return JSONResponse({"personas": [identity.public_view(r) for r in rows]})
        try:
            data = await body_json(request)
        except ValueError as exc:
            return _err(400, str(exc))
        definition, preset = _definition_from(data)
        pid, key = await identity.create_persona(
            mind.store, name=str(data.get("name", ""))[:80], definition=definition, preset=preset,
            user_name=str(data.get("user_name", ""))[:80], timezone=str(data.get("timezone", "UTC"))[:64],
            contact=_clean_contact(data.get("contact")), preferred_channel=_channel(data.get("preferred_channel")))
        return JSONResponse({"id": pid, "key": key, "note": "Store this key: it IS the persona's identity. "
                                                            "Only its hash is kept."}, status_code=201)

    # ------------------------------------------------------------------ Cursor native subagent lifecycle
    async def cursor_subagent_event(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        try:
            event = await body_json(request)
            from ..cursor_bridge import receive_cursor_subagent
            result = await receive_cursor_subagent(mind, persona["id"], event)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        return JSONResponse(result, status_code=200 if result.get("ok") else 404)

    # ------------------------------------------------------------------ chat
    async def responses(request: Request):
        from .responses import responses_endpoint
        return await responses_endpoint(request, chat_completions)

    async def chat_completions(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        # No app-defined per-minute limit for authenticated requests.
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        # One public model: retired aliases must never silently select alternate
        # code. The ONE exception is the Cursor adapter's own face: on the tagged
        # persona-cursor-deepinfra surface a request may name a hard-coded alias
        # from config.CURSOR_MODEL_ALIASES (or the CURSOR-DEEPINFRA worker's own
        # model id; "sawii-persona" keeps the cheap DeepSeek-V4.1-Flash baseline).
        # Aliases route ONLY this turn's execution/host-lane traffic (R-side
        # motor work through choose_host_inference); Persona's conversational
        # voice and reasoning keep their normal model (lobe A). Every generic
        # client, companion, or any model string outside the table still gets
        # the 400 below - cursor_alias_model_id() returns "" when the surface
        # gate is not met, so this path never resolves a table alias off-face.
        surface = _surface(request)
        from ..host_lanes import cursor_alias_model_id
        requested_model = data.get("model", "sawii-persona")
        cursor_model = cursor_alias_model_id(
            requested_model, surface=surface,
            cursor_worker=getattr(mind, "cursor_infer", None))
        if requested_model != "sawii-persona" and not cursor_model:
            return _err(400, "only sawii-persona is supported")
        if "sawii_host_mode" in data:
            return _err(400, "sawii_host_mode was removed; one unified Persona runtime")
        from ..host_lanes import parse_lane
        try:
            lane = parse_lane(request.headers.get("x-sawii-lane"), data.get("sawii_lane"))
        except ValueError as exc:
            return _err(400, str(exc))
        selected_session = request.headers.get("x-sawii-session-id") or data.get("sawii_session_id")
        if lane and (not isinstance(selected_session, str) or not 1 <= len(selected_session) <= 128):
            return _err(400, "explicit lane requires a session id of 1..128 characters")
        purpose = data.get("sawii_purpose", "conversation")
        if purpose not in ("conversation", "observation") or (purpose == "observation" and lane != "L"):
            return _err(400, "observation purpose requires L")
        lane_headers = {"X-Sawii-Lane": lane} if lane else {}
        msgs = data.get("messages") or []
        text = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                text = c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in (c or []) if isinstance(p, dict) and p.get("type") == "text")
                break
        if not text.strip():
            return _err(400, "no user message")
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(now())

        gateway_tools = data.get("gateway_tools")
        if not isinstance(gateway_tools, list):
            gateway_tools = None
        gateway_webhook = data.get("gateway_webhook")
        if not isinstance(gateway_webhook, dict):
            gateway_webhook = None
        mode: dict = {"lane": lane, "purpose": purpose} if lane else {}
        # The resolved alias model rides into the turn here, and BOTH the
        # streaming and non-streaming calls below splat mode as **kwargs, so
        # one assignment covers both faces. Empty off the Cursor surface (the
        # gate above), where mind.py ignores alias_model entirely and the host
        # keeps its normal provider.
        mode["alias_model"] = cursor_model
        # Visiting agent's own tools (READY-PLAN Step 3). Sawii decides, the caller runs.
        client_tools, client_results = _client_tools(data, msgs)
        # Production instrumentation: log tool capability names only, never
        # prompts, identity keys, arguments or tool results.
        if client_tools or data.get("tools"):
            from ..host_delegation import is_host_background_tool
            tool_names = [t["function"]["name"] for t in client_tools or []]
            bg_names = [t["function"]["name"] for t in client_tools or []
                        if is_host_background_tool(t)]
            log.info("host_lane_contract lane=%s tools=%s background_tools=%s "
                     "tool_results=%s streaming=%s",
                     lane or "auto", tool_names[:40], bg_names[:20],
                     len(client_results or []), bool(data.get("stream")))
        # Agent hosts commonly advertise their tools on every inference call.
        # An explicit L turn may see schemas and results for observation; it
        # must not execute them. Do not reject the whole host turn with HTTP 400.
        if lane == "L":
            mode["tools"] = False
        if client_tools or client_results:
            mode["client_tools"] = client_tools
            mode["client_results"] = client_results
        if client_results:
            delegated.pop(persona["id"], None)  # the host's job reported back
        # The OpenAI-compatible endpoint is an inference HOST connection.
        # Cursor/Cline/Hermes own their agent loops and tool runners. Persona
        # supplies the R model with shared memory; Companion uses its separate
        # app/voice endpoints to obtain the L conversation experience.
        # Default: the host owns the conversation, Persona is only its model.
        # A host that *wants* the persona (its own app, or a partner opting in)
        # asks for it explicitly — the persona system prompt is then appended
        # alongside the host's own system message, never overwriting it.
        home_wanted = str(data.get("sawii_home") or "").strip().lower() in {
            "1", "true", "yes", "on", "persona"
        } or (request.headers.get("x-sawii-home") or "").strip().lower() in {
            "1", "true", "yes", "on", "persona"
        }
        mode["home"] = bool(home_wanted)
        host_system, host_messages = _host_context(msgs)
        if lane == "R" and data.get("response_format"):
            host_system += "\nReturn only JSON matching this response format, without markdown:\n" + json.dumps(data["response_format"])
        if host_system:
            mode["host_system"] = host_system
        if host_messages is not None:
            mode["host_messages"] = host_messages
        im = (request.headers.get("x-sawii-interrupt") or str(data.get("sawii_interrupt") or "")).strip().lower()
        if im in ("steer", "stop"):
            mode["interrupt_mode"] = im
        mode["session_id"] = str(request.headers.get("x-sawii-session-id") or
                                 data.get("sawii_session_id") or "")[:128]
        mode.update(dual=True,
                    steer=_flag(request.headers.get("x-sawii-steer"), data.get("sawii_steer")))
        # Resolved on the cursor surface only; the mind re-points the cursor
        # worker with it and ignores it for its own conversation (see _respond_inner).
        if cursor_model:
            mode["alias_model"] = cursor_model
        zone = (request.headers.get("x-sawii-timezone") or str(data.get("timezone") or "")).strip()
        if zone:
            mode["tz"] = zone[:64]
        if data.get("stream"):
            async def turn_events():
                """The turn's own events, with the persona's lane rows merged in.

                The shared timeline publishes on the live bus, so a client that only
                reads this response stream (Cursor, Hermes) would watch a turn with
                no R1..Rn in it at all. Their rows ride along as sawii_event frames.
                """
                q = mind.bus.subscribe(persona["id"])
                src = mind.respond(persona["id"], text, surface=surface,
                                   gateway_tools=gateway_tools,
                                   gateway_webhook=gateway_webhook, **mode).__aiter__()
                try:
                    while True:
                        try:
                            ev = await src.__anext__()
                        except StopAsyncIteration:
                            return
                        while True:
                            try:
                                bev = q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if bev.get("type") in KERNEL_EVENTS:
                                yield bev
                        yield ev
                finally:
                    mind.bus.unsubscribe(persona["id"], q)
                    await src.aclose()

            async def gen():
              finish = "stop"
              call_index = 0     # running tool_calls index: batches must not renumber each other
              try:
                async for ev in turn_events():
                    if ev["type"] == "start":
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": "sawii-persona", "sawii_turn": ev["turn_id"],
                                 "sawii_status": "preparing", "choices": [
                                     {"index": 0, "delta": {}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif ev["type"] == "delta":
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": "sawii-persona",
                                 "choices": [{"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    elif ev["type"] == "error":
                        # Prefer the real lobe/provider detail (e.g. HTTP 402) so
                        # Tasker/Hands don't only show a sanitized dead-end string.
                        detail = str(ev.get("detail") or "").strip()[:240]
                        msg = detail or "The response failed before it was complete."
                        yield f"data: {json.dumps({'error': {'message': msg, 'type': 'ResponseError'}})}\n\n"
                    elif ev["type"] in ("tool_start", "tool_done"):
                        # Live tool chip events (sawii-only extension; OpenAI clients
                        # ignore unknown chunk fields).
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": "sawii-persona", "sawii_tool": ev,
                                 "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    elif ev["type"] == "client_tool_calls":
                        from ..host_delegation import route_host_output
                        _, sent_calls = route_host_output("", ev["calls"])
                        delta = {"tool_calls": [
                            {"index": i, "id": c["id"], "type": "function",
                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                            for i, c in enumerate(sent_calls, start=call_index)]}
                        call_index += len(sent_calls)
                        # The work goes to the host: say so, so a client that
                        # closes this stream (Hermes, Cursor) knows the next
                        # frames are a delegated job, not silence.
                        delegated[persona["id"]] = now()
                        mind.bus.publish(persona["id"], {"type": "delegated",
                                                         "tools": [c["name"] for c in sent_calls]})
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": "sawii-persona", "sawii_status": "delegated",
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        finish = "tool_calls"
                    elif ev["type"] in KERNEL_EVENTS:
                        # The two lobes, visible (beat/hold/silent/verify/steer).
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": "sawii-persona", "sawii_event": ev,
                                 "choices": [{"index": 0, "delta": {}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    elif ev["type"] == "done":
                        extra = {"sawii_text": ev.get("text", "")}
                        if ev.get("metrics"):
                            extra["sawii_metrics"] = ev["metrics"]
                        if ev.get("turn_id"):
                            extra["sawii_turn"] = ev["turn_id"]
                        if "verdict" in ev:
                            extra["sawii_verdict"] = ev["verdict"]
                        if ev.get("inner"):
                            extra["sawii_inner"] = ev["inner"]  # who spoke to A mid-reply (B/C), kinds only
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                 "model": "sawii-persona", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                                 **extra}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
              except Exception as exc:  # noqa: BLE001
                log.exception("chat stream failed")
                err = {"error": {"message": f"internal error: {type(exc).__name__}",
                                 "type": type(exc).__name__}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
              yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **lane_headers})
        out = await mind.chat(persona["id"], text, surface=surface, gateway_tools=gateway_tools,
                              gateway_webhook=gateway_webhook, **mode)
        from ..host_delegation import route_host_output
        routed_text, routed_calls = route_host_output(out.get("text", ""), out.get("client_tool_calls") or [])
        message: dict = {"role": "assistant", "content": routed_text}
        finish = "stop"
        if routed_calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in routed_calls]
            finish = "tool_calls"
        body = {"id": cid, "object": "chat.completion", "created": created, "model": "sawii-persona",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        if "verdict" in out:
            body["sawii_verdict"] = out["verdict"]
        if out.get("metrics"):
            body["sawii_metrics"] = out["metrics"]
        return JSONResponse(body, headers=lane_headers)

    async def perception_frame(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..perception import screen_frame
        return JSONResponse(await screen_frame(mind, persona["id"]))

    async def interrupt_post(request: Request):
        """A new command while she is still replying. Body: {"text": "..."}.
        accepted + mode "steer": she switches to it after her current sentence
        (do NOT also send it as a message). "stop": she ends after the current
        sentence; send it as the next message. Not accepted: just send it."""
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        from ..host_lanes import parse_lane
        try:
            lane = parse_lane(request.headers.get("x-sawii-lane"), data.get("sawii_lane"))
        except ValueError as exc:
            return _err(400, str(exc))
        return JSONResponse(mind.interrupt(persona["id"], str(data.get("text", "")), lane=lane,
                                         session_id=str(data.get("sawii_session_id") or "")))

    # Durable adaptive goals. The same persona key gates all status and commands.
    async def adaptive_tasks(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        agent = mind.goal_agent
        if request.method == "GET":
            return JSONResponse({"tasks": agent.list(persona["id"])})
        try:
            body = await body_json(request)
            task = await agent.create(persona["id"], str(body.get("goal", "")))
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        return JSONResponse(agent.status(persona["id"], task.id), status_code=202)

    async def adaptive_task(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        agent = mind.goal_agent
        pid, task_id = persona["id"], request.path_params["task_id"]
        status = agent.status(pid, task_id)
        if status is None:
            return _err(404, "unknown task")
        if request.method == "GET":
            return JSONResponse(status)
        try:
            body = await body_json(request)
            action = body.get("action")
            if action == "pause":
                agent.runner.pause(pid, task_id)
            elif action == "cancel":
                agent.runner.cancel(pid, task_id)
            elif action == "resume":
                await agent.resume(pid, task_id)
            else:
                return _err(400, "action must be pause, cancel or resume")
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        return JSONResponse(agent.status(pid, task_id))

    # ------------------------------------------------------------------ persona
    async def persona_get_put(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        if request.method == "GET":
            return JSONResponse(identity.public_view(persona))
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        updates = {}
        if any(k in data for k in ("definition", "preset", "clone")):
            definition, preset = _definition_from(data)
            updates.update(definition=definition, preset=preset)
        for k, n in (("name", 80), ("user_name", 80), ("timezone", 64)):
            if k in data:
                updates[k] = str(data[k])[:n]
        if "contact" in data:
            updates["contact"] = jdump(_clean_contact(data["contact"]))
        if "preferred_channel" in data:
            updates["preferred_channel"] = _channel(data["preferred_channel"])
        if "a_model" in data:
            updates["a_model_override"] = str(data["a_model"] or "")[:200]
        if updates:
            await mind.store.update("personas", persona["id"], **updates)
        return JSONResponse(identity.public_view(await identity.get_persona(mind.store, persona["id"])))

    async def presets(request):
        return JSONResponse({"presets": preset_list()})

    async def clock(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..timesense import valid_zone, place_from_zone
        zone = request.headers.get("x-sawii-timezone", "").strip()
        tz = zone if valid_zone(zone) else (persona.get("timezone") or "UTC")
        c = Clock(started_at=now(), last_reply_at=persona["last_reply_at"], last_user_at=persona["last_user_at"],
                  tz=tz, place=place_from_zone(tz))
        return JSONResponse(c.snapshot())

    async def emotion_view(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        own, user = jload(persona["mood"], {}), jload(persona["user_mood"], {})
        timeline = await mind.store.all("SELECT created_at, chord FROM events WHERE persona_id = ? AND chord != '{}' "
                                        "ORDER BY id DESC LIMIT 200", persona["id"])
        return JSONResponse({
            "persona": {"chord": own, "words": emotion.describe(own), "color": emotion.color(own)},
            "user": {"chord": user, "words": emotion.describe(user), "color": emotion.color(user)},
            "timeline": [{"at": t["created_at"], "color": emotion.color(jload(t["chord"], {})),
                          "words": emotion.describe(jload(t["chord"], {}))} for t in reversed(timeline)]})

    async def memory_view(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        return JSONResponse(await mind.memory.inspect(persona["id"]))

    async def memory_wipe(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        try:
            lid = int(request.path_params["layer_id"])
        except ValueError:
            return _err(400, "bad id")
        ok = await mind.memory.wipe(persona["id"], lid)
        return JSONResponse({"wiped": ok}, status_code=200 if ok else 404)

    async def loops_view(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        return JSONResponse({"open": await mind.memory.open_loops(persona["id"])})

    async def assistant_view(request):
        """D5: the life-assistant panel in one call - what she is holding for you
        (reminders with times, open loops), what she is working on in the
        background, what she sent you first, and what came back to her."""
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        pid = persona["id"]
        loops = await mind.memory.open_loops(pid)
        done = await mind.store.all(
            "SELECT id, kind, text, due_at, status, updated_at FROM loops WHERE persona_id = ? "
            "AND status != 'open' ORDER BY updated_at DESC LIMIT 10", pid)
        jobs = await mind.store.all(
            "SELECT id, task, mode, status, substr(result, 1, 600) AS result, created_at, finished_at "
            "FROM deep_jobs WHERE persona_id = ? ORDER BY id DESC LIMIT 15", pid)
        outreach = await mind.store.all(
            "SELECT id, channel, text, reason, status, created_at FROM outbox WHERE persona_id = ? "
            "ORDER BY id DESC LIMIT 15", pid)
        sudden = await mind.store.all(
            "SELECT id, text, source, status, created_at FROM thoughts WHERE persona_id = ? AND source IN "
            "('sudden','recollection','afterthought','dream','reflection') ORDER BY id DESC LIMIT 10", pid)
        return JSONResponse({
            "reminders": [lp for lp in loops if lp.get("due_at")],
            "open": [lp for lp in loops if not lp.get("due_at")],
            "done": done, "jobs": jobs, "outreach": outreach, "on_her_mind": sudden,
            "tz": persona.get("timezone") or "UTC",
        })

    async def loop_update(request):
        """Tick a loop off or cancel it: POST {"status": "done" | "cancelled" | "open"}."""
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        status = str(data.get("status", ""))
        if status not in ("done", "cancelled", "open"):
            return _err(400, "status must be done, cancelled or open")
        lid = int(request.path_params["loop_id"])
        row = await mind.store.one("SELECT id FROM loops WHERE id = ? AND persona_id = ?", lid, persona["id"])
        if not row:
            return _err(404, "no such loop")
        await mind.store.update("loops", lid, status=status, updated_at=now())
        mind.bus.publish(persona["id"], {"type": "memory_updated", "loop": lid})
        return JSONResponse({"ok": True, "id": lid, "status": status})

    async def anchor_view(request):
        """The wrongness dataset C keeps, and its rehearsal history. Read-only:
        this is the outside the mind cannot forge (life/anchor.py)."""
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        cases = await anchor.wrong_cases(mind.store, persona["id"])
        runs = await mind.store.all(
            "SELECT cases, passed, score, created_at FROM anchor_runs WHERE persona_id = ? "
            "ORDER BY id DESC LIMIT 30", persona["id"])
        return JSONResponse({"cases": cases, "runs": runs, "score": persona.get("anchor_score"),
                             "at": persona.get("anchor_at")})

    async def sleep_now(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        return JSONResponse(await rt.sleep.run(persona))

    async def subconscious_view(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        jobs = await mind.store.all("SELECT id, provider, status, output_model, detail, created_at FROM finetune_jobs "
                                    "WHERE persona_id = ? ORDER BY id DESC LIMIT 10", persona["id"])
        return JSONResponse({"buffer": await mind.subconscious.occupancy(persona["id"]), "jobs": jobs,
                             "a_model": persona["a_model_override"] or None})

    async def deep_submit(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(400, str(exc))
        task = str(data.get("task", "")).strip()
        if not task:
            return _err(400, "task required")
        mode = data.get("mode") if data.get("mode") in ("auto", "single", "map", "speculative") else "auto"
        job = await mind.deep.submit(persona["id"], task, mode)
        return JSONResponse({"job": job}, status_code=202)

    async def deep_get(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        row = await mind.store.one("SELECT * FROM deep_jobs WHERE id = ? AND persona_id = ?",
                                   int(request.path_params["job_id"]), persona["id"])
        if not row:
            return _err(404, "no such job")
        row["trace"] = jload(row["trace"], {})
        return JSONResponse(row)

    # ------------------------------------------------------------------ live
    async def messages_poll(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        try:
            wait = min(30.0, max(0.0, float(request.query_params.get("wait", "0") or 0)))
        except (TypeError, ValueError):
            wait = 0.0
        deadline = time.monotonic() + wait
        while True:
            rows = await mind.store.all("SELECT id, text, reason, channel, created_at FROM outbox WHERE persona_id = ? "
                                        "AND status IN ('queued','sent') AND channel = 'app' ORDER BY id", persona["id"])
            if rows or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.5)
        for r in rows:
            await mind.store.update("outbox", r["id"], status="read")
        return JSONResponse({"messages": rows})

    async def events_sse(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        pid = persona["id"]
        q = mind.bus.subscribe(pid)

        async def gen():
            try:
                yield "retry: 3000\n\n"
                p = persona
                ticks = 0
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=1.0)
                        if ev.get("type") == "message":
                            await mind.store.update("outbox", ev["id"], status="read")
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        # The clock tick used to hit the DB once per second, per open
                        # connection. The client interpolates its own counter from the
                        # last frame, so a refresh every 5s is plenty and cuts the
                        # per-connection query rate by 5x.
                        ticks += 1
                        if ticks % 5 == 0:
                            fresh = await identity.get_persona(mind.store, pid)
                            if fresh:
                                p = fresh
                            from ..timesense import valid_zone, place_from_zone
                            zone = valid_zone(request.query_params.get("tz", "")) or (p.get("timezone") or "UTC")
                            c = Clock(started_at=now(), last_reply_at=p["last_reply_at"],
                                      last_user_at=p["last_user_at"], tz=zone, place=place_from_zone(zone))
                            yield f"event: clock\ndata: {json.dumps(c.snapshot())}\n\n"
                        # A host-delegated job runs after its turn stream closed.
                        # While one is outstanding, keep saying so: minutes of
                        # silence read as a dead connection to every client.
                        at = delegated.get(pid)
                        if at is not None:
                            if now() - at > DELEGATED_TTL:
                                delegated.pop(pid, None)
                            elif ticks % 15 == 0:
                                yield ("event: delegated\ndata: " + json.dumps(
                                    {"type": "delegated", "still_running": True,
                                     "since": at}) + "\n\n")
                    if await request.is_disconnected():
                        break
            finally:
                mind.bus.unsubscribe(pid, q)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def hands_route(ws: WebSocket):
        from ..hands import hands_ws
        await hands_ws(ws, mind)

    async def hands_status(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..hands import hands_status as _hs
        return JSONResponse(_hs(mind, persona["id"]))

    async def computer_status(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        info = mind.computer.status(persona["id"])
        info["browser"] = mind.browser_hub is not None
        info["desktop"] = mind.desktop.enabled
        info["tools"] = sorted(set(info.get("tools", [])) | mind.desktop.names())
        return JSONResponse(info)

    async def plugin_list(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        return JSONResponse({"plugins": json.loads(await mind.plugins.call(persona["id"], "list_plugins", {}))})

    async def file_get(request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        found = mind.plugins.documents.get(persona["id"], request.path_params["file_id"])
        if not found:
            return _err(404, "no such file")
        path, meta = found
        from urllib.parse import quote
        from ..documents import FORMATS
        ext = FORMATS[meta["format"]][1]
        name = str(meta.get("name") or "document" + ext)
        ascii_name = "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch in "-_."))
        if ascii_name.startswith(".") or not ascii_name:
            ascii_name = "document" + ext
        return Response(path.read_bytes(), media_type=FORMATS[meta["format"]][0], headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff"})

    async def voice_status_view(request: Request):
        """Companion voice wiring (no secrets): asr/native/tts + Vocode pipeline."""
        from ..voice.vocode import pipeline_status
        pipeline = pipeline_status(mind, rt.speech)
        return JSONResponse({
            "ok": True,
            "websocket": "/v1/voice",
            "web_ui": "/app",
            "persona_speech": {
                "asr": pipeline.get("asr"),
                "tts": bool(rt.speech.has_tts),
                "native": pipeline.get("native"),
                "tts_model": getattr(settings.tts, "model", None),
                "hear": getattr(settings.voice_native, "model", None),
                "speak": getattr(settings.tts, "model", None),
                "dual_listen": True,
                "shared_mind": "DualLoop",
                "architecture": "vocode-faithful",
            },
            "pipeline": pipeline,
        }, headers={"Cache-Control": "no-store"})

    async def voice_ws(ws: WebSocket):
        await ws.accept()
        key = _bearer(ws)
        hello_tz = ""
        playback_ack = False
        if not key:
            try:
                hello = parse_frame(await asyncio.wait_for(ws.receive_text(), timeout=10))
                key = str(hello.get("key", ""))
                hello_tz = str(hello.get("tz", ""))[:64]
                playback_ack = hello.get("playback_ack") is True
            except Exception:  # noqa: BLE001
                await ws.close(code=4401)
                return
        persona = await identity.persona_by_key(mind.store, key)
        if not persona:
            await ws.send_json({"type": "error", "detail": "invalid persona key"})
            await ws.close(code=4401)
            return
        voice_lane = ws.query_params.get("lane")
        if voice_lane not in (None, "L"):
            await ws.close(code=4400)
            return
        session = VoiceSession(mind, persona, rt.speech, ws.send_json, ws.send_bytes,
                               lane=voice_lane,
                               dual=True,
                               steer=_flag(ws.query_params.get("steer"), None),
                               tz=hello_tz or ws.query_params.get("tz", ""),
                               playback_ack=playback_ack,
                               tasker_intent=(ws.query_params.get("tasker_intent") == "1"),
                               force_native_audio=(ws.query_params.get("tasker_intent") == "1"))
        q = mind.bus.subscribe(persona["id"])

        async def popups():
            while True:
                ev = await q.get()
                if ev.get("type") == "message":
                    await mind.store.update("outbox", ev["id"], status="read")
                    await session.speak_text(ev["text"])

        pop = asyncio.create_task(popups())
        await ws.send_json({"type": "ready", "persona": persona["name"],
                            "tts": "none" if settings.voice_output == "text" else
                                   "server" if rt.speech.has_tts else "client",
                            "realtime_voice": session.realtime_voice,
                            "voice_gate": session.gate,
                            "provider": getattr(mind.a, "provider", type(mind.a).__name__),
                            "model": getattr(mind.a, "base_model", ""),
                            "sampling_mode": getattr(mind.a, "sampling_mode", "unknown"),
                            "memory_policy": "background" if session.realtime_voice else "initial-grounding",
                            "asr": "native" if session.force_native_audio or settings.tinker_native_audio else
                                   "server" if rt.speech.has_asr else "client",
                            "architecture": ("dual-lobe-a-fast+b-native"
                                             if getattr(session, "dual_lobe_voice", False)
                                             else "vocode-faithful"),
                            "hear": getattr(settings.voice_native, "model", ""),
                            "speak": getattr(settings.tts, "model", ""),
                            "dual_listen": True,
                            "dual_lobe_voice": bool(getattr(session, "dual_lobe_voice", False)),
                            "lobe_a": (getattr(session._lobe_a, "host_label", "")
                                       if getattr(session, "_lobe_a", None) is not None else ""),
                            "lobe_b": getattr(settings.voice_native, "model", "") or "DualLoop",
                            "shared_mind": "DualLoop"})
        # A connected voice call is engagement even between utterances. The
        # microphone never goes idle just because the last sentence finished.
        voice_goals = getattr(mind, "goal_agent", None)
        if voice_goals is not None:
            voice_goals.engaged(persona["id"], True)
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await session.on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    await session.handle_frame(parse_frame(msg["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            pop.cancel()
            mind.bus.unsubscribe(persona["id"], q)
            try:
                await session.finish()
            finally:
                if voice_goals is not None:
                    voice_goals.engaged(persona["id"], False)

    # ------------------------------------------------------------------ inbound channels
    async def _inbound(persona: dict, text: str, channel: str):
        out = await mind.chat(persona["id"], text, channel=channel)
        reply = out.get("text", "")
        if reply:
            try:
                await rt.channels.deliver(persona, channel, reply)
            except Exception as exc:  # noqa: BLE001
                log.warning("reply delivery failed on %s: %s", channel, exc)

    async def hook_twilio(request: Request):
        try:
            form = dict(await request.form())
        except Exception:  # noqa: BLE001
            return _err(400, "bad form")
        url = settings.public_base_url.rstrip("/") + request.url.path
        if not twilio_signature_ok(settings.twilio_token, url, form, request.headers.get("x-twilio-signature", "")):
            return _err(403, "bad signature")
        sender = str(form.get("From", ""))
        channel = "whatsapp" if sender.startswith("whatsapp:") else "sms"
        persona = await identity.find_by_contact(mind.store, channel, sender.replace("whatsapp:", ""))
        if persona is None and channel == "whatsapp":
            persona = await identity.find_by_contact(mind.store, channel, sender)
        if persona and str(form.get("Body", "")).strip():
            mind.spawn(_inbound(persona, str(form["Body"]), channel), name="inbound-twilio")
        return Response("<Response></Response>", media_type="application/xml")

    async def hook_telegram(request: Request):
        if not settings.telegram_secret or not hmac.compare_digest(
                request.headers.get("x-telegram-bot-api-secret-token", ""), settings.telegram_secret):
            return _err(403, "bad secret")
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError):
            return _err(400, "bad body")
        msg = data.get("message") or {}
        chat_id = str((msg.get("chat") or {}).get("id", ""))
        persona = await identity.find_by_contact(mind.store, "telegram", chat_id)
        if persona and msg.get("text"):
            mind.spawn(_inbound(persona, str(msg["text"]), "telegram"), name="inbound-telegram")
        return JSONResponse({"ok": True})

    async def lane_question(request: Request):
        """A tool running on the gateway, mid-call, is asking her something.

        It waits on the gateway side until `reply_to_question` posts the answer
        back. Here it is parked and surfaced in the next brief; it never speaks
        on its own and never interrupts the turn in flight.
        """
        secret = os.environ.get("GATEWAY_INTERNAL_SECRET", "").strip()
        if not secret or not hmac.compare_digest(
                request.headers.get("x-internal-secret", "") or "", secret):
            return _err(401, "bad secret")
        try:
            data = await body_json(request)
        except Exception:  # noqa: BLE001
            return _err(400, "bad body")
        key = str(data.get("persona_key") or "").strip()
        if not key:
            return _err(400, "missing persona_key")
        persona = await identity.persona_by_key(mind.store, key)
        if persona is None:
            return _err(404, "unknown persona")
        mind.deliver_question(persona["id"], {
            "question_id": str(data.get("question_id") or ""),
            "tool_call_id": str(data.get("tool_call_id") or ""),
            "question": str(data.get("question") or ""),
            "default_if_quiet": str(data.get("default_if_quiet") or ""),
        })
        return JSONResponse({"ok": True})

    async def hook_email(request: Request):
        if not settings.email_inbound_secret or not hmac.compare_digest(
                request.query_params.get("secret", ""), settings.email_inbound_secret):
            return _err(403, "bad secret")
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError):
            return _err(400, "bad body")
        sender = str(data.get("from", "")).strip()
        if "<" in sender:
            sender = sender.split("<", 1)[1].rstrip(">")
        persona = await identity.find_by_contact(mind.store, "email", sender)
        if persona and str(data.get("text", "")).strip():
            mind.spawn(_inbound(persona, str(data["text"])[:20000], "email"), name="inbound-email")
        return JSONResponse({"ok": True})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await mind.shared_timeline.initialize()
        await mind.picture_board.warm()
        await mind.conversation_board.warm()
        await mind.execution_blueprint.warm()
        await mind.live_commands.recover_on_startup()
        if os.environ.get("SAWII_TINKER_CACHE_PROBE") == "1":
            async def probe_tinker_cache():
                from ..llm_tinker import TinkerInklingLLM

                if not isinstance(mind.a, TinkerInklingLLM):
                    log.warning("tinker_cache_probe skipped because A does not use the native Tinker adapter")
                    return
                prefix = "\n".join(f"Synthetic cache verification row {i}: red green blue."
                                   for i in range(96))
                counts = []
                try:
                    for suffix in ("first", "second", "third"):
                        result = await mind.a.chat([
                            {"role": "system", "content": prefix},
                            {"role": "user", "content": f"Reply OK for the {suffix} probe."},
                        ], max_tokens=8, temperature=0)
                        counts.append(result.cached_tokens)
                    log.warning("tinker_cache_probe cached_tokens=%s", counts)
                except Exception as exc:
                    import traceback

                    frame = traceback.extract_tb(exc.__traceback__)[-1]
                    log.warning("tinker_cache_probe failed: %s at %s:%s (%s)",
                                type(exc).__name__, os.path.basename(frame.filename), frame.lineno, frame.name)

            mind.spawn(probe_tinker_cache(), name="tinker-cache-probe")
        if settings.background_enabled:
            rt.initiative.start()
        yield
        await rt.initiative.stop()
        await mind.live_commands.shutdown()
        await mind.drain(timeout=10)
        # Hooks added after the app was built (browser hub etc.). Starlette 1.x
        # has no router.on_shutdown, so add-ons register here instead.
        for hook in getattr(app.state, "shutdown_hooks", []):
            try:
                await hook()
            except Exception:  # noqa: BLE001
                log.exception("shutdown hook failed")

    def companion_route():
        """The companion client if it has been built, else the old single page.

        When SERVE_CONSOLE is off the backend serves no UI at all: the gateway
        is the only interface, so there is no second one to confuse it with.
        """
        if not settings.serve_console:
            async def console_disabled(request):
                return JSONResponse({
                    "ok": True,
                    "console": "disabled",
                    "app": "the companion lives on the gateway, not this service",
                })
            return Route("/app", console_disabled, methods=["GET"])
        if os.path.isdir(COMPANION_DIR) and os.path.exists(os.path.join(COMPANION_DIR, "index.html")):
            return Mount("/app", app=StaticFiles(directory=COMPANION_DIR, html=True), name="companion")
        return Route("/app", web_app)

    # Testing interface: turn execution with instrumentation
    async def _test_persona(request, data) -> tuple[int | None, JSONResponse | None]:
        """The harness runs on the caller's own persona. Only the admin token may
        name another one (persona_id). It used to accept ANY bearer string and
        any persona id."""
        key = _bearer(request)
        if settings.admin_token and key and hmac.compare_digest(key, settings.admin_token):
            pid = data.get("persona_id")
            if pid is None:
                return None, _err(400, "admin: persona_id is required")
            try:
                return int(pid), None
            except (TypeError, ValueError):
                return None, _err(400, "persona_id must be an integer")
        persona = await identity.persona_by_key(mind.store, key) if key else None
        if not persona:
            return None, _err(401, "invalid persona key")
        # No app-defined per-minute limit for authenticated requests.
        return persona["id"], None

    async def test_turn(request: Request):
        """POST /v1/test/turn {"user_message": "...", "dual": true|false, "home": true|false}
        One real turn through the mind; returns the reply, metrics, tools,
        kernel events and L/R weights before/after."""
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError):
            return _err(400, "bad body")
        pid, denied = await _test_persona(request, data)
        if denied:
            return denied
        msg = str(data.get("user_message") or data.get("message") or "").strip()
        if not msg:
            return _err(400, "missing user_message")
        dual = _flag(None, data.get("dual"))
        res = await run_test_turn(mind, persona_id=pid, user_message=msg, dual=dual,
                                  home=bool(data.get("home", True)))
        return JSONResponse(res.to_dict(), status_code=200 if not res.error else 502)

    async def test_benchmark(request: Request):
        """POST /v1/test/benchmark {"iterations": 5, "dual": true|false}
        Real turns; ttft / longest silence / reply time stats. Run once with
        dual true and once false for the side-by-side."""
        try:
            data = await body_json(request)
        except (ValueError, json.JSONDecodeError):
            return _err(400, "bad body")
        pid, denied = await _test_persona(request, data)
        if denied:
            return denied
        try:
            iterations = max(1, min(int(data.get("iterations", 5)), 20))
        except (TypeError, ValueError):
            return _err(400, "iterations must be an integer")
        out = await run_benchmark(mind, persona_id=pid, iterations=iterations, dual=_flag(None, data.get("dual")),
                                  messages=data.get("messages") if isinstance(data.get("messages"), list) else None)
        return JSONResponse(out)

    async def knowledge_list(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..knowledge import list_sources
        return JSONResponse({"sources": await list_sources(mind.store, persona["id"])})

    async def knowledge_post(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..knowledge import ingest
        ctype = request.headers.get("content-type", "")
        try:
            if "application/json" in ctype:
                data = await request.json()
                name = str(data.get("name") or "note.txt")
                raw = str(data.get("text") or "").encode()
            else:
                form = await request.form()
                upload = form.get("file")
                if upload is None or not hasattr(upload, "read"):
                    return _err(400, "file is required")
                name = str(getattr(upload, "filename", "") or "upload.txt")
                raw = await upload.read()
            saved = await ingest(mind.store, mind.embedder, persona["id"], name, raw, llm=mind.a)
        except ValueError as exc:
            return _err(400, str(exc))
        return JSONResponse(saved)

    async def knowledge_delete(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..knowledge import delete_source
        try:
            source_id = int(request.path_params["source_id"])
        except (TypeError, ValueError):
            return _err(400, "bad id")
        if not await delete_source(mind.store, persona["id"], source_id):
            return _err(404, "unknown file")
        return JSONResponse({"ok": True})

    async def voice_clone_view(request: Request):
        persona = await persona_or_401(request)
        if not persona:
            return _err(401, "invalid persona key")
        from ..voice.clone import clear, save, status
        if request.method == "DELETE":
            clear(settings.data_dir, persona["id"])
            return JSONResponse({"present": False})
        if request.method == "GET":
            return JSONResponse(status(settings.data_dir, persona["id"]))
        try:
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                return _err(400, "file is required")
            name = str(getattr(upload, "filename", "") or "clone.wav")
            raw = await upload.read()
            note = str(form.get("reference_text") or "")
            saved = save(settings.data_dir, persona["id"], raw, name, reference_text=note)
        except ValueError as exc:
            return _err(400, str(exc))
        return JSONResponse(saved)

    routes = [
        Route("/", lambda r: RedirectResponse("/app/") if settings.serve_console
              else JSONResponse({"ok": True, "app": "gateway", "health": "/health"})),
        Route("/health", health),
        Route("/health/subsystems", health_subsystems),
        companion_route(),
        Route("/admin/personas", admin_personas, methods=["GET", "POST"]),
        Route("/v1/models", models),
        Route("/v1/products", products),
        Route("/v1/interrupt", interrupt_post, methods=["POST"]),
        Route("/v1/commands", command_post, methods=["POST"]),
        Route("/v1/commands", command_list, methods=["GET"]),
        Route("/v1/timeline", timeline_get, methods=["GET"]),
        Route("/v1/cursor/subagent-event", cursor_subagent_event, methods=["POST"]),
        Route("/v1/commands/{command_id}", command_get, methods=["GET"]),
        Route("/v1/agents", agent_lanes, methods=["GET"]),
        Route("/v1/persona/screen", perception_frame, methods=["GET"]),
        Route("/v1/adaptive/tasks", adaptive_tasks, methods=["GET", "POST"]),
        Route("/v1/adaptive/tasks/{task_id}", adaptive_task, methods=["GET", "POST"]),
        Route("/v1/hands", hands_status),
        Route("/v1/computer", computer_status),
        Route("/v1/plugins", plugin_list),
        Route("/v1/files/{file_id}", file_get),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/responses", responses, methods=["POST"]),
        Route("/v1/persona", persona_get_put, methods=["GET", "PUT"]),
        Route("/v1/persona/presets", presets),
        Route("/v1/persona/clock", clock),
        Route("/v1/persona/conversation-board", conversation_board_view, methods=["GET"]),
        Route("/v1/persona/execution-blueprint", execution_blueprint_view, methods=["GET"]),
        Route("/v1/persona/execution-sketchpad", execution_board_view, methods=["GET", "POST"]),
        Route("/v1/persona/execution-board", execution_board_view, methods=["GET", "POST"]),
        # Legacy route is an alias of Execution Board, not a separate third board.
        Route("/v1/persona/picture", execution_board_view, methods=["GET", "POST"]),
        Route("/v1/persona/emotion", emotion_view),
        Route("/v1/persona/memory", memory_view),
        Route("/v1/persona/memory/{layer_id}", memory_wipe, methods=["DELETE"]),
        Route("/v1/persona/loops", loops_view),
        Route("/v1/persona/loops/{loop_id:int}", loop_update, methods=["POST"]),
        Route("/v1/persona/assistant", assistant_view),
        Route("/v1/knowledge", knowledge_list, methods=["GET"]),
        Route("/v1/knowledge", knowledge_post, methods=["POST"]),
        Route("/v1/knowledge/{source_id:int}", knowledge_delete, methods=["DELETE"]),
        Route("/v1/voice/clone", voice_clone_view, methods=["GET", "POST", "DELETE"]),
        Route("/v1/voice-status", voice_status_view, methods=["GET"]),
        Route("/v1/persona/anchor", anchor_view),
        Route("/v1/persona/sleep", sleep_now, methods=["POST"]),
        Route("/v1/persona/subconscious", subconscious_view),
        Route("/v1/deep", deep_submit, methods=["POST"]),
        Route("/v1/deep/{job_id:int}", deep_get),
        Route("/v1/messages", messages_poll),
        Route("/v1/events", events_sse),
        WebSocketRoute("/v1/voice", voice_ws),
        WebSocketRoute("/v1/hands", hands_route),
        Route("/v1/test/turn", test_turn, methods=["POST"]),
        Route("/v1/test/benchmark", test_benchmark, methods=["POST"]),
        Route("/hooks/twilio", hook_twilio, methods=["POST"]),
        Route("/hooks/telegram", hook_telegram, methods=["POST"]),
        Route("/hooks/email", hook_email, methods=["POST"]),
        Route("/v1/lane/questions", lane_question, methods=["POST"]),
    ]
    # CORS: the browser build is same-origin, but the desktop and mobile shells
    # load bundled assets from capacitor://localhost / tauri://localhost and call
    # this server cross-origin. Auth is a bearer key, never a cookie, so a
    # permissive default is safe. See ui/PLATFORMS.md.
    middleware = []
    origins = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
    if origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=origins,
                                      allow_methods=["*"], allow_headers=["*"],
                                      expose_headers=["*"], max_age=86400))
    app = Starlette(routes=routes, middleware=middleware, lifespan=lifespan)
    app.state.runtime = rt

    # Gateway (accounts + sessions). Enabled by SAWII_GATEWAY=1. Lazy SQLite
    # — the DB file is only created when a route is first hit. Safe to merge
    # without flipping the flag.
    import os as _os
    if _os.getenv("SAWII_GATEWAY", "0") in ("1", "true", "yes"):
        from ..gateway import register_gateway_routes
        register_gateway_routes(app)
    # Her browser is part of the app. SAWII_BROWSER_PANE=0 turns the pane off.
    # SAWII_BROWSER_DRY_RUN=1 keeps the stub the tests use.
    import os as _os
    if _os.getenv("SAWII_BROWSER_PANE", "1") in ("1", "true", "yes"):
        from ..gateway.browser import register_browser_routes
        _dry = _os.getenv("SAWII_BROWSER_DRY_RUN", "0") in ("1", "true", "yes")
        _cdp = _os.getenv("SAWII_BROWSER_CDP_URL") or None
        register_browser_routes(app, dry_run=_dry, cdp_url=_cdp)
    # Live computer cards (Background stage). SAWII_COMPUTER_STREAM=0 disables.
    if _os.getenv("SAWII_COMPUTER_STREAM", "1") in ("1", "true", "yes"):
        from ..gateway.computer import register_computer_routes
        register_computer_routes(app)
    # The court deck: the headless layer of concurrent desktop cards
    # (persona/court.py). SAWII_COURTS=0 turns the routes off.
    if _os.getenv("SAWII_COURTS", "1") in ("1", "true", "yes"):
        from ..court import register_court_routes
        register_court_routes(app)
    # The phone button is on the companion, so the call routes are on.
    # Without Twilio credentials they are a dry-run: the screen places the
    # call and you type what the other person says. SAWII_CALLS_ENABLED=0
    # turns the routes off.
    import os as _os
    if _os.getenv("SAWII_CALLS_ENABLED", "1") in ("1", "true", "yes"):
        from ..voice.calls import register_call_routes
        register_call_routes(app)
    return app


def _definition_from(data: dict) -> tuple[str, str]:
    """preset id | clone fields | free text - always the user's words."""
    if isinstance(data.get("clone"), dict):
        c = data["clone"]
        samples = c.get("sample_messages") or []
        if isinstance(samples, str):
            samples = [s for s in samples.splitlines() if s.strip()]
        return build_clone(name=str(c.get("name", "someone"))[:80], relationship=str(c.get("relationship", ""))[:200],
                           how_they_talk=str(c.get("how_they_talk", ""))[:3000],
                           sample_messages=[str(s)[:500] for s in samples][:40], traits=str(c.get("traits", ""))[:3000],
                           shared_memories=str(c.get("shared_memories", ""))[:5000],
                           things_they_would_never_say=str(c.get("never", ""))[:1000]), "clone"
    preset = str(data.get("preset", "") or "")
    if preset in PRESETS:
        extra = str(data.get("definition", "") or "").strip()
        return (PRESETS[preset]["definition"] + ("\n" + extra if extra else ""))[:12000], preset
    return str(data.get("definition", "") or "")[:12000], ""


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def _host_context(msgs: list) -> tuple[str, list[dict] | None]:
    """A visiting agent's system prompt, and its transcript when it sends one.

    A client that sends only the newest message relies on her server-side
    history (None). An agent that sends the conversation owns it: we pass it
    through as OpenAI messages (assistant tool_calls + tool results kept)."""
    system = "\n\n".join(_text_of(m.get("content")) for m in msgs
                           if isinstance(m, dict) and m.get("role") in ("system", "developer"))
    rest = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")]
    has_images = any(isinstance(m.get("content"), list) and
                     any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
                     for m in rest)
    if len(rest) <= 1 and not has_images:
        return system.strip(), None
    out: list[dict] = []
    for m in rest:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": str(m.get("tool_call_id", "")),
                        "content": _text_of(m.get("content"))})
        elif role == "assistant":
            item: dict = {"role": "assistant", "content": _text_of(m.get("content")) or None}
            calls = [tc for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)]
            if calls:
                item["tool_calls"] = [{"id": str(tc.get("id", "")), "type": "function",
                                       "function": {"name": str((tc.get("function") or {}).get("name", "")),
                                                    "arguments": str((tc.get("function") or {}).get("arguments", "{}"))}}
                                      for tc in calls]
            elif item["content"] is None:
                item["content"] = ""
            out.append(item)
        else:
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": str(part.get("text", ""))})
                    elif part.get("type") == "image_url" and isinstance(part.get("image_url"), dict):
                        url = part["image_url"].get("url")
                        if isinstance(url, str) and (url.startswith("data:image/") or url.startswith("https://")):
                            image = {"url": url}
                            if part["image_url"].get("detail") in ("auto", "low", "high"):
                                image["detail"] = part["image_url"]["detail"]
                            parts.append({"type": "image_url", "image_url": image})
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": "user", "content": _text_of(content)})
    return system.strip(), out


def _client_tools(data: dict, msgs: list) -> tuple[list[dict] | None, list[dict] | None]:
    """OpenAI ``tools`` from a visiting agent, minus anything named like Sawii's
    own mind tools; plus the trailing ``role:"tool"`` answers, if this request
    is the continuation after Sawii called them."""
    from ..brains.voice_a import A_TOOLS
    own = {t["function"]["name"] for t in A_TOOLS}
    # A visiting host may expose its own delegate_task tool. Keep it only
    # when the schema advertises the background subagent contract.
    from ..host_delegation import is_host_background_tool
    tools = data.get("tools")
    client_tools = None
    if isinstance(tools, list):
        client_tools = [t for t in tools if isinstance(t, dict) and t.get("type", "function") == "function"
                        and isinstance(t.get("function"), dict) and t["function"].get("name")
                        and (t["function"]["name"] not in own or is_host_background_tool(t))] or None
    results: list[dict] = []
    names: dict = {}
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    names[tc.get("id")] = (tc.get("function") or {}).get("name")
    for m in reversed(msgs):
        if not (isinstance(m, dict) and m.get("role") == "tool"):
            break
        c = m.get("content")
        if not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False)
        tid = m.get("tool_call_id", "")
        results.insert(0, {"tool_call_id": tid, "name": m.get("name") or names.get(tid) or "", "content": c})
    return client_tools, (results or None)


# "hands" rides the same channel so a background job's approval wait and its
# completion reach every client, not only the live SSE channel.
KERNEL_EVENTS = frozenset({"beat", "hold", "silent", "verify", "steer", "interrupt", "floor", "speaker",
                           "todo", "trim", "parts", "hands", "shared_timeline"})


def _lobes(settings, header, body) -> bool | None:
    """The mind is the two lobes.

    When the server default is on, a request cannot turn that off. When this
    process was started with SAWII_DUAL=0, the header still selects the path
    so the suite can measure the older speak() path.
    """
    if settings.dual:
        return True
    return _flag(header, body)


def _flag(header, body) -> bool | None:
    """'1'/'0' header (or a JSON bool in the body) -> True/False; absent -> None (use the default)."""
    for raw in (header, body):
        if raw is None or raw == "":
            continue
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return None


def _surface(request) -> str:
    """Where they are talking from (device/app). Provenance only - never memory content."""
    import re as _re
    parts = [request.headers.get(h, "") for h in ("x-dl-surface-device", "x-dl-surface-app")]
    raw = " / ".join(p for p in parts if p) or request.headers.get("x-surface", "")
    return _re.sub(r"[^\w .:/@+-]", "", raw)[:120]


def _clean_contact(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    return {k: str(v)[:200] for k, v in raw.items() if k in ("sms", "whatsapp", "email", "telegram") and v}


def _channel(v) -> str:
    return v if v in CHANNELS else "app"


def main_app() -> Starlette:  # uvicorn sawii_persona.api.app:main_app --factory
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()



