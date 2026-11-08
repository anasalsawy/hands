"""Stitch — wire every Hands element into one running app.

This is the entry point. It:

1. Starts Persona (the mind) as the central server.
2. Spawns one sawii_hands.py per R-lane as Persona delegates work.
3. Registers the Link adapter so Persona can connect to any external agent.
4. Exposes the combined API: Persona's own routes + the desktop shell websocket.
5. Wires live voice for testing — reuses Persona ``/v1/voice`` + the existing
   ``voice-module`` sidecar (STT/TTS/barge-in). Nothing voice-related is rebuilt.

Local mode (Persona + engine in one process):

    python stitch.py local --root ~/Projects --deepinfra-key di-... --port 8800

Remote mode (connect to an existing Persona server):

    python stitch.py remote --persona-url https://your-persona --persona-key sp_...
                     --root ~/Projects

The engine has no model and no provider key. ``sawii_hands.py`` is a pure tool
executor: it advertises its tools over the ``/v1/hands`` WebSocket, and Persona's
R-lane emits every tool call and reads every result back. For screen work the R
asks for ``ocu_get_app_state`` and picks the next click itself, so the VLM *is*
R. There is deliberately no ``--engine-preset`` and no engine API-key flag: the
only key that reaches this process is PERSONA's own ``--deepinfra-key`` (below)
and the persona key the engine authenticates with. ``presets/`` configures a
*standalone* UI-TARS Desktop operator, which is a different program.

Register an external agent via API bridge:

    python stitch.py local --root ~/Projects --deepinfra-key di-... \\
        --agent browser-use:http://localhost:7860:YOUR_KEY:model-name \\
        --agent openai-helper:https://api.openai.com/v1:sk-...:gpt-4o

``--deepinfra-key`` is PERSONA's key, not the engine's. It seeds
``DEEPINFRA_API_KEY`` for the mind (the L/R lobes in ``persona/llm.py``) and
nothing else. ``sawii_hands.py`` has no provider imports at all, and the spawn
path in ``persona/link.py`` hands the engine only ``--url``, ``--key`` (the
persona key), ``--root``, ``--approve``, ``--place`` and ``--lane``.

Live voice (local mode):

    Open http://127.0.0.1:8800/app  → mic button uses Persona /v1/voice
    GET  http://127.0.0.1:8800/v1/voice-status  → wiring + sidecar health
    Sidecar (if voice-module found): http://127.0.0.1:18766/health
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

log = logging.getLogger("hands.stitch")

HANDS_ROOT = Path(__file__).resolve().parent


def _load_workspace_env() -> None:
    """Pull keys from the parent sawii-persona workspace when present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (
        HANDS_ROOT.parent / "keys.env",
        HANDS_ROOT.parent / ".env",
        HANDS_ROOT / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _install_shutdown_handlers(server, links, voice_sidecar=None) -> None:
    """Register graceful shutdown. Windows has no asyncio signal handlers."""
    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        asyncio.create_task(_shutdown(server, links, voice_sidecar))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            # ProactorEventLoop on Windows — fall back to signal.signal.
            try:
                signal.signal(
                    sig,
                    lambda *_a, _s=sig: loop.call_soon_threadsafe(_request_shutdown),
                )
            except (ValueError, OSError) as exc:
                log.warning("Could not install %s handler (%s); use Ctrl+C via uvicorn", sig, exc)
            break


# ═══════════════════════════════════════════════════════════════════════════
# Local mode: Persona + engine in one process
# ═══════════════════════════════════════════════════════════════════════════

async def run_local(args) -> None:
    """Start Persona inside this process and auto-connect the hands engine."""
    _load_workspace_env()
    # Import Persona's own server factory.
    sys.path.insert(0, str(HANDS_ROOT))
    from persona.config import load_settings
    from persona.api.app import Runtime, create_app
    from persona.link import LinkRegistry
    from persona.identity import create_persona, new_key
    from voice_sidecar import VoiceSidecar, voice_status

    # Override settings from CLI flags.
    os.environ.setdefault("SAWII_STORE_PATH", str(Path(args.data_dir) / "sawii.db"))
    # PERSONA's key, for the mind. This is the L/R lobe credential (persona/llm.py
    # reads DEEPINFRA_API_KEY). It is NOT an engine credential: sawii_hands.py has
    # no provider key and no LLM of its own — it thinks through R over /v1/hands.
    if args.deepinfra_key:
        os.environ.setdefault("DEEPINFRA_API_KEY", args.deepinfra_key)
    # Local Hands default: Featherless OpenAI-compatible embeddings.
    # Default provider is Gemini; a bad/expired GEMINI_API_KEY 401s, embedder
    # falls back to local-hash, then mind.require_initial_semantic raises
    # "Required initial semantic embedding unavailable" and kills phone +
    # /v1/voice turns. Prefer Featherless when the caller hasn't chosen.
    if not os.environ.get("SAWII_EMBEDDING_PROVIDER"):
        os.environ["SAWII_EMBEDDING_PROVIDER"] = "featherless"
    # Map SAWII_FEATHERLESS_API_KEY into EMBED_API_KEY if unset.
    if os.environ.get("SAWII_FEATHERLESS_API_KEY") and not os.environ.get("SAWII_EMBED_API_KEY"):
        os.environ["SAWII_EMBED_API_KEY"] = os.environ["SAWII_FEATHERLESS_API_KEY"]


    voice_sidecar = VoiceSidecar()
    if not getattr(args, "no_voice", False):
        side = voice_sidecar.start()
        if side.get("ok"):
            log.info("Voice sidecar: %s (ready=%s)", side.get("url"), side.get("ready"))
        else:
            log.warning(
                "Voice sidecar not started (%s) — Persona /v1/voice + /app mic still available",
                side.get("error"),
            )

    settings = load_settings()
    rt = Runtime(settings)
    app = create_app(settings, rt)

    # ── Persona key for local mode ────────────────────────────────────
    # In local mode we need a real persona key for spawned engines to
    # authenticate with the /v1/hands WebSocket. Generate one if none
    # was provided on the CLI or in the environment.
    persona_key = args.persona_key or os.environ.get("SAWII_PERSONA_KEY", "")
    persona_id = 0

    if not persona_key:
        # Create a default persona so spawned engines can auth.
        try:
            persona_id, persona_key = await create_persona(
                rt.mind.store,
                name="local",
                definition="Local Persona started by stitch.py",
                preset="default",
                user_name=os.environ.get("USER", "user"),
            )
            # The full key must be printed: only its hash is stored, so this line
            # is the ONE copy of the credential. The desktop's ConnectView tells
            # the user to copy it from here ("Created local persona id=… key=…");
            # an elided hint here makes the deck unreachable (every /v1/courts
            # call 401s and /v1/court-events closes straight away with 4401).
            log.info("Created local persona id=%d key=%s", persona_id, persona_key)
        except Exception:
            # Table may not exist yet or DB might be read-only on first run.
            # Fall back to a generated key — the hands_ws endpoint
            # will try to look it up and fail, but the engine can still
            # connect once a persona is created through the admin API.
            persona_key = new_key()
            log.warning("Could not create local persona; using generated key %s…%s "
                        "(engines will not connect until a persona exists)",
                        persona_key[:7], persona_key[-4:])

    # ── LinkRegistry ──────────────────────────────────────────────────
    links = LinkRegistry()
    rt.mind.links = links

    # Wire hub to registry so API bridge drivers can execute tools.
    # The hub lives on the mind; persona_id may be 0 if we only have a
    # key but haven't resolved the id yet — set_hub accepts 0 and the
    # bridge simply can't execute hub tools until a real id is set.
    links.set_hub(rt.mind.hands, persona_id)

    # ── Register the built-in sawii_hands engine ──────────────────────
    engine_path = str(Path(__file__).parent / "sawii_hands.py")
    links.register({
        "name": "hands-engine",
        "binary": engine_path,
        "persona_url": f"http://127.0.0.1:{args.port}",
        "persona_key": persona_key,
        "roots": args.root or [],
        "approve": args.approve,
        "place": "computer",
    })

    # ── Register any external agents from --agent flags ───────────────
    for agent_spec in (args.agent or []):
        _register_agent_from_spec(links, agent_spec)

    # ── Hook: every R-lane the mind opens gets its own engine ─────────
    # The design rule: the user states a goal, the mind splits it, and each
    # R-line AUTOMATICALLY gets its own sawii_hands.py. No client calls a spawn
    # endpoint; concurrency is the mind's decision, not a UI control.
    original_start = rt.mind.lanes.start
    spawn_tasks: set[asyncio.Task] = set()

    def _spawn_for_line(pid: int, line) -> None:
        """Put exactly one engine behind one lane, and record the truth on it.

        Runs off the request path, because ``Link._spawn_delegate`` now waits
        until the engine shows up on /v1/hands and that must not stall the turn
        which opened the line. The task is held in ``spawn_tasks`` on purpose:
        the loop only keeps a weak reference to a bare ``create_task``, so an
        unheld spawn can be garbage collected mid-flight and the lane silently
        ends up with no motor and no error — the exact bug this fixes.
        """

        async def _run() -> None:
            engine_link = links.get("hands-engine")
            if engine_link is None:
                line.host_status = "failed: no hands-engine Link registered"
                await rt.mind.lanes.save(pid, line)
                log.error("Lane %s has no registered engine to spawn", line.id)
                return
            try:
                linked = await engine_link.delegate(
                    line.id, line.instruction or line.topic,
                    persona_id=pid, persona_key=links.persona_key(pid))
                if getattr(linked, "status", "") == "failed":
                    line.host_status = (
                        f"failed: {getattr(linked, 'error', '') or 'engine died'}")[:200]
                    await rt.mind.lanes.save(pid, line)
                    log.error("Engine for lane %s (%s) failed: %s",
                              line.label, line.id, line.host_status)
                    return
                line.delegated = True
                line.host_status = "spawned"
                await rt.mind.lanes.save(pid, line)
                log.info("Auto-spawned hands engine for lane %s (%s)", line.label, line.id)
            except Exception as exc:  # noqa: BLE001 - say so, never hang silently
                line.host_status = f"failed: {type(exc).__name__}: {exc}"[:200]
                try:
                    await rt.mind.lanes.save(pid, line)
                except Exception:  # noqa: BLE001
                    log.debug("could not persist failed lane %s", line.id, exc_info=True)
                log.exception("Failed to spawn engine for lane %s", line.id)

        task = asyncio.create_task(_run(), name=f"engine-{line.id}")
        spawn_tasks.add(task)
        task.add_done_callback(spawn_tasks.discard)

    async def lane_start_with_spawn(pid, topic, instruction):
        # Read the caller's key while still inside its request context, so the
        # engine we start can authenticate as the persona that asked.
        key = request_key.get() or ""
        if key and pid:
            links.remember_persona_key(pid, key)
        line = await original_start(pid, topic, instruction)

        # Update the persona_id on the registry if not yet set, now that
        # we have a real pid from a real request.
        if links._persona_id == 0 and pid:
            links.set_hub(rt.mind.hands, pid)

        # If no host is running tools for this line, give it its own engine.
        if not line.delegated:
            _spawn_for_line(pid, line)
        return line

    rt.mind.lanes.start = lane_start_with_spawn

    # ── Link + voice status endpoints ─────────────────────────────────
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def link_status(request):
        # Reconcile with the hub first: ``active_lanes`` must list exactly the
        # runners /v1/hands can see, never a hand that died or never checked in.
        await links.observe_hub()
        return JSONResponse({"links": links.info()})

    async def voice_status_view(request):
        payload = voice_status()
        payload["sidecar"] = {
            "running": voice_sidecar.running,
            "url": voice_sidecar.url,
            "pid": voice_sidecar.proc.pid if voice_sidecar.proc else None,
        }
        # Persona speech capability from the already-running Runtime.
        speech = getattr(rt, "speech", None)
        s = getattr(rt, "s", None) or settings
        mind = getattr(rt, "mind", None)
        try:
            from persona.voice.vocode import pipeline_status
            pipeline = pipeline_status(mind, speech) if mind is not None else {}
        except Exception as exc:  # noqa: BLE001
            pipeline = {"error": type(exc).__name__}
        payload["persona_speech"] = {
            "asr": pipeline.get("asr") or (
                "native" if getattr(s, "native_audio_default", False) else
                ("server" if speech and speech.has_asr else "client")),
            "tts": bool(getattr(speech, "has_tts", False)) if speech else False,
            "native": pipeline.get("native") or {
                "enabled": bool(getattr(s, "native_audio_default", False)),
                "model": getattr(getattr(s, "voice_native", None), "model", None),
            },
            "tts_model": getattr(getattr(s, "tts", None), "model", None),
            "dual_listen": True,
            "shared_mind": "DualLoop",
            "architecture": "vocode-faithful",
            "websocket": f"ws://127.0.0.1:{args.port}/v1/voice",
            "test_ui": f"http://127.0.0.1:{args.port}/app",
        }
        payload["pipeline"] = pipeline
        return JSONResponse(payload)

    # Inject our routes into the app (Persona already owns /v1/voice).
    app.routes.insert(0, Route("/v1/link", link_status))
    app.routes.insert(0, Route("/v1/voice-status", voice_status_view))

    # ── Which key is this persona using? ──────────────────────────────
    # A spawned engine registers on /v1/hands under whatever key it connects
    # with, and the hub keys runners by (persona, place, lane). So an engine
    # started for persona B must connect with B's key — with the boot persona's
    # key it lands in another persona's slot and the lane has no hand at all.
    #
    # The key is read off a middleware layer wrapped around the ASGI *callable*
    # at serve time (never reassigning ``app``, which uvicorn reads positionally
    # as its app string), and it is only ever what the caller already sent us:
    # nothing is persisted, and the store keeps only a hash.
    from contextvars import ContextVar
    request_key: ContextVar[str] = ContextVar("sawii_request_key", default="")

    async def remember_key_app(scope, receive, send):
        if scope.get("type") not in ("http", "websocket"):
            await app(scope, receive, send)
            return
        auth = ""
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                auth = value.decode("latin-1")
                break
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        token = request_key.set(key)
        try:
            await app(scope, receive, send)
        finally:
            request_key.reset(token)

    serve_app = remember_key_app

    # ── Start the Starlette/ASGI server ───────────────────────────────
    import uvicorn
    config = uvicorn.Config(serve_app, host="0.0.0.0", port=args.port, log_level="info")
    server = uvicorn.Server(config)

    _install_shutdown_handlers(server, links, voice_sidecar)

    log.info("─── Hands stitched ───")
    log.info("  Persona:    http://0.0.0.0:%d", args.port)
    log.info("  Engine:     %s  (model-less: no key, no LLM — thinks through R over /v1/hands)",
             engine_path)
    log.info("  Roots:      %s", ", ".join(args.root) if args.root else "(none — add --root)")
    log.info("  Agents:     %s", ", ".join(links.strategies()) or "(engine only)")
    log.info("  Key:        %s", persona_key)
    log.info("  Link info:  http://0.0.0.0:%d/v1/link", args.port)
    log.info("  Courts:     http://0.0.0.0:%d/v1/courts  (the deck of concurrent desktop cards)", args.port)
    log.info("  Voice UI:   http://0.0.0.0:%d/app  (mic → /v1/voice)", args.port)
    log.info("  Voice diag: http://0.0.0.0:%d/v1/voice-status", args.port)
    if voice_sidecar.url:
        log.info("  Voice sand: %s/health", voice_sidecar.url)
    log.info("──────────────────────")
    await server.serve()


# ═══════════════════════════════════════════════════════════════════════════
# Remote mode: connect to an existing Persona server
# ═══════════════════════════════════════════════════════════════════════════

async def run_remote(args) -> None:
    """Connect to an existing Persona server and register the hands engine."""
    sys.path.insert(0, str(Path(__file__).parent))
    from persona.link import LinkRegistry

    links = LinkRegistry()

    engine_path = str(Path(__file__).parent / "sawii_hands.py")

    # Register the engine so we know what it is.
    links.register({
        "name": "hands-engine",
        "binary": engine_path,
        "persona_url": args.persona_url,
        "persona_key": args.persona_key,
        "roots": args.root or [],
        "approve": args.approve,
        "place": "computer",
    })

    # Register any external agents.
    for agent_spec in (args.agent or []):
        _register_agent_from_spec(links, agent_spec)

    # Start one persistent engine connected to Persona.
    from sawii_hands import Hands, serve
    hands = Hands(args.root or ["."], args.approve)

    log.info("─── Hands connecting ───")
    log.info("  Persona:  %s", args.persona_url)
    log.info("  Engine:   %s", engine_path)
    log.info("  Roots:    %s", ", ".join(args.root) if args.root else "(none — add --root)")
    log.info("  Agents:   %s", ", ".join(links.strategies()) or "(engine only)")
    log.info("────────────────────────")
    await serve(args.persona_url, args.persona_key, hands, place="computer")


# ═══════════════════════════════════════════════════════════════════════════
# Agent registration from CLI --agent flag
# ═══════════════════════════════════════════════════════════════════════════

def _register_agent_from_spec(links, spec: str) -> None:
    """Parse an --agent spec and register it.

    Format: name:api_url:api_key:model[:agent_type]
    Example: browser-use:http://localhost:7860:MY_KEY:gpt-4o:browser_use
             openai-helper:https://api.openai.com/v1:sk-...:gpt-4o

    The API key can be an env var reference like $OPENAI_API_KEY.
    """
    parts = spec.split(":")
    if len(parts) < 3:
        log.warning("Ignoring bad --agent spec (need name:url:key[:model[:type]]): %s", spec)
        return

    name = parts[0]
    # Rejoin URL parts that were split on the colon in http://
    if len(parts) >= 4 and parts[1] in ("http", "https"):
        api_url = f"{parts[1]}:{parts[2]}"
        rest = parts[3:]
    else:
        api_url = parts[1]
        rest = parts[2:]

    api_key = rest[0] if len(rest) > 0 else ""
    model = rest[1] if len(rest) > 1 else ""
    agent_type = rest[2] if len(rest) > 2 else "openai"

    # Resolve env var references in the key.
    if api_key.startswith("$"):
        api_key = os.environ.get(api_key[1:], "")

    config: dict = {
        "name": name,
        "api_key": api_key,
        "model": model,
        "agent_type": agent_type,
    }

    # Native drivers (e.g. browser_use_native) don't need an API URL —
    # they import the library directly. api_url is optional for them.
    if agent_type == "browser_use_native":
        if api_url:
            config["base_url"] = api_url  # optional LLM endpoint override
    else:
        config["api_url"] = api_url

    link = links.register(config)
    log.info("Registered external agent: %s (%s via %s)",
             name, link.agent.strategy, api_url or "(native in-process)")


# ═══════════════════════════════════════════════════════════════════════════
# Shutdown
# ═══════════════════════════════════════════════════════════════════════════

async def _shutdown(server, links, voice_sidecar=None):
    log.info("Shutting down…")
    await links.shutdown_all()
    if voice_sidecar is not None:
        try:
            voice_sidecar.stop()
        except Exception:
            log.exception("voice sidecar stop failed")
    server.should_exit = True


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stitch — wire Hands elements into one running app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stitch.py local --root ~/Projects --deepinfra-key di-...
  python stitch.py local --root ~/Projects --agent browser-use:http://localhost:7860:$KEY:model
  python stitch.py local --root ~/Projects --agent bu-native:_:$OPENAI_API_KEY:gpt-4o:browser_use_native
  python stitch.py remote --persona-url https://my-persona --persona-key sp_... --root ~/Projects

Native browser-use (browser-use as in-process library, not API bridge):
  --agent bu:_:$OPENAI_API_KEY:gpt-4o-mini:browser_use_native
  The _ placeholder means no API URL is needed — browser-use runs in-process.
  The api_key is the LLM key (OpenAI/Anthropic), NOT a browser-use cloud key.
""",
    )
    sub = ap.add_subparsers(dest="mode")

    # ── Local mode ────────────────────────────────────────────────────
    local = sub.add_parser("local", help="Run Persona locally with auto-spawned engines")
    local.add_argument("--port", type=int, default=8800)
    local.add_argument("--root", action="append", default=[],
                       help="Folders the engine may access (repeat for more)")
    local.add_argument("--approve", choices=["ask", "writes", "auto"], default="ask")
    local.add_argument("--deepinfra-key", default=os.environ.get("DEEPINFRA_API_KEY", ""),
                       help="PERSONA's (the mind's) DeepInfra key — seeds DEEPINFRA_API_KEY for the "
                            "L/R lobes in persona/llm.py. NOT an engine key: sawii_hands.py has no "
                            "LLM and gets its thinking from R over /v1/hands.")
    local.add_argument("--persona-key", default=os.environ.get("SAWII_PERSONA_KEY", ""))
    local.add_argument("--data-dir", default=str(Path.home() / ".hands"),
                       help="Where to store the database and files")
    local.add_argument("--agent", action="append", default=[],
                       help="Register an external agent (name:api_url:api_key:model[:type]). "
                            "Repeat for multiple agents.")
    local.add_argument(
        "--no-voice",
        action="store_true",
        help="Do not start the voice-module sidecar (Persona /v1/voice still available)",
    )

    # ── Remote mode ───────────────────────────────────────────────────
    remote = sub.add_parser("remote", help="Connect engine to a remote Persona")
    remote.add_argument("--persona-url", required=True)
    remote.add_argument("--persona-key", required=True)
    remote.add_argument("--root", action="append", default=[],
                        help="Folders the engine may access (repeat for more)")
    remote.add_argument("--approve", choices=["ask", "writes", "auto"], default="ask")
    remote.add_argument("--agent", action="append", default=[],
                        help="Register an external agent (name:api_url:api_key:model[:type])")

    args = ap.parse_args()
    if args.mode is None:
        ap.print_help()
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.mode == "local":
        asyncio.run(run_local(args))
    else:
        asyncio.run(run_remote(args))


if __name__ == "__main__":
    main()
