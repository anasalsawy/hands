"""Stitch -- wire every Hands element into one running app.

This is the entry point. It:

1. Starts Persona (the mind) as the central server.
2. Exposes the combined API: Persona's own routes + the desktop shell websocket.
3. Wires live voice for testing -- reuses Persona ``/v1/voice`` + the existing
   ``voice-module`` sidecar. Nothing voice-related is rebuilt.

Persona is the mind; Agent TARS is the execution engine. Each R-lane the mind opens
gets its own Agent TARS engine (``engine/persona-loop.ts``, a child process that has
no model and no provider key). Persona supplies inference, the plan tool, approvals
and observation; TARS executes every browser, filesystem, terminal and search action.

    python stitch.py local --root ~/Projects --deepinfra-key di-... --port 8800

``--deepinfra-key`` is PERSONA's key (the L/R lobes in ``persona/llm.py``), never the
engine's. ``presets/`` configures the *standalone* UI-TARS Desktop app, a different
program.

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


def _install_shutdown_handlers(server, voice_sidecar=None) -> None:
    """Register graceful shutdown. Windows has no asyncio signal handlers."""
    loop = asyncio.get_running_loop()

    def _request_shutdown() -> None:
        asyncio.create_task(_shutdown(server, voice_sidecar))

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
    key_file = Path(args.data_dir) / "local_key"

    if not persona_key and key_file.exists():
        # Reuse the persona this machine already created, so the app reconnects
        # on its own after a restart instead of asking for a key it can't show twice.
        try:
            saved = key_file.read_text().strip()
            from persona.identity import persona_by_key
            row = await persona_by_key(rt.mind.store, saved) if saved else None
            if row:
                persona_key, persona_id = saved, int(row["id"])
                log.info("Reusing local persona id=%d", persona_id)
        except Exception:  # noqa: BLE001
            log.debug("saved local key unusable; creating a new persona", exc_info=True)

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
            try:
                key_file.parent.mkdir(parents=True, exist_ok=True)
                key_file.write_text(persona_key)
                try:
                    key_file.chmod(0o600)
                except OSError:
                    pass
            except OSError:
                log.warning("could not save the local persona key to %s", key_file)
        except Exception:
            # Table may not exist yet or DB might be read-only on first run.
            # Fall back to a generated key — the hands_ws endpoint
            # will try to look it up and fail, but the engine can still
            # connect once a persona is created through the admin API.
            persona_key = new_key()
            log.warning("Could not create local persona; using generated key %s…%s "
                        "(engines will not connect until a persona exists)",
                        persona_key[:7], persona_key[-4:])

    # ── Engine configuration ──────────────────────────────────────────
    # Agent TARS runs inside each lane's work (persona/court.py::_drive); it only
    # needs to know where it may work and how Persona gates its actions.
    if args.root:
        os.environ["HANDS_ROOT"] = os.pathsep.join(os.path.abspath(r) for r in args.root)
    os.environ.setdefault("SAWII_HANDS_APPROVE", args.approve)
    links = None

    # ── Link + voice status endpoints ─────────────────────────────────
    from starlette.responses import JSONResponse
    from starlette.routing import Route

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
    app.routes.insert(0, Route("/v1/voice-status", voice_status_view))
    extra = getattr(args, "extra_routes", None)
    if extra is not None:
        for r in extra(app=app, rt=rt, settings=settings, persona_key=persona_key,
                       persona_id=persona_id, links=links):
            app.routes.insert(0, r)

    serve_app = app

    # ── Start the Starlette/ASGI server ───────────────────────────────
    import uvicorn
    config = uvicorn.Config(serve_app, host=getattr(args, "host", "127.0.0.1"), port=args.port, log_level="info")
    server = uvicorn.Server(config)

    _install_shutdown_handlers(server, voice_sidecar)

    log.info("─── Hands stitched ───")
    log.info("  Persona:    http://%s:%d", getattr(args, "host", "127.0.0.1"), args.port)
    log.info("  Engine:     Agent TARS (model-less: Persona's R lobe does the thinking)")
    log.info("  Roots:      %s", ", ".join(args.root) if args.root else "(none — add --root)")
    log.info("  Key:        %s…  (hidden)", persona_key[:6])
    log.info("  Courts:     http://0.0.0.0:%d/v1/courts  (the deck of concurrent desktop cards)", args.port)
    log.info("  Voice UI:   http://0.0.0.0:%d/app  (mic → /v1/voice)", args.port)
    log.info("  Voice diag: http://0.0.0.0:%d/v1/voice-status", args.port)
    if voice_sidecar.url:
        log.info("  Voice sand: %s/health", voice_sidecar.url)
    log.info("──────────────────────")
    await server.serve()


# ═══════════════════════════════════════════════════════════════════════════

async def _shutdown(server, voice_sidecar=None):
    log.info("Shutting down…")
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
        prog="stitch",
        description="Hands: Persona (the mind) with Agent TARS (the execution engine) under it.",
    )
    sub = ap.add_subparsers(dest="mode")

    local = sub.add_parser("local", help="Run Persona locally; each R-lane gets its own Agent TARS engine")
    local.add_argument("--port", type=int, default=8800)
    local.add_argument("--host", default="127.0.0.1",
                       help="Interface to listen on (default loopback; the deck can drive your computer)")
    local.add_argument("--root", action="append", default=[],
                       help="Folders the engine may access (repeat for more)")
    local.add_argument("--approve", choices=["ask", "writes", "auto"], default="ask")
    local.add_argument("--deepinfra-key", default=os.environ.get("DEEPINFRA_API_KEY", ""),
                       help="PERSONA's (the mind's) DeepInfra key -- seeds DEEPINFRA_API_KEY for the "
                            "L/R lobes in persona/llm.py. NOT an engine key: Agent TARS has no model.")
    local.add_argument("--persona-key", default=os.environ.get("SAWII_PERSONA_KEY", ""))
    local.add_argument("--data-dir", default=str(Path.home() / ".hands"),
                       help="Where to store the database and files")
    local.add_argument("--no-voice", action="store_true",
                       help="Do not start the voice-module sidecar (Persona /v1/voice still available)")

    args = ap.parse_args()
    if args.mode is None:
        ap.print_help()
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(run_local(args))


if __name__ == "__main__":
    main()
