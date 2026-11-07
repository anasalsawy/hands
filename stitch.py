"""Stitch — wire every Hands element into one running app.

This is the entry point. It:

1. Starts Persona (the mind) as the central server.
2. Spawns one sawii_hands.py per R-lane as Persona delegates work.
3. Registers the Link adapter so Persona can connect to any external agent.
4. Exposes the combined API: Persona's own routes + the desktop shell websocket.

Local mode (Persona + engine in one process):

    python stitch.py local --root ~/Projects --deepinfra-key di-... --port 8800

Remote mode (connect to an existing Persona server):

    python stitch.py remote --persona-url https://your-persona --persona-key sp_...
                     --root ~/Projects --engine-preset presets/ui-tars-deepinfra.yaml

Register an external agent via API bridge:

    python stitch.py local --root ~/Projects --deepinfra-key di-... \\
        --agent browser-use:http://localhost:7860:YOUR_KEY:model-name \\
        --agent openai-helper:https://api.openai.com/v1:sk-...:gpt-4o
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


# ═══════════════════════════════════════════════════════════════════════════
# Local mode: Persona + engine in one process
# ═══════════════════════════════════════════════════════════════════════════

async def run_local(args) -> None:
    """Start Persona inside this process and auto-connect the hands engine."""
    # Import Persona's own server factory.
    sys.path.insert(0, str(Path(__file__).parent))
    from persona.config import load_settings
    from persona.api.app import Runtime, create_app
    from persona.link import LinkRegistry
    from persona.identity import create_persona, new_key

    # Override settings from CLI flags.
    os.environ.setdefault("SAWII_STORE_PATH", str(Path(args.data_dir) / "sawii.db"))
    if args.deepinfra_key:
        os.environ.setdefault("DEEPINFRA_API_KEY", args.deepinfra_key)

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
            log.info("Created local persona id=%d key=%s…%s",
                     persona_id, persona_key[:7], persona_key[-4:])
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

    # ── Hook: auto-spawn engine for lanes with no host delegation ─────
    original_start = rt.mind.lanes.start

    async def lane_start_with_spawn(pid, topic, instruction):
        line = await original_start(pid, topic, instruction)

        # Update the persona_id on the registry if not yet set, now that
        # we have a real pid from a real request.
        if links._persona_id == 0 and pid:
            links.set_hub(rt.mind.hands, pid)

        # If no host is running tools for this line, spawn an engine.
        if not line.delegated:
            engine_link = links.get("hands-engine")
            if engine_link:
                try:
                    await engine_link.delegate(line.id, instruction)
                    line.delegated = True
                    line.host_status = "spawned"
                    await rt.mind.lanes.save(pid, line)
                    log.info("Auto-spawned hands engine for lane %s (%s)", line.label, line.id)
                except Exception:
                    log.exception("Failed to spawn engine for lane %s", line.id)
        return line

    rt.mind.lanes.start = lane_start_with_spawn

    # ── Link status endpoint ──────────────────────────────────────────
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def link_status(request):
        return JSONResponse({"links": links.info()})

    # Inject our route into the app.
    app.routes.insert(0, Route("/v1/link", link_status))

    # ── Start the Starlette/ASGI server ───────────────────────────────
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="info")
    server = uvicorn.Server(config)

    # Graceful shutdown.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(server, links)))

    log.info("─── Hands stitched ───")
    log.info("  Persona:    http://0.0.0.0:%d", args.port)
    log.info("  Engine:     %s", engine_path)
    log.info("  Roots:      %s", ", ".join(args.root) if args.root else "(none — add --root)")
    log.info("  Agents:     %s", ", ".join(links.strategies()) or "(engine only)")
    log.info("  Key:        %s…%s", persona_key[:7], persona_key[-4:])
    log.info("  Link info:  http://0.0.0.0:%d/v1/link", args.port)
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

    config = {
        "name": name,
        "api_url": api_url,
        "api_key": api_key,
        "model": model,
        "agent_type": agent_type,
    }

    link = links.register(config)
    log.info("Registered external agent: %s (%s via %s)",
             name, link.agent.strategy, api_url)


# ═══════════════════════════════════════════════════════════════════════════
# Shutdown
# ═══════════════════════════════════════════════════════════════════════════

async def _shutdown(server, links):
    log.info("Shutting down…")
    await links.shutdown_all()
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
  python stitch.py remote --persona-url https://my-persona --persona-key sp_... --root ~/Projects
""",
    )
    sub = ap.add_subparsers(dest="mode")

    # ── Local mode ────────────────────────────────────────────────────
    local = sub.add_parser("local", help="Run Persona locally with auto-spawned engines")
    local.add_argument("--port", type=int, default=8800)
    local.add_argument("--root", action="append", default=[],
                       help="Folders the engine may access (repeat for more)")
    local.add_argument("--approve", choices=["ask", "writes", "auto"], default="ask")
    local.add_argument("--deepinfra-key", default=os.environ.get("DEEPINFRA_API_KEY", ""))
    local.add_argument("--persona-key", default=os.environ.get("SAWII_PERSONA_KEY", ""))
    local.add_argument("--data-dir", default=str(Path.home() / ".hands"),
                       help="Where to store the database and files")
    local.add_argument("--agent", action="append", default=[],
                       help="Register an external agent (name:api_url:api_key:model[:type]). "
                            "Repeat for multiple agents.")

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
