"""Stitch — wire every Hands element into one running app.

This is the entry point. It:

1. Starts Persona (the mind) as the central server.
2. Spawns one sawii_hands.py per R-lane as Persona delegates work.
3. Registers the Link adapter so Persona can connect to any external agent.
4. Exposes the combined API: Persona's own routes + the desktop shell websocket.

    python stitch.py --persona-url https://your-persona --persona-key sp_...
                     --root ~/Projects --engine-preset presets/ui-tars-deepinfra.yaml

Or, for a fully local setup (Persona runs inside this process):

    python stitch.py --local --root ~/Projects --deepinfra-key di-...
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


async def run_local(args) -> None:
    """Start Persona inside this process and auto-connect the hands engine."""
    # Import Persona's own server factory.
    sys.path.insert(0, str(Path(__file__).parent))
    from persona.config import load_settings
    from persona.api.app import Runtime, create_app
    from persona.link import LinkRegistry

    # Override settings from CLI flags.
    os.environ.setdefault("SAWII_STORE_PATH", str(Path(args.data_dir) / "sawii.db"))
    if args.deepinfra_key:
        os.environ.setdefault("DEEPINFRA_API_KEY", args.deepinfra_key)

    settings = load_settings()
    rt = Runtime(settings)
    app = create_app(settings, rt)

    # Register the Link adapter on the mind for external agent connections.
    links = LinkRegistry()
    rt.mind.links = links

    # Register the sawii_hands engine as a spawn-able agent.
    engine_path = str(Path(__file__).parent / "sawii_hands.py")
    links.register({
        "name": "hands-engine",
        "binary": engine_path,
        "persona_url": f"http://127.0.0.1:{args.port}",
        "persona_key": args.persona_key or "",
        "roots": args.root or [],
        "approve": args.approve,
        "place": "computer",
    })

    # Hook into the lane book: when Persona delegates to an R-lane and there's
    # no host tool for it, spawn a hands engine instance for that lane.
    original_start = rt.mind.lanes.start

    async def lane_start_with_spawn(pid, topic, instruction):
        line = await original_start(pid, topic, instruction)
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

    # Start the Starlette/ASGI server.
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="info")
    server = uvicorn.Server(config)

    # Graceful shutdown.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(server, links)))

    log.info("Hands starting on http://0.0.0.0:%d", args.port)
    log.info("Engine: %s", engine_path)
    log.info("Roots: %s", ", ".join(args.root) if args.root else "(none — add --root)")
    await server.serve()


async def run_remote(args) -> None:
    """Connect to an existing Persona server and register the hands engine."""
    sys.path.insert(0, str(Path(__file__).parent))
    from persona.link import LinkRegistry

    links = LinkRegistry()

    engine_path = str(Path(__file__).parent / "sawii_hands.py")

    # Register and immediately start the engine as a persistent worker.
    link = links.register({
        "name": "hands-engine",
        "binary": engine_path,
        "persona_url": args.persona_url,
        "persona_key": args.persona_key,
        "roots": args.root or [],
        "approve": args.approve,
        "place": "computer",
    })

    # Start one persistent engine connected to Persona.
    from sawii_hands import Hands, serve
    hands = Hands(args.root or ["."], args.approve)

    log.info("Connecting hands engine to %s", args.persona_url)
    await serve(args.persona_url, args.persona_key, hands, place="computer")


async def _shutdown(server, links):
    log.info("Shutting down…")
    await links.shutdown_all()
    server.should_exit = True


def main():
    ap = argparse.ArgumentParser(
        description="Stitch — wire Hands elements into one running app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode")

    # Local mode: run Persona + engine in one process.
    local = sub.add_parser("local", help="Run Persona locally with auto-spawned engines")
    local.add_argument("--port", type=int, default=8800)
    local.add_argument("--root", action="append", default=[],
                       help="Folders the engine may access (repeat for more)")
    local.add_argument("--approve", choices=["ask", "writes", "auto"], default="ask")
    local.add_argument("--deepinfra-key", default=os.environ.get("DEEPINFRA_API_KEY", ""))
    local.add_argument("--persona-key", default=os.environ.get("SAWII_PERSONA_KEY", ""))
    local.add_argument("--data-dir", default=str(Path.home() / ".hands"),
                       help="Where to store the database and files")

    # Remote mode: connect to an existing Persona server.
    remote = sub.add_parser("remote", help="Connect engine to a remote Persona")
    remote.add_argument("--persona-url", required=True)
    remote.add_argument("--persona-key", required=True)
    remote.add_argument("--root", action="append", default=[],
                        help="Folders the engine may access (repeat for more)")
    remote.add_argument("--approve", choices=["ask", "writes", "auto"], default="ask")

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
