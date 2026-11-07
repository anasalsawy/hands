"""Command line: serve, create a persona, run a sleep cycle by hand."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .config import load_settings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sawii-persona")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the API + always-on heartbeat")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8800)
    c = sub.add_parser("create-persona", help="create a persona and print its key")
    c.add_argument("--name", default="")
    c.add_argument("--preset", default="")
    c.add_argument("--definition", default="")
    c.add_argument("--user-name", default="")
    c.add_argument("--timezone", default="UTC")
    sl = sub.add_parser("sleep", help="run one sleep cycle now for a persona id")
    sl.add_argument("persona_id", type=int)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn
        from . import bootstrap
        from .api.app import create_app
        # Bootstrap a persona from SAWII_BOOTSTRAP_FROM when the store is empty.
        # Idempotent: does nothing when any persona already exists.
        try:
            bootstrap.run()
        except Exception as exc:  # noqa: BLE001 - a bootstrap failure must not block the server
            logging.getLogger(__name__).warning("bootstrap failed: %s", exc)
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info",
                    access_log=False)  # keys can travel in ?key= (EventSource); never log them
    elif args.cmd == "create-persona":
        from .identity import create_persona
        from .profiles import PRESETS
        from .store import Store
        settings = load_settings()
        store = Store(settings.db_path)
        definition = args.definition or (PRESETS[args.preset]["definition"] if args.preset in PRESETS else "")
        pid, key = asyncio.run(create_persona(store, name=args.name, definition=definition, preset=args.preset,
                                              user_name=args.user_name, timezone=args.timezone))
        print(json.dumps({"id": pid, "key": key}))
    elif args.cmd == "sleep":
        from .api.app import Runtime
        from .identity import get_persona

        async def run():
            rt = Runtime(load_settings())
            persona = await get_persona(rt.mind.store, args.persona_id)
            print(json.dumps(await rt.sleep.run(persona), indent=2, default=str))
        asyncio.run(run())


if __name__ == "__main__":
    main()
