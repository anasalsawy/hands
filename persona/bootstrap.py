"""First-run bootstrap: create a persona from a JSON file if the store has none.

Useful when the API is unreachable from the operator's machine (sandbox, policy,
firewall): the operator sets ``SAWII_BOOTSTRAP_FROM=/app/prompts/<name>.json``, the
container creates the persona on startup if the store is empty, and logs the key.

Idempotent: does nothing when any persona already exists.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from .config import load_settings
from .identity import create_persona
from .store import Store

log = logging.getLogger(__name__)


async def _run_async(settings) -> None:
    store = Store(settings.db_path)
    try:
        path = (os.environ.get("SAWII_BOOTSTRAP_FROM") or "").strip()
        sync = (os.environ.get("SAWII_BOOTSTRAP_SYNC") or "").strip().lower() in {"1", "true", "yes", "on"}
        row = await store.one("SELECT COUNT(*) AS n FROM personas")
        existing = row["n"] if row else 0
        if existing and not sync:
            log.info("[bootstrap] %d persona(s) already exist; skipping (set SAWII_BOOTSTRAP_SYNC=true to re-apply "
                     "the definition from the file)", existing)
            return
        if not path:
            log.info("[bootstrap] SAWII_BOOTSTRAP_FROM not set; starting with an empty store")
            return
        p = Path(path)
        if not p.exists():
            log.warning("[bootstrap] SAWII_BOOTSTRAP_FROM=%s does not exist", path)
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        name = str(data.get("name", "") or "")[:80]
        # Sync mode: update an existing persona with the same name instead of creating a new one.
        if existing and sync:
            match = await store.one("SELECT id FROM personas WHERE name = ? ORDER BY id LIMIT 1", name)
            if match:
                await store.update(
                    "personas", match["id"],
                    definition=str(data.get("definition", "") or "")[:12000],
                    preset=str(data.get("preset", "") or ""),
                    user_name=str(data.get("user_name", "") or "")[:80],
                    timezone=str(data.get("timezone", "UTC") or "UTC")[:64],
                    preferred_channel=str(data.get("preferred_channel", "app") or "app"),
                )
                log.warning("[bootstrap] SYNCED persona #%s name=%s from %s", match["id"], name, path)
                return
            log.info("[bootstrap] SYNC: no persona named %s found; falling through to create", name)
        pid, key = await create_persona(
            store,
            name=name,
            definition=str(data.get("definition", "") or "")[:12000],
            preset=str(data.get("preset", "") or ""),
            user_name=str(data.get("user_name", "") or "")[:80],
            timezone=str(data.get("timezone", "UTC") or "UTC")[:64],
            contact=data.get("contact") if isinstance(data.get("contact"), dict) else {},
            preferred_channel=str(data.get("preferred_channel", "app") or "app"),
        )
        log.warning("[bootstrap] CREATED persona #%s name=%s", pid, name)
        log.warning("[bootstrap] KEY=%s  (store it; this is the only time it is printed)", key)
        # Also persist the key next to the DB so it can be retrieved via `cat` later.
        out = Path(settings.data_dir) / f"first-persona-{pid}.txt"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"id={pid}\nname={name}\nkey={key}\n", encoding="utf-8")
            log.warning("[bootstrap] key also written to %s", out)
        except Exception as exc:  # noqa: BLE001
            log.warning("[bootstrap] could not write key file: %s", exc)
    finally:
        store.close()


def run() -> None:
    """Synchronous entry point for container startup."""
    asyncio.run(_run_async(load_settings()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
