"""Per-user persona minting (A2) — provision a persona on account creation.

The companion-app's upstream.py spoke HTTP to a separate sawii-persona
service. We ARE sawii-persona now, so we call `identity.create_persona`
directly in-process and record the (user_id → persona_key, upstream_id)
mapping in the gateway's `personas` table.

The persona key is returned to the client exactly once (at registration)
and never again — only its SHA-256 hash is stored in the mind's own
personas table. The gateway stores the raw key under its own encryption-at-
rest layer when it's set up (future PR); for now it's stored plaintext
alongside users, matching the companion-app's design.

Seed template: if SAWII_PERSONA_SEED_FILE points to a JSON file with
{name, definition, timezone}, newly minted personas get that template as
their definition. Otherwise a minimal default is used.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..identity import create_persona
from .db import get_db, now

log = logging.getLogger(__name__)


_seed_cache: dict | None = None


def load_seed() -> dict:
    """Persona template for newly provisioned accounts. Falls back to {}."""
    global _seed_cache
    if _seed_cache is not None:
        return _seed_cache
    _seed_cache = {}
    path = os.environ.get("SAWII_PERSONA_SEED_FILE", "").strip()
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                _seed_cache = {
                    "name": str(data.get("name", ""))[:80],
                    "definition": str(data.get("definition", ""))[:12000],
                    "timezone": str(data.get("timezone", "UTC"))[:64],
                }
        except (OSError, ValueError) as exc:
            log.warning("could not read SAWII_PERSONA_SEED_FILE=%s: %s", path, exc)
    return _seed_cache


def reset_seed_cache_for_tests() -> None:
    global _seed_cache
    _seed_cache = None


async def provision_persona_for_user(
    mind_store: Any,
    user_id: int,
    *,
    user_name: str = "",
    timezone: str = "UTC",
) -> dict:
    """Create a mind-side persona for this user and record the mapping.

    Idempotent: if the user already has a persona on record, returns the
    existing row without creating a new one. (If someone wants to reset a
    persona, they do it via a separate route; registration-time minting
    runs once per account.)
    """
    db = get_db()
    existing = db.one("SELECT * FROM personas WHERE user_id=?", (user_id,))
    if existing:
        return existing

    seed = load_seed()
    pid, key = await create_persona(
        mind_store,
        name=seed.get("name", "") or user_name or "",
        definition=seed.get("definition", ""),
        user_name=user_name,
        timezone=seed.get("timezone", "") or timezone or "UTC",
        preferred_channel="app",
    )
    db.write(
        "INSERT INTO personas (user_id, persona_key, upstream_id, provisioned_at, source)"
        " VALUES (?,?,?,?,?)",
        (user_id, key, pid, now(), "self"),
    )
    return {
        "user_id": user_id,
        "persona_key": key,
        "upstream_id": pid,
        "provisioned_at": now(),
        "source": "self",
    }


def get_persona_for_user(user_id: int) -> dict | None:
    """Return the gateway's persona record for this user, or None."""
    return get_db().one("SELECT * FROM personas WHERE user_id=?", (user_id,))
