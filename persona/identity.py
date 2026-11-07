"""Persona = API key. The key is the identity; only its hash is stored."""
from __future__ import annotations

import hashlib
import hmac
import secrets

from .store import Store, jdump, jload, now


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return "sp_" + secrets.token_urlsafe(32)


async def create_persona(store: Store, *, name: str = "", definition: str = "", preset: str = "",
                         user_name: str = "", timezone: str = "UTC", contact: dict | None = None,
                         preferred_channel: str = "app") -> tuple[int, str]:
    key = new_key()
    pid = await store.insert(
        "personas", key_hash=hash_key(key), key_hint=key[:7] + "…" + key[-4:], name=name,
        definition=definition, preset=preset, user_name=user_name, timezone=timezone or "UTC",
        contact=jdump(contact or {}), preferred_channel=preferred_channel, created_at=now())
    return pid, key


async def persona_by_key(store: Store, key: str) -> dict | None:
    if not key or len(key) > 200:
        return None
    row = await store.one("SELECT * FROM personas WHERE key_hash = ?", hash_key(key))
    if row and hmac.compare_digest(row["key_hash"], hash_key(key)):
        return row
    return None


async def get_persona(store: Store, pid: int) -> dict | None:
    return await store.one("SELECT * FROM personas WHERE id = ?", pid)


async def find_by_contact(store: Store, channel: str, address: str) -> dict | None:
    """Inbound SMS/WhatsApp/email/telegram -> which persona the sender belongs to."""
    address = (address or "").strip().lower()
    for row in await store.all("SELECT * FROM personas"):
        contact = jload(row["contact"], {}) or {}
        val = str(contact.get(channel, "")).strip().lower()
        if val and val == address:
            return row
    return None


def public_view(row: dict) -> dict:
    return {
        "id": row["id"], "name": row["name"], "preset": row["preset"], "user_name": row["user_name"],
        "timezone": row["timezone"], "contact": jload(row["contact"], {}),
        "preferred_channel": row["preferred_channel"], "definition": row["definition"],
        "a_model": row["a_model_override"] or None, "key_hint": row["key_hint"],
    }
