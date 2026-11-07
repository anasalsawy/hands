"""The subconscious buffer (ND-SMC / CSMC): keep the day's raw "juice".

Raw data is the asset. A turn is kept byte-exact or not kept at all - no code
path summarizes, normalizes or corrects it. The juice filter decides what is
worth baking into the weights tonight:

    variance = 0.45 * novelty   (distance from THIS person's ordinary, an EMA baseline)
             + 0.35 * style      (deterministic surface markers: stutters, caps, ellipses, slang...)
             + 0.20 * affect     (C's own read of the moment: chord intensity + salience)

Two gates on the juice tier (Anas's 2026-10-21 target: not just correctness —
character shaping):
    variance >= floor        → factual juice (correctness-shaped weights)
    chord_delta >= shaped_me → relational juice (character-shaped weights)

Either gate kept means the row is journaled. Both means premium juice. See
the two separate fine-tune paths (persona-base vs character-layer LoRA) in
life/finetune.py.

These rows are invisible to A. They are not memories A can recall; they are
what A *becomes* after the nightly fine-tune.
"""
from __future__ import annotations

import re

import numpy as np

from .. import emotion
from ..embedder import Embedder, cosine, from_blob, to_blob
from ..store import Store, jdump, jload, now


# How much movement in B's chord counts as "this exchange shaped me". The chord
# is normalized per-axis to [0, 1], so a delta of 0.25 across a few axes means
# the person's read of the user noticeably shifted. Tunable via
# SAWII_SHAPED_ME_FLOOR env var at the module level in Subconscious.__init__.
_DEFAULT_SHAPED_ME_FLOOR = 0.25


def chord_distance(a: dict, b: dict) -> float:
    """L1 distance between two chord dicts over the union of their axes.
    Missing axes are treated as 0. Returns a float >= 0; typical move is
    0.05..0.4 across one turn, so SHAPED_ME_FLOOR ~0.25 catches real shifts
    without firing on noise.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    axes = set(a.keys()) | set(b.keys())
    total = 0.0
    for ax in axes:
        va = float(a.get(ax, 0.0) or 0.0)
        vb = float(b.get(ax, 0.0) or 0.0)
        total += abs(va - vb)
    return round(total, 4)

_MARKERS: list[tuple[str, re.Pattern]] = [
    ("hesitant", re.compile(r"\b(\w)-\1\w*|\b(um+|uh+|er+|hmm+)\b", re.I)),
    ("halting", re.compile(r"(\.\.\.|…)")),
    ("shouting", re.compile(r"\b[A-Z]{3,}\b")),
    ("emphatic", re.compile(r"[!?]{2,}")),
    ("informal", re.compile(r"\b(lol|lmao|haha+|ya know|gonna|wanna|kinda|bro|dude|omg)\b", re.I)),
    ("guarded", re.compile(r"\b(maybe|i guess|idk|i don'?t know|sort of|whatever)\b", re.I)),
    ("absolute", re.compile(r"\b(always|never|everyone|nobody|nothing ever)\b", re.I)),
    ("coarse", re.compile(r"\b(fuck\w*|shit\w*|damn|hell|crap|bitch\w*)\b", re.I)),
    ("vulnerable", re.compile(r"\b(i('m| am) (scared|afraid|lonely|lost)|i miss|i can'?t (stop|cope)|hurts?)\b", re.I)),
    ("laughing", re.compile(r"(ha){2,}|😂|🤣", re.I)),
]


def style_markers(text: str) -> list[str]:
    found = []
    for name, rx in _MARKERS:
        if name == "shouting":
            caps = [w for w in rx.findall(text or "") if w not in {"OK", "USA", "API", "LOL"}]
            if len(caps) >= 1 and len(text) > 8:
                found.append(name)
        elif rx.search(text or ""):
            found.append(name)
    return found


def style_score(markers: list[str]) -> float:
    return min(1.0, 0.28 * len(markers))


class Subconscious:
    def __init__(self, store: Store, embedder: Embedder, *, floor: float = 0.55, buffer_rows: int = 2000,
                 bowl_ratio: float = 0.9, shaped_me_floor: float | None = None):
        self.store = store
        self.embedder = embedder
        self.floor = floor
        self.buffer_rows = buffer_rows
        self.bowl_ratio = bowl_ratio
        import os
        if shaped_me_floor is None:
            try:
                shaped_me_floor = float(os.environ.get(
                    "SAWII_SHAPED_ME_FLOOR", _DEFAULT_SHAPED_ME_FLOOR))
            except (TypeError, ValueError):
                shaped_me_floor = _DEFAULT_SHAPED_ME_FLOOR
        self.shaped_me_floor = shaped_me_floor

    async def ingest(self, persona: dict, turn_id: int, user_text: str, reply: str, *, chord: dict,
                     salience: int) -> dict:
        """Score one turn; keep it raw if it is juice. Never raises into the turn.

        Two gates (either keeps the row):
        - variance >= floor : factual juice. The classic "high-signal exchange".
        - chord_delta >= shaped_me_floor : relational juice. B's read of the
          user shifted enough that this exchange *marked* Maya. Even a plain
          "ok" that moved the mood arc from anxious to settled is worth keeping.
        """
        model, vec = await self.embedder.embed_one(user_text)
        v = np.asarray(vec, dtype=np.float32)
        base = from_blob(persona.get("baseline"))
        same_space = base is not None and persona.get("baseline_model") == model and base.shape == v.shape
        novelty = 1.0 - max(0.0, cosine(v, base)) if same_space else 0.5
        markers = style_markers(user_text)
        affect = min(1.0, emotion.intensity(chord) * 1.6 + 0.15 * (max(1, salience) - 1))
        variance = round(0.45 * novelty + 0.35 * style_score(markers) + 0.20 * affect, 3)

        # Chord-delta signal — the character gate.
        prev_chord = jload(persona.get("last_chord", "{}"), {}) or {}
        chord_delta = chord_distance(prev_chord, chord or {})
        shaped_me = chord_delta >= self.shaped_me_floor

        variance_keeps = variance >= self.floor and len(user_text.strip()) >= 3
        shape_keeps = shaped_me and len(user_text.strip()) >= 3
        kept = variance_keeps or shape_keeps

        # the baseline is this person's ordinary; peaks move it only a little
        alpha = 0.1 / 4 if kept else 0.1
        new_base = v if not same_space else (1 - alpha) * base + alpha * v

        # Persona update: baseline (unchanged) + last_chord (new — carries
        # across turns so the next turn can measure delta against this chord).
        await self.store.update(
            "personas", persona["id"],
            baseline=to_blob(new_base.tolist()),
            baseline_model=model,
            last_chord=jdump(chord or {}),
        )

        juice_id = None
        if kept:
            juice_id = await self.store.insert(
                "juice", persona_id=persona["id"], turn_id=turn_id, raw_transcript=user_text, reply=reply,
                style_markers=jdump(markers), variance=variance, sensory_vector=to_blob(vec),
                raw_chars=len(user_text) + len(reply),
                chord_delta=chord_delta,
                shaped_me=1 if shaped_me else 0,
                chord_before=jdump(prev_chord),
                chord_after=jdump(chord or {}),
                created_at=now(),
            )
            await self.trim(persona["id"])
        return {
            "variance": variance,
            "kept": kept,
            "markers": markers,
            "novelty": round(novelty, 3),
            "juice_id": juice_id,
            "chord_delta": chord_delta,
            "shaped_me": shaped_me,
            "kept_by": (
                "both" if variance_keeps and shape_keeps
                else "variance" if variance_keeps
                else "shaped_me" if shape_keeps
                else None
            ),
        }

    async def trim(self, _persona_or_pid=None) -> int:
        """Stubborn memory: no overflow. Juice is kept raw forever; nothing is evicted for space.
        The method is kept so existing callers do not have to change; it always returns 0."""
        return 0

    async def occupancy(self, pid: int) -> dict:
        rows = await self.store.all("SELECT tier, COUNT(*) AS n, SUM(raw_chars + length(compressed)) AS b "
                                    "FROM juice WHERE persona_id = ? GROUP BY tier", pid)
        tiers = {r["tier"]: r["n"] for r in rows}
        total = sum(tiers.values())
        return {"rows": total, "capacity": self.buffer_rows, "utilization": round(total / self.buffer_rows, 4),
                "tiers": tiers, "bytes_held": int(sum((r["b"] or 0) for r in rows))}
