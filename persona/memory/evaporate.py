"""The overflowing cup: what the mind has already absorbed, it may stop carrying.

Stubborn memory said nothing is ever faded, compressed or evicted. That was a
reaction to naive decay, and it still holds for everything **live**: a tender
mark, an open loop, a hypothesis, a rule, anything recalled recently, anything
not yet consolidated.

This is the other half, and it is not forgetting. Once a moment is
*consolidated* - baked into the weights, or carried by the narrative the mind
wakes up into - the raw record is no longer the only thing holding it. It can
wear away the way a person's does: you do not remember the exact words of a
conversation from three years ago; you remember that it happened, and what it
meant, because it is part of you now.

So the cup is not a limit on memory. It is a limit on the **raw record**, and it
only ever overflows with what has already been absorbed:

    full    the record as written
    gist    the anchor and the head of it - the details have worn a little
    trace   that it happened, when, and what it was about - the details are gone

Rules, by construction:

* **Nothing happens until the cup is over budget.** Under it, the old promise
  stands unchanged.
* **One rung per night.** Nothing evaporates in a single night; the ladder is
  climbed a step at a time, so the mind always has a night to notice.
* **The person's own words survive longest.** ``gist`` keeps the quote; only
  ``trace`` lets the verbatim words go.
* **Never touched:** tender marks, live state (hypotheses, entities, loops,
  rules, thoughts), anything recalled inside the protection window, and the
  journal of turns (which is not on the ladder at all).

The reduction is local and deterministic - no model call, no cost.
"""
from __future__ import annotations

import logging

from ..store import now

log = logging.getLogger(__name__)

# From whole to worn away.
LADDER = ("full", "gist", "trace")

# Layers that may wear away. Live state is not on the ladder at all.
EVAPORABLE = ("fact", "collateral", "note", "reading", "afterthought")

# Where a record has got to.
NEXT = {"full": "gist", "gist": "trace"}


def gist_of(row: dict, limit: int = 140) -> str:
    """Lossy, deterministic, local: the anchor plus the head of the record."""
    text = " ".join(str(row.get("text", "")).split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    anchor = str(row.get("anchor", "") or "").strip()
    if anchor and anchor.lower() not in text.lower():
        return f"{anchor}: {text}"
    return text


def trace_of(row: dict) -> str:
    """What is left at the end: that it happened, and what it was about."""
    anchor = str(row.get("anchor", "") or "").strip()
    if not anchor:
        anchor = str(row.get("kind", "") or "something")
    return f"(worn away - you remember {anchor} came up)"


async def occupancy(store, pid: int) -> dict:
    """How full the cup is: the raw record still on the ladder, in characters."""
    rows = await store.all(
        "SELECT fidelity, COUNT(*) AS n, SUM(length(text) + length(quote)) AS chars "
        "FROM layers WHERE persona_id = ? AND status = 'active' AND fidelity != 'trace' "
        "GROUP BY fidelity", pid)
    by = {r["fidelity"]: {"n": int(r["n"]), "chars": int(r["chars"] or 0)} for r in rows}
    return {"total_chars": sum(v["chars"] for v in by.values()), "by_fidelity": by}


async def candidates(store, pid: int, *, protect_days: float = 14.0, limit: int = 400) -> list[dict]:
    """The records that may wear away, most-consolidated first.

    Consolidation is read from the juice tier: a turn that has been *baked* into
    the weights is carried by the mind itself, so its raw record may go first.
    Never returned: tender marks, live state, anything recalled inside the
    protection window, anything already at the end of the ladder.
    """
    cutoff = now() - protect_days * 86400
    return await store.all(
        "SELECT l.id, l.layer, l.fidelity, l.text, l.quote, l.anchor, l.kind, l.created_at, "
        "       l.recalled_count, l.last_recalled_at, COALESCE(j.tier, '') AS juice_tier "
        "FROM layers l "
        "LEFT JOIN events e ON e.id = l.event_id "
        "LEFT JOIN juice  j ON j.turn_id = e.turn_id "
        "WHERE l.persona_id = ? AND l.status = 'active' AND l.tender = 0 "
        "  AND l.fidelity != 'trace' "
        "  AND (l.recalled_count = 0 OR l.last_recalled_at < ?) "
        "ORDER BY (COALESCE(j.tier, '') IN ('baked','compressed')) DESC, "
        "         l.recalled_count ASC, l.id ASC "
        "LIMIT ?", pid, cutoff, limit)


async def evaporate(store, pid: int, *, budget: int, protect_days: float = 14.0,
                    max_steps: int = 400, dry_run: bool = False) -> dict:
    """Climb the ladder one rung at a time, only while the cup is over ``budget``."""
    before = await occupancy(store, pid)
    report = {"over": before["total_chars"] > budget, "budget": budget,
              "before_chars": before["total_chars"], "after_chars": before["total_chars"],
              "to_gist": 0, "to_trace": 0, "steps": 0, "dry_run": dry_run}
    if not report["over"]:
        return report

    remaining = before["total_chars"] - budget
    for row in await candidates(store, pid, protect_days=protect_days, limit=max_steps):
        if remaining <= 0:
            break
        nxt = NEXT.get(row["fidelity"], "trace")
        text = gist_of(row) if nxt == "gist" else trace_of(row)
        quote = row["quote"] if nxt == "gist" else ""   # the person's words survive longest
        saved = (len(row["text"]) + len(row["quote"])) - (len(text) + len(quote))
        report["to_gist" if nxt == "gist" else "to_trace"] += 1
        report["steps"] += 1
        if not dry_run:
            await store.update("layers", row["id"], fidelity=nxt, text=text, quote=quote,
                               evaporated_at=now())
        remaining -= max(0, saved)
        log.info("cup: layer %s %s -> %s (%+d chars)", row["id"], row["fidelity"], nxt, -saved)

    if not dry_run:
        report["after_chars"] = (await occupancy(store, pid))["total_chars"]
    return report
