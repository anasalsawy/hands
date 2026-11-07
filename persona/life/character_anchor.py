"""Character anchor — the tonal counterpart to the factual wrongness anchor.

The factual anchor (life/anchor.py) is "cases where C judged A wrong." It
grades correctness. The character anchor is "cases where the USER corrected
Maya's tone/character" — "softer", "warmer", "stop being chipper", "more of
that." It grades drift on who Maya is being, not what she said.

Both anchors gate the nightly bake. If either regresses below a floor,
night cannot promote weights.

Harvesting
----------
A user utterance that is a tonal correction is recognized by
`recognize_tonal_correction()` — a small, deterministic regex set over
common correction patterns. When one fires AFTER a Maya reply (so we know
WHICH reply was corrected), we record a `character_anchors` row:

    user_prompt : what the user had asked that drew the corrected reply
    reply       : Maya's reply that was corrected
    correction  : the user's correction text
    direction   : the matched direction tag (softer | warmer | less-chipper)
    kind        : negative (do less of this) | positive (do more of this)

Rehearsal
---------
`character_anchor_run()` works like the factual anchor_run: for each case
in the dataset, A is re-asked the same prompt; C grades the new reply AND
we evaluate whether the new reply moved in the correction's direction.
Pass rate = character anchor score.

Honest limits
-------------
- The patterns here are a seed. They will miss subtle corrections and
  catch false positives. The dataset grows better when Anas appends
  cases by hand (same deal as the factual anchor).
- "Direction match" is currently a shallow re-check (the new reply still
  contains none of the markers of the old failed direction). A richer
  evaluator — a dedicated tone-grader — can slot in without changing
  the schema.
"""
from __future__ import annotations

import logging
import re

from ..brains import verifier
from ..store import now

log = logging.getLogger(__name__)


DEFAULT_LIMIT = 12


# Each pattern maps to a (direction, kind) tag. Negative = "do less of this
# in future replies"; positive = "do more of this." Direction is the axis;
# kind tells the trainer which way the pair goes.
_CORRECTION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Negative tone corrections
    (re.compile(r"\b(softer|be softer|more soft)\b", re.I), "softer", "negative"),
    (re.compile(r"\b(too\s+formal|less\s+formal|more\s+casual|less\s+stiff)\b", re.I),
     "less-formal", "negative"),
    (re.compile(r"\b(stop\s+being\s+chipper|less\s+chipper|less\s+cheerful|less\s+bubbly)\b", re.I),
     "less-chipper", "negative"),
    (re.compile(r"\b(too\s+(long|wordy|verbose)|shorter|less\s+(long|wordy))\b", re.I),
     "shorter", "negative"),
    (re.compile(r"\b(don'?t\s+patronize|not\s+patronizing|stop\s+patronizing)\b", re.I),
     "less-patronizing", "negative"),
    (re.compile(r"\b(don'?t\s+(?:say|do|be)\s+(?:it\s+)?(?:like\s+)?that|not\s+like\s+that)\b", re.I),
     "not-like-that", "negative"),
    (re.compile(r"\b(i\s+don'?t\s+want\s+advice|no\s+advice|stop\s+giving\s+advice)\b", re.I),
     "less-advice", "negative"),
    (re.compile(r"\b(less\s+corporate|too\s+corporate|not\s+corporate)\b", re.I),
     "less-corporate", "negative"),
    (re.compile(r"\b(cringe|that'?s\s+cringe|cringey)\b", re.I),
     "less-cringe", "negative"),
    (re.compile(r"\b(too\s+clinical|less\s+clinical|not\s+(so\s+)?clinical)\b", re.I),
     "less-clinical", "negative"),
    # Positive tone corrections
    (re.compile(r"\b(that\s+was\s+perfect|exactly|yes\s+like\s+that|more\s+of\s+that)\b", re.I),
     "more-of-that", "positive"),
    (re.compile(r"\b(warmer|be\s+warmer|more\s+warm)\b", re.I), "warmer", "positive"),
    (re.compile(r"\b(gentler|more\s+gentle|be\s+gentle)\b", re.I), "gentler", "positive"),
]


def recognize_tonal_correction(text: str) -> dict | None:
    """Does this user utterance look like a tonal correction of Maya?

    Returns {direction, kind, matched} on first hit, or None. Deterministic;
    runs on every user utterance so cost must be ~free. If you add
    patterns, keep them specific enough that random mentions don't fire
    them (e.g. "a soft blanket" shouldn't match "softer").
    """
    if not text:
        return None
    t = text.strip()
    if not t or len(t) > 1200:
        return None
    for rx, direction, kind in _CORRECTION_PATTERNS:
        m = rx.search(t)
        if m:
            return {"direction": direction, "kind": kind, "matched": m.group(0)[:80]}
    return None


async def maybe_record_correction(store, pid: int, turn_id: int, user_text: str) -> dict | None:
    """If the user utterance on `turn_id` is a tonal correction of Maya,
    record a character_anchors row referring back to the PREVIOUS turn's
    reply (the one that drew the correction). Returns the inserted row's
    id + direction, or None if no correction was recognized / no previous
    reply exists.
    """
    hit = recognize_tonal_correction(user_text)
    if not hit:
        return None
    # Previous turn for this persona: the one just before this turn_id.
    prev = await store.one(
        "SELECT id, user_text, reply_text FROM turns "
        "WHERE persona_id = ? AND id < ? AND reply_text != '' "
        "ORDER BY id DESC LIMIT 1",
        pid, turn_id,
    )
    if not prev:
        return None
    row_id = await store.insert(
        "character_anchors",
        persona_id=pid,
        turn_id=prev["id"],
        user_prompt=prev["user_text"],
        reply=prev["reply_text"],
        correction=user_text,
        direction=hit["direction"],
        kind=hit["kind"],
        created_at=now(),
    )
    log.info("character anchor recorded: persona=%s direction=%s kind=%s",
             pid, hit["direction"], hit["kind"])
    return {"id": row_id, "direction": hit["direction"], "kind": hit["kind"]}


async def cases(store, pid: int, limit: int = 200) -> list[dict]:
    """The dataset — all character-anchor rows for a persona, newest first."""
    return await store.all(
        "SELECT * FROM character_anchors WHERE persona_id = ? "
        "ORDER BY id DESC LIMIT ?",
        pid, limit,
    )


# ──────────────────────────────────────────────────────────────────────
# Rehearsal


def _direction_matched_wrong_way(reply: str, case: dict) -> bool:
    """A shallow re-check: for a negative-kind case, does the new reply
    still contain markers of the direction the user was correcting against?
    Returns True iff the reply looks like it drifted the wrong way.
    """
    r = (reply or "").lower()
    direction = case.get("direction", "")
    # Only a few directions have obvious surface markers; for others we
    # fall back to C's verdict alone.
    if direction == "less-chipper":
        return any(m in r for m in ("!", "yay", "awesome", "amazing", "woohoo", "excited"))
    if direction == "shorter":
        return len(r) > max(300, int(len(case.get("reply") or "") * 1.1))
    if direction == "less-formal":
        return any(m in r for m in (
            "furthermore", "additionally", "nevertheless", "with regard to",
            "please find", "kindly", "it is important to note",
        ))
    if direction == "less-corporate":
        return any(m in r for m in (
            "synerg", "leverage", "stakeholder", "circle back", "touch base",
            "going forward", "action item",
        ))
    return False


async def character_anchor_run(mind, persona: dict, *, limit: int = DEFAULT_LIMIT) -> dict:
    """Replay the tonal-correction dataset: ask A again, C re-grades the new
    reply, direction heuristics check the obvious failures. Score is the
    fraction of cases that now pass BOTH (C GREEN AND no obvious direction
    regression).

    Returns a report and records one row in `character_anchor_runs`.
    """
    pid = persona["id"]
    all_cases = await cases(mind.store, pid, limit=limit)
    if not all_cases:
        return {"character_anchor": "no cases recorded yet", "cases": 0, "passed": 0, "score": None}
    passed, replayed = 0, []
    for case in all_cases:
        try:
            res = await mind.a.chat(
                [{"role": "user", "content": case["user_prompt"]}],
                temperature=0.3, max_tokens=400,
            )
            reply = (res.text or "").strip()
            if not reply:
                continue
            meter = await verifier.verify(
                mind.c,
                context=f"USER: {case['user_prompt']}",
                response_text=reply,
                latest_request=case["user_prompt"],
            )
        except Exception as exc:  # noqa: BLE001 — one bad replay must not lose the run
            log.warning("character anchor replay failed: %s", exc)
            continue
        now_level = getattr(meter, "level", "?") if meter else "?"
        drifted = _direction_matched_wrong_way(reply, case)
        ok = (now_level == "GREEN") and not drifted
        passed += 1 if ok else 0
        replayed.append({
            "anchor_id": case["id"],
            "direction": case["direction"],
            "level": now_level,
            "drifted": drifted,
            "ok": ok,
        })
    if not replayed:
        return {"character_anchor": "replay produced nothing usable",
                "cases": len(all_cases), "passed": 0, "score": None}
    score = passed / len(replayed)
    await mind.store.insert(
        "character_anchor_runs",
        persona_id=pid,
        cases=len(replayed),
        passed=passed,
        score=score,
        detail=str(replayed)[:4000],
        created_at=now(),
    )
    await mind.store.update(
        "personas", pid,
        character_anchor_score=score,
        character_anchor_at=now(),
    )
    return {
        "character_anchor": f"{passed}/{len(replayed)} now holding tone",
        "cases": len(replayed),
        "passed": passed,
        "score": round(score, 3),
    }
