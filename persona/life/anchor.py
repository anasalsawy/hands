"""The anchor: the wrongness dataset C keeps, and its rehearsal during sleep.

C is the verifier now, so every judgement it makes is also a record. The
``verdicts`` table already holds each level, its findings and the turn it judged
- so the dataset the aging loop needs is simply that, joined to what was
actually said.

During sleep those cases are replayed: A is asked the same thing again, C
verifies the new answer, and the pass rate becomes the anchor score.

This is the piece of the world the mind cannot forge - the outside the verifier
is checked against. It is deliberately small, cheap and read-only: **a
rehearsal, never a conversation.** Nothing here opens a turn, writes memory, or
touches the journal.

Honest limitation: the replay asks A bare - no recall, no persona frame - so the
score measures the model's own drift rather than a full turn. That is exactly
what an anchor is for, but it means the number is a drift signal, not a quality
score.
"""
from __future__ import annotations

import logging

from ..brains import verifier
from ..store import now

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 12


async def wrong_cases(store, pid: int, limit: int = 200) -> list[dict]:
    """Every recorded case where C judged A wrong, newest first."""
    return await store.all(
        "SELECT v.id, v.turn_id, v.level, v.rationale, v.findings, v.created_at, "
        "       t.user_text, t.reply_text "
        "FROM verdicts v JOIN turns t ON t.id = v.turn_id "
        "WHERE v.persona_id = ? AND v.level != 'GREEN' AND t.user_text != '' "
        "ORDER BY v.id DESC LIMIT ?", pid, limit)


async def anchor_run(mind, persona: dict, *, limit: int = DEFAULT_LIMIT) -> dict:
    """Replay the wrongness dataset: ask A again, let C verify, score the pass rate.

    Returns a report and records one row in ``anchor_runs`` - the history the
    gate reads to tell whether the mind is drifting.
    """
    pid = persona["id"]
    cases = await wrong_cases(mind.store, pid, limit=limit)
    if not cases:
        return {"anchor": "no cases recorded yet", "cases": 0, "passed": 0, "score": None}
    passed, replayed = 0, []
    for case in cases:
        try:
            res = await mind.a.chat([{"role": "user", "content": case["user_text"]}],
                                    temperature=0.3, max_tokens=400)
            reply = (res.text or "").strip()
            if not reply:
                continue
            meter = await verifier.verify(mind.c, context=f"USER: {case['user_text']}",
                                          response_text=reply, latest_request=case["user_text"])
        except Exception as exc:  # noqa: BLE001 - one bad replay must not lose the run
            log.warning("anchor replay failed: %s", exc)
            continue
        now_level = getattr(meter, "level", "?") if meter else "?"
        ok = now_level == "GREEN"
        passed += 1 if ok else 0
        replayed.append({"verdict_id": case["id"], "was": case["level"], "now": now_level, "ok": ok})
    if not replayed:
        return {"anchor": "replay produced nothing usable", "cases": len(cases), "passed": 0, "score": None}
    score = passed / len(replayed)
    await mind.store.insert("anchor_runs", persona_id=pid, cases=len(replayed), passed=passed,
                            score=score, detail=str(replayed)[:4000], created_at=now())
    await mind.store.update("personas", pid, anchor_score=score, anchor_at=now())
    return {"anchor": f"{passed}/{len(replayed)} now clean", "cases": len(replayed),
            "passed": passed, "score": round(score, 3)}
