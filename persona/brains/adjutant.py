"""B - the adjutant (operations). Runs after A answers, never blocks the person:
collateral memory, open loops, the conscience note, standing rules, afterthoughts.
The verdict is no longer B's - verification moved to C (see README.md).
"""
from __future__ import annotations

import logging

from ..llm import LLM, extract_json
from ..timesense import local_dt
from .prompts import B_SYSTEM

log = logging.getLogger(__name__)


def transcript(tail: list[dict], user_text: str, reply: str, limit: int = 6000) -> str:
    lines = []
    for t in tail[-4:]:
        lines.append(f"USER: {t['user_text']}")
        if t["reply_text"]:
            lines.append(f"A: {t['reply_text']}")
    lines.append(">>> THIS TURN")
    lines.append(f"USER: {user_text}")
    lines.append(f"A: {reply}")
    text = "\n".join(lines)
    return text[-limit:]


async def review(llm: LLM, *, persona: dict, user_text: str, reply: str, tail: list[dict], recall_text: str,
                 consults: list[dict], loops: list[dict], psyche_brief: str, at: float, a_chord: dict | None = None,
                 rules: list[dict] | None = None) -> dict:
    tz = persona.get("timezone") or "UTC"
    parts = [f"Local time now: {local_dt(at, tz).isoformat(timespec='minutes')} ({tz})"]
    if psyche_brief:
        parts.append("Brief from the psyche lobe (as of last turn):\n" + psyche_brief)
    if loops:
        parts.append("Open loops:\n" + "\n".join(
            f"#{lp['id']} [{lp['kind']}] {lp['text']}" + (f" (due {local_dt(lp['due_at'], tz).isoformat(timespec='minutes')})"
                                                           if lp["due_at"] else "") for lp in loops[:20]))
    if rules:
        parts.append("The person's standing requests (drop by id if they lift one):\n" + "\n".join(
            f"#{r['id']} {r['text']}" for r in rules))
    if a_chord:
        from ..emotion import render
        parts.append(f"How A declared it feels after this answer: {render(a_chord)}")
    if recall_text:
        parts.append("Memories A was given this turn:\n" + recall_text[:3000])
    if consults:
        parts.append("Consultant minds (parallel work on this question):\n" + "\n".join(
            f"[{c['from']}] {c['said'][:1200]}" for c in consults))
    parts.append("Transcript:\n" + transcript(tail, user_text, reply))
    res = await llm.chat([{"role": "system", "content": B_SYSTEM},
                          {"role": "user", "content": "\n\n".join(parts)}],
                         temperature=0.2, max_tokens=1400, json_mode=True)
    data = extract_json(res.text) or {}
    verdict = data.get("verdict") if isinstance(data.get("verdict"), dict) else {}
    level = str(verdict.get("level", "YELLOW")).upper()
    if level not in ("GREEN", "YELLOW", "RED"):
        level = "YELLOW"
    findings = [f for f in (verdict.get("findings") or []) if isinstance(f, dict)][:8]
    return {
        "level": level,
        "rationale": str(verdict.get("rationale", "") or ("unparseable review" if not data else ""))[:600],
        "findings": findings,
        "correction": str(data.get("correction_required", "") or "").strip()[:600],
        "collateral": [c for c in (data.get("collateral") or []) if isinstance(c, dict)][:6],
        "loops": [lp for lp in (data.get("loops") or []) if isinstance(lp, dict)][:8],
        "conscience": str(data.get("conscience", "") or "").strip()[:500],
        "afterthought": str(data.get("afterthought", "") or "").strip()[:600],
        "dropped": [d for d in (data.get("dropped") or []) if isinstance(d, dict)][:3],
        "rules": [r for r in (data.get("rules") or []) if isinstance(r, dict)][:4],
        "regulate": str(data.get("regulate", "") or "").strip()[:300],
        # B's edit of the one picture of them that A reads on every turn. Quoted
        # lines only; the picture is memory, not a read of the mood.
        "picture": [p for p in (data.get("picture") or []) if isinstance(p, dict)][:10],
        "parsed": bool(data),
    }
