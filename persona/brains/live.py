"""The other lobes, live: what B and C do *while* A is still speaking.

    gut_read     C reads the person's message the moment it lands - tone, and
                 whether it matches what C predicted they would say (surprise)
    live_check   B listens to A's unfinished answer, sentence by sentence, and
                 stops it if A says something false or breaks a standing request

Both run concurrently with A's stream. What they find goes into the inner-voice
queue, and A hears it at the next sentence boundary. A single model cannot
check its own sentence before finishing the next one; a second mind can.
"""
from __future__ import annotations

from ..llm import LLM, extract_json
from .prompts import GUT_READ, LIVE_CHECK


async def gut_read(llm: LLM, *, user_text: str, tail: list[dict], expect: str, user_mood: str = "") -> dict:
    parts = []
    for t in tail[-2:]:
        parts.append(f"PERSON: {t['user_text'][:400]}")
        if t.get("reply_text"):
            parts.append(f"A: {t['reply_text'][:400]}")
    if user_mood:
        parts.append(f"How they have seemed lately: {user_mood}")
    parts.append(f"Your prediction of their next message: {expect or '(none yet)'}")
    parts.append(f"Their message, just now: {user_text[:3000]}")
    res = await llm.chat([{"role": "system", "content": GUT_READ}, {"role": "user", "content": "\n".join(parts)}],
                         temperature=0.2, max_tokens=250, json_mode=True)
    data = extract_json(res.text) or {}
    return {"tone": str(data.get("tone", "") or "").strip()[:300],
            "surprise": str(data.get("surprise", "") or "").strip()[:300] if expect else ""}


async def live_check(llm: LLM, *, user_text: str, partial: str, recall_text: str, rules: list[dict]) -> str:
    """Return a one-sentence note to A if A must be stopped, else ''."""
    parts = []
    if rules:
        parts.append("The person's standing requests:\n" + "\n".join(f"- {r['text']}" for r in rules))
    if recall_text:
        parts.append("Memories A was given:\n" + recall_text[:3000])
    parts.append(f"PERSON: {user_text[:2000]}")
    parts.append(f"A, so far (still speaking): {partial[-2500:]}")
    res = await llm.chat([{"role": "system", "content": LIVE_CHECK}, {"role": "user", "content": "\n\n".join(parts)}],
                         temperature=0.0, max_tokens=200, json_mode=True)
    data = extract_json(res.text) or {}
    if data.get("interrupt") is True:
        return str(data.get("note", "") or "").strip()[:300]
    return ""
