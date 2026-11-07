"""B as the rememberer: a second mind that searches the past while the first one talks.

A single model can only remember what was put in its context before it started
speaking. Here a separate lobe keeps looking *while A is answering*, after A has
answered, and while the person is away - and it judges what truly connects.

    search    - deterministic, wide and crude (recall.candidates): it only gathers
    judgement - B's (an LLM call): does any of this really connect, and how
    content   - rendered from the stored records, never from B's paraphrase;
                B's one-line "connection" is labeled as B's view

Nothing here is random and nothing is timed: whether A is interrupted mid-answer,
or the memory comes back as an "oh wait" afterwards, depends only on whether B
finished thinking before A finished talking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..llm import LLM, extract_json
from ..memory.recall import Recall, render_event
from ..timesense import local_dt
from .prompts import REMEMBER

log = logging.getLogger(__name__)

KIND_LABEL = {
    "repeat": "they already told you this",
    "contradiction": "this does not match what they say now",
    "open": "still unresolved",
    "link": "",
}


@dataclass
class Recollection:
    event_ids: list[int] = field(default_factory=list)
    layer_ids: list[int] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)
    text: str = ""

    def __bool__(self) -> bool:
        return bool(self.event_ids)


def _listing(cands: list[dict], tz: str) -> str:
    lines = []
    for c in cands:
        when = local_dt(c["ev_at"], tz).strftime("%Y-%m-%d")
        line = f"#{c['event_id']} ({when}) [{c['layer']}·{c['author']}] {c['text'][:300]}"
        if c["quote"] and c["quote"] not in c["text"]:
            line += f' — they said: "{c["quote"][:200]}"'
        lines.append(line)
    return "\n".join(lines)


async def remember(llm: LLM, recall: Recall, persona: dict, *, situation: str, context: str,
                   candidates: list[dict], limit: int = 2) -> Recollection:
    """Ask B whether anything in ``candidates`` truly connects to ``context``."""
    out = Recollection()
    if not candidates:
        return out
    tz = persona.get("timezone") or "UTC"
    system = REMEMBER.format(situation=situation, limit=limit)
    user = f"{context}\n\nPast moments (shortlist):\n{_listing(candidates, tz)}"
    res = await llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                         temperature=0.1, max_tokens=500, json_mode=True)
    data = extract_json(res.text) or {}
    allowed = {c["event_id"] for c in candidates}
    picked: list[tuple[int, str, str]] = []
    for m in data.get("memories") or []:
        if not isinstance(m, dict):
            continue
        try:
            ev = int(m.get("id"))
        except (TypeError, ValueError):
            continue
        if ev not in allowed or ev in [p[0] for p in picked]:
            continue  # B may only point at real records it was shown
        kind = str(m.get("kind", "link")).lower()
        picked.append((ev, kind if kind in KIND_LABEL else "link", str(m.get("connection", "") or "").strip()[:300]))
        if len(picked) >= limit:
            break
    if not picked:
        return out
    layers = await recall.event_layers(persona["id"], [p[0] for p in picked])
    blocks = []
    for ev, kind, conn in picked:
        rows = layers.get(ev)
        if not rows:
            continue
        block = render_event(rows, tz, KIND_LABEL[kind])
        if conn:
            block += f"\n   (B's sense of the connection: {conn})"
        blocks.append(block)
        out.event_ids.append(ev)
        out.layer_ids += [r["id"] for r in rows]
        out.kinds.append(kind)
        out.connections.append(conn)
    out.text = "\n".join(blocks)
    return out


def context_for(user_text: str, tail: list[dict], reply: str = "") -> str:
    lines = ["The conversation so far (latest last):"]
    for t in tail[-3:]:
        if t.get("user_text"):
            lines.append(f"PERSON: {t['user_text'][:500]}")
        if t.get("reply_text"):
            lines.append(f"A: {t['reply_text'][:500]}")
    lines.append(f"PERSON (now): {user_text[:2000]}")
    if reply:
        lines.append(f"A (just answered): {reply[:2000]}")
    return "\n".join(lines)
