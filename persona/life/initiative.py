"""Initiative: the persona is always on and speaks first when it has a reason.

Every heartbeat, per persona:
  * reminders that came due           (always delivered - the person asked for them)
  * B's required corrections and afterthoughts ("oh, one more thing...")
  * finished workshop jobs
  * the thought stream: B's reflections and recollections, dream associations
  * long silences ("haven't heard from you in a while")
Then the voicing gate and its twin, the inhibitor: most thoughts are never said.
Quiet hours, a daily cap, a minimum gap, and "don't interrupt a live conversation"
keep it a friend and not a notification machine. A writes the message in the
persona's voice; B checks it contains nothing invented before it is sent.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..brains import rememberer
from ..brains.prompts import OUTREACH, OUTREACH_CHECK, REMEMBER_IDLE
from ..channels.adapters import Channels
from ..identity import get_persona
from ..llm import extract_json, strip_reasoning
from ..profiles import persona_frame
from ..store import jload, now
from ..timesense import fmt_duration, in_quiet_hours, local_dt

log = logging.getLogger(__name__)

SPEAK_THRESHOLD = 0.6


class Initiative:
    def __init__(self, mind, channels: Channels, sleep=None):
        self.mind = mind
        self.channels = channels
        self.sleep = sleep
        self.ticks = 0
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ outbox
    async def send(self, persona: dict, text: str, *, reason: str, channel: str | None = None) -> int:
        pid = persona["id"]
        ch = self.channels.pick(persona, channel)
        oid = await self.mind.store.insert("outbox", persona_id=pid, channel=ch, text=text, reason=reason[:200],
                                           created_at=now())
        # it is part of the one stream: A will see it said this
        tid = await self.mind.memory.new_turn(pid, "", channel=ch, kind="outreach")
        await self.mind.memory.finish_turn(tid, text)
        await self.mind.store.update("personas", pid, last_reply_at=now())
        status, err = "sent", ""
        try:
            await self.channels.deliver(persona, ch, text)
        except Exception as exc:  # noqa: BLE001
            status, err = "failed", str(exc)[:300]
            log.warning("delivery failed on %s: %s", ch, exc)
        await self.mind.store.update("outbox", oid, status=status, error=err, sent_at=now())
        self.mind.bus.publish(pid, {"type": "message", "id": oid, "text": text, "reason": reason, "channel": ch,
                                    "at": now()})
        return oid

    async def sent_today(self, pid: int) -> int:
        row = await self.mind.store.one("SELECT COUNT(*) AS n FROM outbox WHERE persona_id = ? AND created_at > ? "
                                        "AND reason NOT LIKE 'reminder%'", pid, now() - 86400)
        return row["n"]

    async def last_outreach(self, pid: int) -> float:
        row = await self.mind.store.one("SELECT MAX(created_at) AS t FROM outbox WHERE persona_id = ? "
                                        "AND reason NOT LIKE 'reminder%'", pid)
        return row["t"] or 0.0

    # ------------------------------------------------------------------ heartbeat
    async def tick(self, at: float | None = None) -> list[dict]:
        at = at or now()
        self.ticks += 1
        actions = []
        for p in await self.mind.store.all("SELECT id FROM personas"):
            try:
                actions += await self.tick_persona(p["id"], at)
            except Exception as exc:  # noqa: BLE001
                log.warning("heartbeat failed for persona %s: %s", p["id"], exc)
        return actions

    async def tick_persona(self, pid: int, at: float) -> list[dict]:
        persona = await get_persona(self.mind.store, pid)
        actions: list[dict] = []
        # 1. reminders: exact, always
        for lp in await self.mind.memory.due_loops(pid, at):
            try:
                text = await self._compose(persona, f"It is time for a reminder they asked for: {lp['text']} "
                                                    f"(they said: \"{lp['quote']}\")", "")
            except Exception as exc:  # noqa: BLE001 - a reminder must never be lost to a model outage
                log.warning("reminder compose failed: %s", exc)
                text = ""
            text = text or f"Reminder: {lp['text']}"
            await self.send(persona, text, reason=f"reminder #{lp['id']}")
            await self.mind.store.update("loops", lp["id"], notified_at=at)
            actions.append({"persona": pid, "action": "reminder", "loop": lp["id"]})
        # 2. sleep when it is time
        if self.sleep is not None and self.sleep.due(persona, at):
            actions.append({"persona": pid, "action": "sleep", "report": await self.sleep.run(persona)})
            persona = await get_persona(self.mind.store, pid)
        # 3. reflection: while they are away, B thinks back over what you last talked about
        if self.ticks % max(1, self.mind.s.reflect_every_ticks) == 0:
            thought = await self.reflect(persona, at)
            if thought:
                actions.append({"persona": pid, "action": "reflect", "thought": thought})
        # 4. voicing gate + inhibitor
        actions += await self.consider(pid, at, persona)
        return actions

    async def consider(self, pid: int, at: float | None = None, persona: dict | None = None) -> list[dict]:
        """Run the voicing gate once for a persona (heartbeat, or right after a sudden recall)."""
        at = at or now()
        persona = persona or await get_persona(self.mind.store, pid)
        actions: list[dict] = []
        cand = await self._candidates(persona, at)
        if not cand:
            return actions
        best = max(cand, key=lambda c: c["score"])
        verdict = await self._gate(persona, best, at)
        actions.append({"persona": pid, "action": "gate", "candidate": best["why"], "verdict": verdict})
        if verdict == "speak":
            text = await self._compose(persona, best["why"], best.get("material", ""))
            if text and await self._check(persona, text, best):
                await self.send(persona, text, reason=best["why"][:180])
                await self._resolve(persona, best, "voiced")
                actions.append({"persona": pid, "action": "outreach", "text": text})
            else:
                await self._resolve(persona, best, "inhibited")
        elif verdict == "drop":
            await self._resolve(persona, best, "inhibited")
        return actions

    async def _candidates(self, persona: dict, at: float) -> list[dict]:
        pid = persona["id"]
        out = []
        for t in await self.mind.store.all("SELECT * FROM thoughts WHERE persona_id = ? AND status='pending' "
                                           "ORDER BY id DESC LIMIT 20", pid):
            age_h = (at - t["created_at"]) / 3600
            score = float(t["score"])
            if t["source"] in ("afterthought", "sudden", "dropped") and age_h > 2:
                score *= 0.3  # "oh wait" is only natural soon after
            why = {"correction": "You need to correct something you told them: ",
                   "afterthought": "An afterthought on what you just talked about: ",
                   "deep": "Your workshop finished something for them: ",
                   "sudden": "",
                   "reflection": "",
                   "recollection": "",
                   "dream": "A strange connection your mind made overnight: "}.get(t["source"], "A thought: ")
            out.append({"kind": "thought", "id": t["id"], "source": t["source"], "score": score,
                        "why": why + t["text"]})
        last_contact = max(persona["last_user_at"], persona["last_reply_at"])
        unanswered = (await self.mind.store.one(
            "SELECT COUNT(*) AS n FROM outbox WHERE persona_id = ? AND created_at > ? AND reason NOT LIKE 'reminder%'",
            pid, persona["last_user_at"]))["n"]
        if persona["last_user_at"] and at - last_contact > 2.5 * 86400 and unanswered < 2:  # never pester
            days = (at - last_contact) / 86400
            out.append({"kind": "silence", "score": min(0.85, 0.45 + 0.08 * days),
                        "why": f"You have not heard from them in {fmt_duration(at - last_contact)}. Check in like a "
                               "friend would - lightly, maybe with something you remember about them."})
        return out

    async def _gate(self, persona: dict, cand: dict, at: float) -> str:
        """speak | hold | drop. The inhibitor is what makes it polite."""
        if cand["score"] < 0.25:
            return "drop"
        if cand["score"] < SPEAK_THRESHOLD:
            return "hold"
        urgent = cand.get("source") == "correction"
        if in_quiet_hours(at, persona["timezone"], self.mind.s.quiet_hours) and not urgent:
            return "hold"
        if at - persona["last_user_at"] < 120 and cand.get("source") not in ("afterthought", "correction", "sudden"):
            return "hold"  # mid-conversation: don't talk over them
        if await self.sent_today(persona["id"]) >= self.mind.s.daily_outreach_cap and not urgent:
            return "hold"
        if at - await self.last_outreach(persona["id"]) < self.mind.s.min_gap_between_outreach_minutes * 60 and not urgent:
            return "hold"
        return "speak"

    async def _resolve(self, persona: dict, cand: dict, status: str) -> None:
        if cand["kind"] != "thought":
            return
        await self.mind.store.update("thoughts", cand["id"], status=status)
        if status == "voiced":
            sc = self.mind.scratch_for(persona["id"])
            if cand.get("source") == "correction":
                sc.correction = ""  # already corrected out loud
            if cand.get("source") == "sudden":
                sc.sudden = ""      # already said "oh wait..."; don't repeat it next turn

    async def _compose(self, persona: dict, reason: str, material: str) -> str:
        recall = await self.mind.recall.recall(persona, reason, budget=1500)
        tz = persona["timezone"]
        body = [persona_frame(persona["definition"], persona["name"]),
                f"Now: {local_dt(now(), tz).strftime('%A %H:%M')} ({tz}).",
                f"Reason to reach out: {reason}"]
        if persona.get("narrative"):
            body.append("Your own account of yourself:\n" + persona["narrative"])
        if recall.text:
            body.append("Memories:\n" + recall.text)
        if material:
            body.append(material)
        kwargs = {"model": persona["a_model_override"]} if persona.get("a_model_override") else {}
        res = await self.mind.a.chat([{"role": "system", "content": OUTREACH},
                                      {"role": "user", "content": "\n\n".join(body)}],
                                     temperature=0.8, max_tokens=300, **kwargs)
        from ..brains.voice_a import strip_feel
        text, _ = strip_feel(strip_reasoning(res.text))
        if not text or text.strip().upper().startswith("SKIP"):
            return ""
        return text.strip()[:1200]

    async def _check(self, persona: dict, text: str, cand: dict) -> bool:
        try:
            res = await self.mind.b.chat([{"role": "system", "content": OUTREACH_CHECK},
                                          {"role": "user", "content": f"Reason: {cand['why']}\n\nMessage: {text}"}],
                                         temperature=0.0, max_tokens=200, json_mode=True)
            data = extract_json(res.text) or {}
            return bool(data.get("ok", False))
        except Exception as exc:  # noqa: BLE001
            log.warning("outreach check failed, holding message: %s", exc)
            return False

    # ------------------------------------------------------------------ reflection (B, between talks)
    async def reflect(self, persona: dict, at: float | None = None) -> str | None:
        """Once per conversation, after it has gone quiet: B looks back at what was said and searches the
        longer past for what connects. Nothing random - if there is nothing new to think about, it doesn't."""
        at = at or now()
        pid = persona["id"]
        if not persona["last_user_at"] or at - persona["last_user_at"] < 20 * 60:
            return None  # still talking (or never talked)
        last = await self.mind.store.one("SELECT MAX(created_at) AS t FROM thoughts WHERE persona_id = ? AND "
                                         "source = 'reflection'", pid)
        if last["t"] and last["t"] >= persona["last_user_at"]:
            return None  # already reflected on this conversation
        recent = await self.mind.store.all(
            "SELECT id, user_text, reply_text FROM turns WHERE persona_id = ? AND kind = 'chat' AND started_at > ? "
            "ORDER BY id DESC LIMIT 6", pid, persona["last_user_at"] - 6 * 3600)
        if not recent:
            return None
        recent.reverse()
        cue = "\n".join(f"{t['user_text']}\n{t['reply_text']}" for t in recent)
        context = "What you talked about last:\n" + "\n".join(
            f"PERSON: {t['user_text'][:400]}\nA: {t['reply_text'][:400]}" for t in recent)
        cands = await self.mind.recall.candidates(persona, cue, limit=40, exclude_turn_ids={t["id"] for t in recent})
        # mark the conversation as reflected on even if nothing connects, so this runs once
        rec = await rememberer.remember(self.mind.b, self.mind.recall, persona, situation=REMEMBER_IDLE,
                                        context=context, candidates=cands)
        text = ("Thinking back on your last conversation, this came to mind:\n" + rec.text) if rec else ""
        status = "pending" if rec else "decayed"
        await self.mind.store.insert("thoughts", persona_id=pid, text=text or "(nothing connected)",
                                     source="reflection", status=status,
                                     score=0.65 if "open" in rec.kinds else 0.45,
                                     ref=",".join(map(str, rec.event_ids)), created_at=now())
        return text or None

    # ------------------------------------------------------------------ loop
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="heartbeat")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            started = time.monotonic()
            try:
                await self.tick()
                if self.sleep is not None:
                    await self.sleep.ft.poll()
            except Exception as exc:  # noqa: BLE001
                log.warning("heartbeat tick failed: %s", exc)
            await asyncio.sleep(max(1.0, self.mind.s.heartbeat_seconds - (time.monotonic() - started)))
