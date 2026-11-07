"""Sleep: the nightly circadian pass. Everything slow happens here, never on a turn.

    narrative   A rewrites its own first-person story (the stream it wakes up into);
                B audits it against the record - no invented events in its own life story
    psyche      C consolidates readings + hypotheses into one hedged profile; B strikes any line
                with nothing behind it; unconfirmed hypotheses fade (nothing is deleted)
    dream       memories far apart in time are recombined into far-associations; B audits them
    mirror      C watches A's own replies for habits A cannot see in itself
    calibrate   B maps where A tends to be wrong, so A is careful exactly there
    thoughts    stale inner thoughts decay
    anchor      C replays the wrongness dataset it has kept - the outside the mind
                cannot forge, and the seed of the gate on the night's learning (anchor.py)
    cup         records already consolidated into the weights wear away one rung -
                and only while the raw record is over budget (memory/evaporate.py)
    juice       the day's raw high-variance turns are baked into A's weights (finetune.py)
"""
from __future__ import annotations

import logging
import re

from ..brains.prompts import AUDIT, CALIBRATE, DREAM, MIRROR, NARRATIVE, PSYCH_PROFILE
from ..llm import extract_json, strip_reasoning
from ..profiles import persona_frame
from ..store import jload, now
from ..timesense import local_dt
from ..memory import evaporate
from . import anchor
from . import character_anchor as char_anchor
from .finetune import FineTuner

log = logging.getLogger(__name__)


class Sleep:
    def __init__(self, mind, finetuner: FineTuner):
        self.mind = mind
        self.ft = finetuner

    def due(self, persona: dict, at: float | None = None) -> bool:
        at = at or now()
        if at - persona["last_sleep_at"] < 18 * 3600:
            return False
        local = local_dt(at, persona["timezone"])
        idle = at - max(persona["last_user_at"], persona["last_reply_at"]) > 45 * 60
        return local.hour == self.mind.s.sleep_hour and idle

    async def run(self, persona: dict) -> dict:
        pid = persona["id"]
        report: dict = {"persona": pid}
        since = max(persona["narrative_at"], persona["last_sleep_at"])
        for step in (self._narrative, self._psyche, self._dream, self._mirror, self._calibrate,
                     self._anchor, self._character_anchor):
            try:
                report.update(await step(persona, since))
            except Exception as exc:  # noqa: BLE001
                log.warning("sleep step %s failed: %s", step.__name__, exc)
                report[step.__name__] = f"failed: {exc}"[:200]
        # Stubborn memory, still: nothing live is faded or decayed on time. Hypotheses only move on
        # real evidence (confirm/contradict); thoughts stay pending until acted on. What the cup
        # lets go of is different - only the RAW RECORD of what has already been consolidated into
        # the weights, only while over budget, and only one rung per night.
        report["faded_hypotheses"] = 0
        report["decayed_thoughts"] = 0
        try:
            report["cup"] = await evaporate.evaporate(
                self.mind.store, pid, budget=self.mind.s.cup_chars,
                protect_days=self.mind.s.cup_protect_days)
        except Exception as exc:  # noqa: BLE001
            log.warning("cup failed: %s", exc)
            report["cup"] = f"failed: {exc}"[:200]
        # The gate on learning. Two anchors must both hold:
        #   factual  — the wrongness dataset (anchor.py); drift here = Maya
        #              is getting things wrong she used to get right.
        #   character — the tonal-correction dataset (character_anchor.py);
        #              drift here = Maya's tone is wandering from the shape
        #              Anas has been correcting toward.
        # Either below floor → night does NOT bake. A score of None on
        # either anchor means "no cases yet", not evidence of drift.
        anchor_score = report.get("score")
        char_score = report.get("character_anchor_score_val")
        floor = self.mind.s.juice_floor
        refusals = []
        if anchor_score is not None and anchor_score < floor:
            refusals.append(f"factual {anchor_score:.3f} < floor {floor}")
        if char_score is not None and char_score < floor:
            refusals.append(f"character {char_score:.3f} < floor {floor}")
        if refusals:
            msg = "refused: " + "; ".join(refusals)
            log.info("sleep bake %s", msg)
            report["subconscious"] = msg
        else:
            try:
                report["subconscious"] = await self.ft.consolidate(persona)
            except Exception as exc:  # noqa: BLE001
                log.warning("consolidate failed: %s", exc)
                report["subconscious"] = f"failed: {exc}"[:200]
        await self.mind.store.update("personas", pid, last_sleep_at=now())
        return report

    async def _material(self, pid: int, since: float, limit: int = 40) -> str:
        rows = await self.mind.store.all(
            "SELECT layer, author, kind, text, quote, created_at FROM layers WHERE persona_id = ? AND status='active' "
            "AND created_at >= ? ORDER BY salience DESC, id DESC LIMIT ?", pid, since, limit)
        return "\n".join(f"[{r['layer']}·{r['author']}] {r['text']}" + (f' ("{r["quote"]}")' if r["quote"] else "")
                         for r in rows)

    async def _narrative(self, persona: dict, since: float) -> dict:
        pid = persona["id"]
        material = await self._material(pid, since)
        loops = await self.mind.memory.open_loops(pid)
        if not material and not loops:
            return {"narrative": "unchanged (nothing new)"}
        body = [persona_frame(persona["definition"], persona["name"]),
                "Your previous account:\n" + (persona["narrative"] or "(none yet - this is the first one)"),
                "What happened since (from your memory):\n" + (material or "(little)")]
        if loops:
            body.append("Still open:\n" + "\n".join(f"- {lp['text']}" for lp in loops[:10]))
        kwargs = {"model": persona["a_model_override"]} if persona.get("a_model_override") else {}
        res = await self.mind.a.chat([{"role": "system", "content": NARRATIVE},
                                      {"role": "user", "content": "\n\n".join(body)}],
                                     temperature=0.6, max_tokens=500, **kwargs)
        text = strip_reasoning(res.text)
        if len(text) < 20:
            return {"narrative": "kept previous (empty rewrite)"}
        struck = await self._audit(text, "\n\n".join(body[1:]))
        text = _strike(text, struck)
        if len(text) < 20:
            return {"narrative": "kept previous (audit struck it all)", "narrative_struck": len(struck)}
        await self.mind.store.update("personas", pid, narrative=text[:2500], narrative_at=now())
        return {"narrative": "rewritten", "narrative_struck": len(struck)}

    async def _psyche(self, persona: dict, since: float) -> dict:
        pid = persona["id"]
        readings = await self.mind.store.all(
            "SELECT text, quote FROM layers WHERE persona_id = ? AND layer='reading' AND status='active' "
            "ORDER BY id DESC LIMIT 40", pid)
        hyps = await self.mind.memory.hypotheses(pid)
        if not readings and not hyps:
            return {"psych_profile": "unchanged (no evidence)"}
        hl = self.mind.s.psyche_half_life_days
        body = ["Previous account:\n" + (persona["psych_profile"] or "(none yet)"),
                "Hypotheses (confidence now):\n" + "\n".join(
                    f"- ({self.mind.memory.effective_confidence(h, hl):.2f}, seen {h['confirmations']}x) {h['claim']} "
                    f"e.g. {', '.join(jload(h['quotes'], [])[:2])}" for h in hyps[:20]),
                "Recent readings:\n" + "\n".join(f"- {r['text']} (\"{r['quote']}\")" for r in readings)]
        res = await self.mind.b.chat([{"role": "system", "content": PSYCH_PROFILE},
                                      {"role": "user", "content": "\n\n".join(body)}],
                                     temperature=0.3, max_tokens=600)
        text = strip_reasoning(res.text)
        if len(text) < 20:
            return {"psych_profile": "kept previous"}
        struck = await self._audit(text, "\n\n".join(body[1:]))  # C verifies B: evidence or it goes
        text = _strike(text, struck)
        if len(text) < 20:
            return {"psych_profile": "kept previous (audit struck it all)", "profile_struck": len(struck)}
        await self.mind.store.update("personas", pid, psych_profile=text[:3000], psych_at=now())
        return {"psych_profile": "rewritten", "profile_struck": len(struck)}

    async def _dream(self, persona: dict, since: float) -> dict:
        pid = persona["id"]
        rows = await self.mind.store.all(
            "SELECT id, event_id, text, quote FROM layers WHERE persona_id = ? AND status='active' AND salience >= 2 "
            "AND layer IN ('note','fact') ORDER BY id DESC LIMIT 2000", pid)
        by_event: dict[int, dict] = {}
        for r in rows:
            by_event.setdefault(r["event_id"], r)
        pool = list(by_event.values())
        if len(pool) < 3:
            return {"dream": "too few memories"}
        # memories as far apart in time as possible: the oldest, the newest and evenly between
        pool.sort(key=lambda r: r["id"])
        n = min(6, len(pool))
        picks = [pool[round(i * (len(pool) - 1) / (n - 1))] for i in range(n)]
        listing = "\n".join(f"#{p['id']} {p['text']}" + (f' ("{p["quote"]}")' if p["quote"] else "") for p in picks)
        res = await self.mind.a.chat([{"role": "system", "content": DREAM}, {"role": "user", "content": listing}],
                                     temperature=1.0, max_tokens=500, json_mode=True)
        data = extract_json(res.text) or {}
        assoc = [a for a in (data.get("associations") or [])[:3]
                 if isinstance(a, dict) and str(a.get("text", "")).strip()]
        struck = await self._audit("\n".join(str(a["text"]) for a in assoc),
                                   listing + "\n(Associations, metaphors and jokes are fine; invented facts are not.)")
        n = 0
        for a in assoc:
            if any(u in str(a["text"]) for u in struck):
                continue
                await self.mind.store.insert("thoughts", persona_id=pid, text=str(a["text"])[:400], source="dream",
                                             score=0.35, ref=",".join(str(i) for i in (a.get("about_ids") or [])[:6]),
                                             created_at=now())
                n += 1
        return {"dream": f"{n} associations", "dream_struck": len(assoc) - n}

    async def _mirror(self, persona: dict, since: float) -> dict:
        """B looks at A instead of the person: the habits a single mind cannot see in itself."""
        pid = persona["id"]
        rows = await self.mind.store.all("SELECT reply_text FROM turns WHERE persona_id = ? AND reply_text != '' "
                                         "ORDER BY id DESC LIMIT 30", pid)
        if len(rows) < 5:
            return {"mirror": "too few replies"}
        listing = "\n".join(f"- {r['reply_text'][:400]}" for r in reversed(rows))
        res = await self.mind.b.chat([{"role": "system", "content": MIRROR}, {"role": "user", "content": listing}],
                                     temperature=0.2, max_tokens=400, json_mode=True)
        habits = [str(h).strip()[:200] for h in ((extract_json(res.text) or {}).get("habits") or []) if str(h).strip()]
        await self.mind.store.update("personas", pid, habits="\n".join(f"- {h}" for h in habits[:4]))
        return {"mirror": f"{len(habits[:4])} habits"}

    async def _calibrate(self, persona: dict, since: float) -> dict:
        """C maps where A tends to be wrong, from the verdicts it wrote."""
        pid = persona["id"]
        rows = await self.mind.store.all("SELECT level, rationale, findings FROM verdicts WHERE persona_id = ? AND "
                                         "level != 'GREEN' ORDER BY id DESC LIMIT 60", pid)
        if len(rows) < 3:
            return {"calibration": "too few problems to map"}
        lines = []
        for r in rows:
            fs = jload(r["findings"], [])
            detail = "; ".join(f"{f.get('issue', '')}: {f.get('claim', '')}"[:160] for f in fs if isinstance(f, dict))
            lines.append(f"- {r['level']}: {r['rationale'][:200]}" + (f" ({detail})" if detail else ""))
        res = await self.mind.c.chat([{"role": "system", "content": CALIBRATE},
                                      {"role": "user", "content": "\n".join(lines)}],
                                     temperature=0.1, max_tokens=400, json_mode=True)
        areas = [str(a).strip()[:200] for a in ((extract_json(res.text) or {}).get("careful_about") or [])
                 if str(a).strip()]
        await self.mind.store.update("personas", pid, calibration="; ".join(areas[:4]))
        return {"calibration": f"{len(areas[:4])} areas"}

    async def _anchor(self, persona: dict, since: float) -> dict:
        """C rehearses the wrongness dataset: ask A the same things again, verify,
        score. The seed of the gate that will refuse a night's learning when the
        mind has drifted - and the outside the verifier itself is checked against."""
        return await anchor.anchor_run(self.mind, persona)

    async def _character_anchor(self, persona: dict, since: float) -> dict:
        """Replay the tonal-correction dataset. Parallel to _anchor but keyed on
        character (tone) rather than correctness: A is re-asked each case, C
        re-grades, direction heuristics catch obvious regressions.

        Reports under its own key so the gate at run() can read both scores
        without collisions with _anchor's keys.
        """
        out = await char_anchor.character_anchor_run(self.mind, persona)
        # Expose score under a non-colliding key for the bake gate.
        if isinstance(out, dict) and "score" in out:
            out = dict(out)
            out["character_anchor_score_val"] = out["score"]
            del out["score"]  # avoid overwriting the factual anchor's score
        return out

    async def _audit(self, text: str, source: str) -> list[str]:
        """C verifies something the mind wrote about itself or the person; returns sentences to strike."""
        if not text.strip():
            return []
        try:
            res = await self.mind.c.chat([{"role": "system", "content": AUDIT},
                                          {"role": "user", "content": f"TEXT:\n{text}\n\nSOURCE:\n{source[:8000]}"}],
                                         temperature=0.0, max_tokens=500, json_mode=True)
        except Exception as exc:  # noqa: BLE001 - an audit outage must not erase the night's work
            log.warning("audit failed: %s", exc)
            return []
        data = extract_json(res.text) or {}
        return [u.strip() for u in (data.get("unsupported") or []) if isinstance(u, str) and u.strip()
                and u.strip() in text]


def _strike(text: str, sentences: list[str]) -> str:
    for u in sentences:
        text = text.replace(u, "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()
