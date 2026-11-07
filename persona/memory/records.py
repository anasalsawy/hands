"""The memory store: one record per event, three authored layers, plus the
living state around it (psyche hypotheses, entities, open loops, verdicts).

Rules enforced here, in code:
  * quote-or-die - anything attributed to the user carries their verbatim words
  * one writer per layer - A writes fact, B writes collateral, C writes notes/readings
  * raw text is never rewritten - a later reading is appended, the quote stays carved
  * contradiction never deletes - it supersedes (old row kept, marked, linked)
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any

import numpy as np

from .. import emotion
from ..embedder import Embedder, cosine, from_blob, hash_embed, to_blob
from ..store import Store, jdump, jload, now
from ..timesense import ZoneInfo
from .grounding import ground

log = logging.getLogger(__name__)

# One writer per layer, enforced on write. After the lobe swap (C verifies, B
# feels) the psyche layers belong to B. C writes verdicts, not memory layers.
LAYER_AUTHOR = {"fact": "A", "collateral": "B", "note": "B", "reading": "B", "afterthought": "B"}
NOTE_KINDS = {"preference", "plan", "detail", "aside", "joke", "callback", "feeling", "promise", "event",
              "reading", "fact", "collateral", "afterthought"}
SUPERSEDABLE = {"preference", "plan", "detail", "fact"}

_REVISION = re.compile(r"\b(not anymore|no longer|changed my mind|actually|instead|forget (that|it|what)|"
                       r"now (i|it's|its|we)|these days|used to|stopped|switched|moved to)\b", re.I)
_NEG = re.compile(r"\b(hate|dislike|don'?t|do not|never|can'?t stand|stopped|not a fan|no longer|quit)\b", re.I)
_POS = re.compile(r"\b(love|like|enjoy|prefer|adore|into|fan of|always)\b", re.I)
_STOP = set("""a an the and or but if then so to of in on at for with from by is are was were be been am i
you he she it we they me my your our their this that these those do does did have has had not no yes
just really very about what when where why how which who would could should will can im i'm it's its
dont don't like love hate""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9؀-ۿ']+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


def polarity(text: str) -> int:
    neg, pos = bool(_NEG.search(text or "")), bool(_POS.search(text or ""))
    if neg and not pos:
        return -1
    if pos and not neg:
        return 1
    return 0


def parse_due(value: Any, tz: str, base: float | None = None) -> float | None:
    """ISO-8601 (naive = persona local time) or relative ('in 2 hours', '30m')."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    base = base or time.time()
    m = re.fullmatch(r"(?:in\s+)?(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|"
                     r"d|days?|w|weeks?)", text, re.I)
    if m:
        n = float(m.group(1))
        unit = m.group(2).lower()[0]
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        return base + n * mult
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None and ZoneInfo is not None:
        try:
            dt = dt.replace(tzinfo=ZoneInfo(tz or "UTC"))
        except Exception:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.timestamp()


class Memory:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    # ------------------------------------------------------------------ journal
    async def new_turn(self, pid: int, user_text: str, *, channel: str = "app", surface: str = "",
                       kind: str = "chat", started_at: float | None = None, gap: float = 0.0) -> int:
        return await self.store.insert("turns", persona_id=pid, user_text=user_text, channel=channel,
                                       surface=surface[:120], kind=kind, started_at=started_at or now(),
                                       gap_seconds=gap)

    async def finish_turn(self, turn_id: int, reply: str) -> None:
        await self.store.update("turns", turn_id, reply_text=reply, ended_at=now())

    async def tail(self, pid: int, n: int = 0) -> list[dict]:
        # n <= 0 means no arbitrary recent-turn cutoff.
        sql = "SELECT * FROM turns WHERE persona_id = ? AND ended_at > 0 AND kind != 'observation' ORDER BY id DESC"
        rows = (await self.store.all(sql + " LIMIT ?", pid, n) if n > 0
                else await self.store.all(sql, pid))
        return list(reversed(rows))

    async def user_sources(self, pid: int, *, last: int = 30, extra: list[str] | None = None) -> list[str]:
        rows = await self.store.all(
            "SELECT user_text FROM turns WHERE persona_id = ? AND kind = 'chat' ORDER BY id DESC LIMIT ?", pid, last)
        return [*(extra or []), *[r["user_text"] for r in rows]]

    # ------------------------------------------------------------------ events
    async def new_event(self, pid: int, turn_id: int | None, *, chord: dict | None = None,
                        surface: str = "") -> int:
        return await self.store.insert("events", persona_id=pid, turn_id=turn_id,
                                       chord=jdump(emotion.normalize(chord)), surface=surface[:120],
                                       created_at=now())

    async def set_event_chord(self, event_id: int, chord: dict) -> None:
        await self.store.update("events", event_id, chord=jdump(emotion.normalize(chord)))

    async def add_layers(self, pid: int, event_id: int, items: list[dict], *,
                         sources: list[str]) -> list[int]:
        """Validate (quote-or-die, one writer per layer), embed, store, supersede."""
        clean: list[dict] = []
        for it in items:
            layer = str(it.get("layer", "")).strip()
            author = LAYER_AUTHOR.get(layer)
            if not author or it.get("author", author) != author:
                continue  # wrong author for this layer: dropped
            text = str(it.get("text", "")).strip()[:1200]
            quote_in = str(it.get("quote", "") or "").strip()
            quote = ""
            if quote_in:
                quote = ground(quote_in, sources) or ""
                if not quote:
                    log.info("quote-or-die drop (%s/%s): %r", layer, author, quote_in[:80])
                    continue
            # user-attributed layers must carry a quote
            if layer in ("note", "reading", "collateral") and not quote:
                continue
            if not text:
                text = quote
            if not text:
                continue
            kind = str(it.get("kind", "detail")).strip().lower() or "detail"
            if kind not in NOTE_KINDS:
                kind = "detail"
            try:
                salience = max(1, min(3, int(it.get("salience", 1))))
            except (TypeError, ValueError):
                salience = 1
            clean.append({"layer": layer, "author": author, "kind": kind, "text": text, "quote": quote,
                          "anchor": str(it.get("anchor", "") or "")[:200],
                          "chord": emotion.normalize(it.get("chord")), "salience": salience,
                          "tender": 1 if it.get("tender") in (True, 1, "true", "yes") else 0})
        if not clean:
            return []
        embed_texts = [f"{c['anchor']} | {c['text']} | {c['quote']}" for c in clean]
        model, vecs = await self.embedder.embed(embed_texts)
        ids = []
        ts = now()
        for c, v, t in zip(clean, vecs, embed_texts):
            ids.append(await self.store.insert(
                "layers", persona_id=pid, event_id=event_id, layer=c["layer"], author=c["author"],
                kind=c["kind"], text=c["text"], quote=c["quote"], anchor=c["anchor"], chord=jdump(c["chord"]),
                salience=c["salience"], tender=c["tender"], embedding=to_blob(v), embed_model=model,
                hvec=to_blob(hash_embed(t)), created_at=ts))
        await self._supersede(pid, ids)
        return ids

    async def _supersede(self, pid: int, new_ids: list[int]) -> None:
        if not new_ids:
            return
        qs = ",".join("?" for _ in new_ids)
        fresh = await self.store.all(f"SELECT * FROM layers WHERE id IN ({qs})", *new_ids)
        for new in fresh:
            if new["kind"] not in SUPERSEDABLE:
                continue
            olds = await self.store.all(
                "SELECT * FROM layers WHERE persona_id = ? AND status = 'active' AND id < ? AND kind IN "
                "('preference','plan','detail','fact') ORDER BY id DESC", pid, new["id"])
            new_words = content_words(new["text"] + " " + new["quote"] + " " + new["anchor"])
            new_v = from_blob(new["hvec"])
            new_said = new["quote"] or new["text"]
            for old in olds:
                if old["event_id"] == new["event_id"]:
                    continue
                old_words = content_words(old["text"] + " " + old["quote"] + " " + old["anchor"])
                # overlap coefficient: short notes share a topic if most of the smaller one is shared
                overlap = len(new_words & old_words) / max(1, min(len(new_words), len(old_words)))
                sim = cosine(new_v, from_blob(old["hvec"]))
                if overlap < 0.34 and sim < 0.5:
                    continue
                revised = bool(_REVISION.search(new_said))
                flipped = polarity(new_said) * polarity(old["quote"] or old["text"]) == -1
                if revised or flipped:
                    await self.store.update("layers", old["id"], status="superseded", superseded_by=new["id"])

    async def reinterpret(self, pid: int, layer_id: int, reading: str, author: str = "C") -> bool:
        """Reconsolidation: a recalled memory gains a new reading; the quote never changes."""
        row = await self.store.one("SELECT interpretations FROM layers WHERE id = ? AND persona_id = ?",
                                   layer_id, pid)
        if not row or not reading.strip():
            return False
        hist = jload(row["interpretations"], []) or []
        hist.append({"at": now(), "by": author, "reading": reading.strip()[:400]})
        await self.store.update("layers", layer_id, interpretations=jdump(hist[-12:]))
        return True

    async def mark_recalled(self, pid: int, layer_ids: list[int]) -> None:
        if not layer_ids:
            return
        ts = now()
        qs = ",".join("?" for _ in layer_ids)
        await self.store.run(
            f"UPDATE layers SET recalled_count = recalled_count + 1, last_recalled_at = ? "
            f"WHERE persona_id = ? AND id IN ({qs})", ts, pid, *layer_ids)

    # ------------------------------------------------------------------ entities
    async def upsert_entity(self, pid: int, name: str, kind: str, account: str, quote: str) -> int | None:
        name = (name or "").strip()[:80]
        if not name:
            return None
        row = await self.store.one("SELECT * FROM entities WHERE persona_id = ? AND lower(name) = lower(?)",
                                   pid, name)
        if row:
            acc = row["account"]
            if account and account not in acc:
                acc = (acc + " | " + account).strip(" |")[-1500:]
            await self.store.update("entities", row["id"], account=acc, last_quote=quote or row["last_quote"],
                                    mention_count=row["mention_count"] + 1, updated_at=now())
            return row["id"]
        return await self.store.insert("entities", persona_id=pid, name=name, kind=(kind or "person")[:30],
                                       account=(account or "")[:1500], last_quote=quote or "", updated_at=now())

    async def entities_in(self, pid: int, text: str) -> list[dict]:
        low = (text or "").lower()
        rows = await self.store.all("SELECT * FROM entities WHERE persona_id = ? ORDER BY mention_count DESC", pid)
        return [r for r in rows if re.search(r"\b" + re.escape(r["name"].lower()) + r"\b", low)]

    # ------------------------------------------------------------------ psyche
    @staticmethod
    def effective_confidence(h: dict, half_life_days: float = 0.0, at: float | None = None) -> float:
        """Stubborn memory: confidence does NOT fade with time. It only moves on confirm/contradict.
        ``half_life_days`` is kept in the signature for backward compatibility and ignored."""
        return float(h["confidence"])

    async def hypotheses(self, pid: int, *, status: str = "active") -> list[dict]:
        return await self.store.all("SELECT * FROM hypotheses WHERE persona_id = ? AND status = ? "
                                    "ORDER BY confidence DESC, id DESC", pid, status)

    async def apply_hypotheses(self, pid: int, ops: list[dict], sources: list[str]) -> dict:
        """C's psyche layer: new / confirm / contradict - each needs grounded quotes."""
        stats = {"new": 0, "confirm": 0, "contradict": 0, "dropped": 0}
        ts = now()
        for op in ops or []:
            kind = str(op.get("op", "new")).lower()
            quotes = [g for g in (ground(q, sources) for q in (op.get("quotes") or [])[:4] if isinstance(q, str)) if g]
            if not quotes:
                stats["dropped"] += 1
                continue
            if kind == "new":
                claim = str(op.get("claim", "")).strip()[:400]
                if not claim:
                    stats["dropped"] += 1
                    continue
                model, vec = await self.embedder.embed_one(claim)
                await self.store.insert("hypotheses", persona_id=pid, claim=claim, quotes=jdump(quotes),
                                        confidence=min(0.5, 0.3 + 0.07 * len(quotes)), embedding=to_blob(vec),
                                        embed_model=model, hvec=to_blob(hash_embed(claim)),
                                        last_confirmed_at=ts, created_at=ts)
                stats["new"] += 1
                continue
            try:
                hid = int(op.get("id"))
            except (TypeError, ValueError):
                stats["dropped"] += 1
                continue
            h = await self.store.one("SELECT * FROM hypotheses WHERE id = ? AND persona_id = ? AND status='active'",
                                     hid, pid)
            if not h:
                stats["dropped"] += 1
                continue
            all_quotes = (jload(h["quotes"], []) + quotes)[-12:]
            if kind == "confirm":
                c = float(h["confidence"])
                await self.store.update("hypotheses", hid, confidence=min(0.95, c + (1 - c) * 0.25),
                                        confirmations=h["confirmations"] + 1, quotes=jdump(all_quotes),
                                        last_confirmed_at=ts)
                stats["confirm"] += 1
            elif kind == "contradict":
                c = float(h["confidence"]) * 0.55
                revised_claim = str(op.get("revised_claim", "") or "").strip()[:400]
                if c < 0.15 or revised_claim:
                    await self.store.update("hypotheses", hid, status="revised", confidence=c,
                                            contradictions=h["contradictions"] + 1)
                    if revised_claim:
                        model, vec = await self.embedder.embed_one(revised_claim)
                        await self.store.insert("hypotheses", persona_id=pid, claim=revised_claim,
                                                quotes=jdump(quotes), confidence=0.35, revision_of=hid,
                                                embedding=to_blob(vec), embed_model=model,
                                                hvec=to_blob(hash_embed(revised_claim)),
                                                last_confirmed_at=ts, created_at=ts)
                else:
                    await self.store.update("hypotheses", hid, confidence=c, contradictions=h["contradictions"] + 1,
                                            quotes=jdump(all_quotes))
                stats["contradict"] += 1
            else:
                stats["dropped"] += 1
        return stats

    async def fade_hypotheses(self, pid: int, half_life_days: float = 0.0, floor: float = 0.0) -> int:
        """Stubborn memory: nothing is faded on time. Hypotheses only move state when the record itself
        changes (contradiction with a revised claim); nothing is ever quietly aged out."""
        return 0

    # ------------------------------------------------------------------ loops (B)
    async def open_loops(self, pid: int) -> list[dict]:
        return await self.store.all("SELECT * FROM loops WHERE persona_id = ? AND status = 'open' "
                                    "ORDER BY COALESCE(due_at, 9e18), id", pid)

    async def apply_loops(self, pid: int, ops: list[dict], sources: list[str], *, tz: str,
                          turn_id: int | None) -> dict:
        stats = {"opened": 0, "closed": 0, "dropped": 0}
        ts = now()
        for op in ops or []:
            kind = str(op.get("op", "open")).lower()
            if kind == "open":
                text = str(op.get("text", "")).strip()[:300]
                quote = ground(str(op.get("quote", "")), sources) or ""
                if not text or not quote:
                    stats["dropped"] += 1
                    continue
                await self.store.insert("loops", persona_id=pid, kind=str(op.get("kind", "task"))[:20],
                                        text=text, quote=quote, due_at=parse_due(op.get("due"), tz, ts),
                                        turn_id=turn_id, created_at=ts, updated_at=ts)
                stats["opened"] += 1
            elif kind in ("done", "missed", "cancel", "cancelled"):
                try:
                    lid = int(op.get("id"))
                except (TypeError, ValueError):
                    stats["dropped"] += 1
                    continue
                status = {"cancel": "cancelled"}.get(kind, kind)
                n = await self.store.run("UPDATE loops SET status = ?, updated_at = ? WHERE id = ? AND "
                                         "persona_id = ? AND status = 'open'", status, ts, lid, pid)
                stats["closed" if n else "dropped"] += 1
            elif kind == "reschedule":
                try:
                    lid = int(op.get("id"))
                except (TypeError, ValueError):
                    continue
                due = parse_due(op.get("due"), tz, ts)
                if due:
                    await self.store.run("UPDATE loops SET due_at = ?, notified_at = 0, updated_at = ? WHERE id = ? "
                                         "AND persona_id = ?", due, ts, lid, pid)
        return stats

    async def due_loops(self, pid: int, at: float | None = None) -> list[dict]:
        return await self.store.all(
            "SELECT * FROM loops WHERE persona_id = ? AND status = 'open' AND due_at IS NOT NULL AND due_at <= ? "
            "AND notified_at = 0 ORDER BY due_at", pid, at or now())

    # ------------------------------------------------------------------ verdicts (B -> A)
    # ---------------------------------------------------------------- standing requests
    async def rules(self, pid: int) -> list[dict]:
        return await self.store.all("SELECT * FROM rules WHERE persona_id = ? AND status = 'active' ORDER BY id", pid)

    async def apply_rules(self, pid: int, ops: list[dict], sources: list[str]) -> int:
        """B keeps the person's standing requests. Adding one needs their exact words."""
        n = 0
        for op in ops:
            kind = str(op.get("op", "add")).lower()
            if kind == "drop":
                try:
                    rid = int(op.get("id"))
                except (TypeError, ValueError):
                    continue
                n += await self.store.run("UPDATE rules SET status='dropped' WHERE id = ? AND persona_id = ?", rid, pid)
                continue
            text = str(op.get("text", "")).strip()[:300]
            quote = ground(str(op.get("quote", "") or ""), sources)
            if not text or not quote:
                continue
            if await self.store.one("SELECT id FROM rules WHERE persona_id = ? AND status='active' AND text = ?",
                                    pid, text):
                continue
            await self.store.insert("rules", persona_id=pid, text=text, quote=quote, created_at=now())
            n += 1
        return n

    async def record_verdict(self, pid: int, turn_id: int | None, level: str, rationale: str,
                             findings: list) -> int:
        level = level if level in ("GREEN", "YELLOW", "RED") else "YELLOW"
        return await self.store.insert("verdicts", persona_id=pid, turn_id=turn_id, level=level,
                                       rationale=rationale[:600], findings=jdump(findings[:8]), created_at=now())

    async def integrity(self, pid: int, n: int = 3) -> dict:
        rows = await self.store.all("SELECT * FROM verdicts WHERE persona_id = ? ORDER BY id DESC LIMIT 200", pid)
        streak = 0
        for r in rows:
            if r["level"] != "GREEN":
                break
            streak += 1
        total = len(rows)
        greens = sum(1 for r in rows if r["level"] == "GREEN")
        return {"recent": rows[:n], "green_streak": streak, "checked": total,
                "clean_rate": round(greens / total, 2) if total else None}

    # ------------------------------------------------------------------ audit
    async def inspect(self, pid: int, limit: int = 200) -> dict:
        layers = await self.store.all(
            "SELECT id, event_id, layer, author, kind, text, quote, anchor, chord, salience, status, "
            "interpretations, tender, created_at FROM layers WHERE persona_id = ? AND status != 'wiped' "
            "ORDER BY id DESC LIMIT ?", pid, limit)
        for row in layers:
            row["chord"] = jload(row["chord"], {})
            row["interpretations"] = jload(row["interpretations"], [])
        hyps = await self.store.all("SELECT id, claim, quotes, confidence, confirmations, status, created_at "
                                    "FROM hypotheses WHERE persona_id = ? ORDER BY id DESC", pid)
        for h in hyps:
            h["quotes"] = jload(h["quotes"], [])
        return {
            "layers": layers, "hypotheses": hyps,
            "entities": await self.store.all("SELECT * FROM entities WHERE persona_id = ?", pid),
            "loops": await self.store.all("SELECT * FROM loops WHERE persona_id = ? ORDER BY id DESC LIMIT 100", pid),
            "standing_requests": await self.rules(pid),
            "thoughts": await self.store.all("SELECT id, source, text, status, created_at FROM thoughts "
                                             "WHERE persona_id = ? ORDER BY id DESC LIMIT 50", pid),
        }

    async def wipe(self, pid: int, layer_id: int) -> bool:
        """User-requested forgetting: the row's content is erased for real."""
        n = await self.store.run("UPDATE layers SET status='wiped', text='', quote='', anchor='', "
                                 "interpretations='[]', embedding=NULL, hvec=NULL WHERE id = ? AND persona_id = ?",
                                 layer_id, pid)
        return bool(n)


def vec_matrix(rows: list[dict], field: str = "embedding") -> np.ndarray:
    vecs = [from_blob(r[field]) for r in rows]
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    dim = max((v.shape[0] for v in vecs if v is not None), default=0)
    mat = np.zeros((len(vecs), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        if v is not None and v.shape[0] == dim:
            mat[i] = v
    return mat

