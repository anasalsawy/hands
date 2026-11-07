"""B's internal Conversation Board, not a drawing or profile.
B writes an evolving representation to help L understand the entire person,
their circumstances, changing priorities, causal relationships and uncertainty.

Only B authors meaning. Code validates representation, caches it, and persists
revisions. The response hot path only reads an immutable in-memory snapshot.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from .llm import extract_json

log = logging.getLogger(__name__)

BOARD_INSTRUCTIONS = """You are B, privately maintaining YOUR OWN Conversation Board for L.
THIS IS NOT CONTENT FOR THE USER. Do not perform, flatter, decorate, make an
infographic, write a public biography, or fill out a fixed questionnaire.
Think like a thoughtful colleague maintaining a private working understanding
of a complicated person and their world. Its purpose is to help L understand
meaning and respond intelligently. You choose what structure and prose convey
that information; don't force a tree or standardized life categories.

Include EVERYTHING SUPPORTED AND RELEVANT TO UNDERSTANDING CONVERSATION:
 enduring background, values, history, people, experiences and constraints;
 medium-term situations, relationships, projects, burdens, aims and changes;
 immediate state, recent conversation, active concerns and new decisions.
Go beyond shallow event listings. Articulate what matters, why, how intensely,
how long it has mattered, its consequences, dependencies and cross-connections.
Describe priorities, tensions, contradictions, uncertainty, evolving patterns,
what was resolved, what remains unsettled and what L should keep in mind.
Keep domains beyond personal life: professional work, practical obligations,
plans, beliefs, conversational preferences, and whatever else the evidence bears.
Don't overfocus on the latest turn, or delete old stable context merely because
the topic changed. Distinguish user's stated facts, derived readings, and unknowns.
Never invent experiences or claim an inference is a fact. Avoid intrusive or
unnecessary personal details; do not store passwords, tokens or other secrets.

You are not bound to a deterministic list of domains. Build rich interrelated
entries in whatever structure conveys meaning best, and preserve important
stable identifiers across edits. You are allowed to write compact prose when
that is more expressive than links. Don't include unsupported diagnostic
labels, judgments about character or exaggerated emotional inferences.

You MUST NOT generate a replacement board. You are an editor of the existing
board, not its author from scratch. The full previous board stays authoritative.
For this update return ONLY a small bounded PATCH of operations:
{"operations":[
 {"op":"add","id":"unique_new_id","title":"...","understanding":"...",
  "horizon":"stable|ongoing|immediate","weight":"low|medium|high|critical",
  "confidence":"stated|inferred|uncertain","evidence":"source anchor",
  "connections":[{"to":"existing_or_new_id","meaning":"..."}]},
 {"op":"enrich","id":"existing_id","additional_understanding":"new detail to absorb",
  "evidence":"supporting source","weight":"optional weight","horizon":"optional horizon"},
 {"op":"connect","id":"existing_id","to":"other_existing_id","meaning":"relation"},
 {"op":"correct","id":"existing_id","replacement_understanding":"specific corrected information",
  "evidence":"clear contradiction supported by the user"},
 {"op":"retire","id":"existing_id","evidence":"clear explicit resolution or user correction"}
]}
Only change what the latest evidence genuinely warrants. Prior meaning and links
must not disappear on paraphrase, topic switch, or context truncation. An
'enrich' appends/absorbs knowledge without deleting old information. Do not
emit delete/retire unless there is direct explicit evidence that the old entry
ceased to be valid. Do not create broad replacement summaries for old entries.
Empty operations is correct when nothing material changed.
The patch schema is transport, not the reasoning method. Your entries, their groupings,
relationships, relative weight and prose are MODEL GENERATED, not rule-generated.
A status/claim is never 'verified completed' solely because a model wrote it.
"""

def _limit(value, n):
    return str(value or "").strip()[:n]

class ConversationBoard:
    def __init__(self, timeline, bus=None):
        self.timeline = timeline
        self.bus = bus
        self._boards: dict[int, dict] = {}
        self._prompt: dict[int, tuple[str, str, str]] = {}
        self._revisions: dict[int, int] = {}
        self._prompt_days: dict[int, int] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._pending: dict[int, list[str]] = {}
        self._reviewed: set[int] = set()

    @staticmethod
    def _blank():
        return {"orientation": "", "entries": [], "current_reading": "", "unresolved": [], "change_note": ""}

    def peek(self, pid: int) -> dict:
        """Return last prepared board; zero I/O, no await."""
        return self._boards.get(pid, self._blank())

    def revision(self, pid: int) -> int:
        return self._revisions.get(pid, 0)

    def for_prompt(self, pid: int) -> str:
        """L reads a pre-rendered board, not a DB document or generated image."""
        return "\n\n".join(part for part in self.prompt_tiers(pid) if part)

    def prompt_tiers(self, pid: int) -> tuple[str, str, str]:
        day = int(time.time()) // 86400
        if pid in self._boards and self._prompt_days.get(pid) != day:
            self._prompt[pid] = self._render(self._boards[pid], day=day)
            self._prompt_days[pid] = day
        return self._prompt.get(pid, ("", "", ""))

    @staticmethod
    def _validate(raw: dict) -> dict:
        if not isinstance(raw, dict):
            raise ValueError("Conversation Board must be a JSON object")
        entries, seen = [], set()
        for entry in (raw.get("entries") or [])[:5000]:
            if not isinstance(entry, dict):
                continue
            ident = _limit(entry.get("id"), 72)
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,72}", ident) or ident in seen:
                continue
            title = _limit(entry.get("title"), 180)
            understanding = _limit(entry.get("understanding"), 1400)
            horizon = _limit(entry.get("horizon"), 24)
            if not title or not understanding or horizon not in ("stable", "ongoing", "immediate"):
                continue
            seen.add(ident)
            weight = _limit(entry.get("weight"), 16)
            confidence = _limit(entry.get("confidence"), 16)
            entries.append({"id": ident, "title": title, "understanding": understanding,
                            "horizon": horizon,
                            "updated_day": entry.get("updated_day") if type(entry.get("updated_day")) is int else None,
                            "weight": weight if weight in ("low","medium","high","critical") else "medium",
                            "confidence": confidence if confidence in ("stated","inferred","uncertain") else "uncertain",
                            "evidence": _limit(entry.get("evidence"), 400),
                            "connections": entry.get("connections") or []})
        for entry in entries:
            links = []
            for link in entry["connections"][:200]:
                if not isinstance(link, dict):
                    continue
                target = _limit(link.get("to"), 72)
                if target in seen and target != entry["id"]:
                    links.append({"to": target, "meaning": _limit(link.get("meaning"), 300)})
            entry["connections"] = links
        return {
            "orientation": _limit(raw.get("orientation"), 4000),
            "entries": entries,
            "current_reading": _limit(raw.get("current_reading"), 3500),
            "unresolved": [_limit(s, 400) for s in (raw.get("unresolved") or [])[:25] if isinstance(s, str)],
            "change_note": _limit(raw.get("change_note"), 1100),
        }

    prompt_label = "CONVERSATION BOARD"

    def _render(self, board: dict, *, day: int | None = None) -> tuple[str, str, str]:
        day = int(time.time()) // 86400 if day is None else day
        sections = {horizon: [] for horizon in ("stable", "ongoing", "immediate")}
        for entry in board["entries"]:
            horizon = entry["horizon"]
            updated_day = entry.get("updated_day")
            age = 0 if updated_day is None else day - updated_day
            tier = horizon
            if horizon != "stable" and age >= 30:
                tier = "stable"
            elif horizon == "immediate" and age >= 7:
                tier = "ongoing"
            lines = sections[tier]
            history = f"historical {horizon}; " if tier != horizon else ""
            lines.append(f"[{entry['id']}] ({history}impact={entry['weight']}; "
                         f"confidence={entry['confidence']}) {entry['title']}: {entry['understanding']}"
                         + (f" | anchor: {entry['evidence']}" if entry["evidence"] else ""))
            for link in entry["connections"]:
                lines.append(f"  relates to {link['to']}: {link['meaning']}")
        active = sections["immediate"]
        for label, key in (("Orientation", "orientation"), ("Current reading", "current_reading")):
            if board[key]:
                active.append(label + ": " + board[key])
        if board["unresolved"]:
            active.append("Unresolved: " + " | ".join(board["unresolved"]))
        stable, ongoing, immediate = (
            (f"INTERNAL {self.prompt_label} ({horizon})\n" + "\n".join(sections[horizon]))[:budget]
            if sections[horizon] else ""
            for horizon, budget in (("stable", 10000), ("ongoing", 9000), ("immediate", 6000))
        )
        return stable, ongoing, immediate

    @classmethod
    def _apply_patch(cls, previous: dict, operations: list, current_words: str) -> dict:
        """Monotonic accumulation: B may add/enrich; correction needs explicit evidence.

        Existing entries and relations persist untouched unless a narrowly
        evidenced correction or retirement was explicitly returned.
        """
        data = json.loads(json.dumps(previous))
        entries = data["entries"]
        ids = {e["id"] for e in entries}
        user_lower = current_words.lower()
        changed = False
        for op in operations[:32]:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op") or "")
            ident = str(op.get("id") or "")
            entry = next((e for e in entries if e["id"] == ident), None)
            evidence = _limit(op.get("evidence"), 400)
            # No model-generated correction or deletion without a direct
            # supporting phrase from the latest user turn.
            grounded = bool(evidence and len(evidence) >= 8
                            and evidence.lower() in user_lower)
            if kind == "add" and entry is None and len(entries) < 5000:
                candidate = cls._validate({"entries": [op]})["entries"]
                if candidate and grounded:
                    item = candidate[0]
                    item["connections"] = []
                    entries.append(item)
                    ids.add(item["id"])
                    changed = True
            elif kind == "enrich" and entry is not None and grounded:
                detail = _limit(op.get("additional_understanding"), 650)
                if detail and detail.lower() not in entry["understanding"].lower():
                    combined = entry["understanding"] + " | " + detail
                    if len(combined) <= 1400:
                        entry["understanding"] = combined
                        entry["evidence"] = (entry["evidence"] + " | " + evidence)[-400:]
                    else:
                        # Preserve old meaning and append a connected continuation.
                        continuation_id = (ident[:54] + "_ext_" + str(len(entries)))[:70]
                        if continuation_id in ids or len(entries) >= 5000:
                            continue
                        entries.append({
                            "id": continuation_id,
                            "title": "Continuation: " + entry["title"][:150],
                            "understanding": detail,
                            "horizon": entry["horizon"], "weight": entry["weight"],
                            "confidence": entry["confidence"], "evidence": evidence,
                            "connections": [{"to": ident, "meaning": "extends without replacing"}],
                        })
                        ids.add(continuation_id)
                        entry["connections"].append({
                            "to": continuation_id, "meaning": "continued detail"})
                    new_weight = str(op.get("weight") or "")
                    if new_weight in ("low","medium","high","critical"):
                        entry["weight"] = new_weight
                    new_horizon = str(op.get("horizon") or "")
                    if new_horizon in ("stable","ongoing","immediate"):
                        entry["horizon"] = new_horizon
                    changed = True
            elif kind == "connect" and entry is not None:
                target = _limit(op.get("to"), 72)
                meaning = _limit(op.get("meaning"), 300)
                if target in ids and target != ident and meaning and not any(
                    c["to"] == target and c["meaning"] == meaning for c in entry["connections"]
                ):
                    entry["connections"].append({"to": target, "meaning": meaning})
                    changed = True
            elif kind == "correct" and entry is not None and grounded:
                replacement = _limit(op.get("replacement_understanding"), 1400)
                if replacement and replacement != entry["understanding"] and len(entries) < 5000:
                    # Never overwrite a prior thought. Preserve it, and attach
                    # the explicit correction as a new authoritative linked fact.
                    correction_id = (ident[:50] + "_correction_" + str(len(entries)))[:70]
                    if correction_id not in ids:
                        entries.append({
                            "id": correction_id, "title": "Correction to " + entry["title"][:150],
                            "understanding": replacement,
                            "horizon": entry["horizon"], "weight": entry["weight"],
                            "confidence": "stated", "evidence": evidence,
                            "connections": [{"to": ident, "meaning": "supersedes earlier interpretation"}],
                        })
                        ids.add(correction_id)
                        entry["connections"].append({
                            "to": correction_id,
                            "meaning": "correction takes precedence over this earlier statement"})
                        changed = True
            elif kind == "retire" and entry is not None and grounded:
                # Never silently erase history: resolved items are marked with
                # the new evidence rather than deleted from the board.
                entry["understanding"] = (entry["understanding"] +
                    " | Resolved or superseded: " + evidence)[:1400]
                entry["horizon"] = "stable"
                changed = True
        if not changed:
            return previous
        # Keep B's existing orientation unless explicitly amended by a separate
        # grounded operation; never wholesale replace it.
        previous_entries = {entry["id"]: entry for entry in previous["entries"]}
        day = int(time.time()) // 86400
        for entry in entries:
            if entry != previous_entries.get(entry["id"]):
                entry["updated_day"] = day
        return cls._validate(data)

    async def warm(self):
        """Load persisted snapshots once on startup; reads never block L."""
        await self.timeline.initialize()
        rows = await self.timeline.store.all(
            "SELECT t.persona_id,t.seq,t.payload FROM shared_timeline t "
            "INNER JOIN (SELECT persona_id,MAX(seq) AS seq FROM shared_timeline "
            "WHERE kind='conversation_board_state' GROUP BY persona_id) last "
            "ON last.persona_id=t.persona_id AND last.seq=t.seq")
        for row in rows:
            try:
                data = self._validate(json.loads(row["payload"]))
                pid = int(row["persona_id"])
                self._boards[pid] = data
                self._prompt_days.pop(pid, None)
                self._revisions[pid] = int(row["seq"])
                self._reviewed.add(pid)
            except (TypeError, ValueError, KeyError):
                log.warning("Invalid saved conversation board for persona %s", row["persona_id"])

    async def evolve(self, pid: int, llm, *, user_text: str, reply: str,
                     recalled: str = "", tool_facts: list[str] | None = None):
        """Run B after the answer and publish immutable snapshot when evidence changes."""
        if not user_text.strip() or len(user_text.strip()) < 10:
            return
        async with self._locks.setdefault(pid, asyncio.Lock()):
            queue = self._pending.setdefault(pid, [])
            queue.append(user_text[:4000])
            queue[:] = queue[-18:]
            # Do not pay for a generative B call on every conversational turn.
            # Initial board is created once; later edits absorb batches.
            urgent = any(w in user_text.lower() for w in (
                "remember this", "don't forget", "important decision",
                "we decided", "from now on", "never again",
            ))
            if (self.revision(pid) or pid in self._reviewed) and len(queue) < 4 and not urgent:
                return
            try:
                memories, turns = await asyncio.gather(
                    self.timeline.store.all(
                        "SELECT text,quote,kind,created_at FROM layers "
                        "WHERE persona_id=? AND status='active' "
                        "ORDER BY salience DESC,id DESC LIMIT 100", pid),
                    self.timeline.store.all(
                        "SELECT user_text,started_at FROM turns "
                        "WHERE persona_id=? AND kind='chat' ORDER BY id DESC LIMIT 20", pid))
                evidence = {
                    "previous_internal_board": self.peek(pid),
                    "durable_evidence": memories,
                    "recent_conversation": turns,
                    "new_user_messages_since_last_revision": queue[:],
                    "assistant_response": reply[:2000],
                    "memories_retrieved": recalled[:6500],
                    "tool_result_evidence": (tool_facts or [])[:12],
                }
                result = await llm.chat([
                    {"role":"system","content": BOARD_INSTRUCTIONS},
                    {"role":"user","content":json.dumps(evidence,ensure_ascii=False,default=str)[:40000]},
                ], temperature=0.2, max_tokens=2400, json_mode=True)
                patch = extract_json(getattr(result,"text","") or "") or {}
                ops = patch.get("operations") if isinstance(patch, dict) else None
                if not isinstance(ops, list):
                    log.warning("B returned no valid incremental board operations")
                    return
                self._reviewed.add(pid)
                if not ops:
                    # B inspected this batch and found no material changes.
                    # A no-op must not trigger another paid call next turn.
                    queue.clear()
                    return
                previous = self.peek(pid)
                grounding_text = "\n".join(
                    queue + [str(m.get("quote") or "") for m in memories] +
                    [str(t.get("user_text") or "") for t in turns])
                data = self._apply_patch(previous, ops, grounding_text)
                if data == previous:
                    # Valid reviewed batch but nothing applied: don't endlessly
                    # retry the same events at B-model cost.
                    queue.clear()
                    return
                saved = await self.timeline.append(
                    pid,producer="B",kind="conversation_board_state",payload=data,
                    supersedes_seq=self.revision(pid) or None)
                self._boards[pid] = data
                self._prompt_days.pop(pid, None)
                self._revisions[pid] = saved["seq"]
                queue.clear()
                if self.bus:
                    self.bus.publish(pid, {"type":"conversation_board_updated","revision":saved["seq"]})
            except Exception:
                log.exception("Conversation Board update failed; retaining cached version")
