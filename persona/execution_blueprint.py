"""B-authored cumulative execution guidance for L and R, separate from tool receipts.

B is the guide: Conversation Board informs L, Execution Blueprint informs R.
The deterministic Execution Sketchpad is evidence, not planning authority.
"""
from __future__ import annotations

import asyncio
import json
import logging

from .conversation_board import ConversationBoard
from .llm import extract_json

log = logging.getLogger(__name__)

EXECUTION_GUIDE = """You are B, the silent expensive-model guide for execution.
Maintain YOUR OWN internal Execution Blueprint, not a performance for the user,
not a tool log, not a task-status report, and not a fixed deterministic plan.
Your output helps L direct work and helps cheaper R execution agents work with
the project's complete context.

Your blueprint must comprehend the full current undertaking, not merely the
latest command: architectural purpose, scope, files/services/repositories,
constraints, dependencies, acceptance criteria, known failure modes, diagnostic
hypotheses, observed test results, commit/deployment evidence and unfinished
issues. Distinguish verified tool evidence from plans and assumptions. Explain
causal connections and what should be investigated or implemented next.
Do not treat a successful individual tool call as completed overall work.
When uncertain, explicitly preserve uncertainty rather than inventing a result.

The previous blueprint is authoritative historical context. NEVER replace it.
Produce only targeted incremental edits, so earlier knowledge is retained.
Do not add redundant detail or make another model call just to rephrase it.
Output JSON only:
{"operations": [
 {"op":"add","id":"stable_id","title":"concept",
  "understanding":"detailed guidance and why it matters",
  "horizon":"stable|ongoing|immediate","weight":"low|medium|high|critical",
  "confidence":"stated|inferred|uncertain",
  "evidence":"exact supporting substring of user request or tool result",
  "connections":[{"to":"existing_id","meaning":"dependency or relation"}]},
 {"op":"enrich","id":"existing_id","additional_understanding":"new detail",
  "evidence":"supporting substring"},
 {"op":"connect","id":"existing_id","to":"other_id","meaning":"dependency"},
 {"op":"correct","id":"existing_id","replacement_understanding":"correction",
  "evidence":"supporting substring"},
 {"op":"retire","id":"existing_id","evidence":"explicitly verified resolution"}
]}
Only explicit, grounded evidence can justify correction or retirement.
Do not create a new full blueprint or omit existing content.
"""

class ExecutionBlueprint(ConversationBoard):
    """Reuse monotonic revision mechanics, but not the Conversation Board's content."""
    def __init__(self, timeline, bus=None):
        super().__init__(timeline, bus)
        self._pending_events: dict[int, list[str]] = {}

    prompt_label = "EXECUTION BLUEPRINT"

    async def warm(self):
        await self.timeline.initialize()
        rows = await self.timeline.store.all(
            "SELECT t.persona_id,t.seq,t.payload FROM shared_timeline t "
            "INNER JOIN (SELECT persona_id,MAX(seq) AS seq FROM shared_timeline "
            "WHERE kind='execution_blueprint_state' GROUP BY persona_id) last "
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
                log.warning("Ignoring invalid Execution Blueprint for persona %s", row["persona_id"])

    async def evolve(self, pid: int, llm, *, user_text: str = "",
                     tool_facts: list[str] | None = None,
                     sketch: dict | None = None):
        if not (user_text.strip() or tool_facts):
            return
        async with self._locks.setdefault(pid, asyncio.Lock()):
            pending = self._pending_events.setdefault(pid, [])
            pending.extend(([user_text] if user_text.strip() else []) +
                           [str(v) for v in (tool_facts or [])])
            pending[:] = pending[-24:]
            evidence = (pending + [json.dumps(sketch or {}, ensure_ascii=False)])[:25]
            if (self.revision(pid) or pid in self._reviewed) and len(pending) < 3:
                return
            try:
                res = await llm.chat(
                    [{"role": "system", "content": EXECUTION_GUIDE},
                     {"role": "user", "content": json.dumps({
                         "previous_blueprint": self.peek(pid),
                         "recent_user_and_tool_evidence": evidence,
                         "execution_sketchpad": sketch or {},
                     }, ensure_ascii=False, default=str)[:35000]}],
                    temperature=0.2, max_tokens=2800, json_mode=True)
                patch = extract_json(getattr(res, "text", "") or "") or {}
                ops = patch.get("operations") if isinstance(patch, dict) else None
                if not isinstance(ops, list):
                    return
                self._reviewed.add(pid)
                if not ops:
                    pending.clear()
                    return
                source = "\n".join(pending)
                updated = self._apply_patch(self.peek(pid), ops, source)
                if updated == self.peek(pid):
                    # The batch was inspected; don't repeatedly call B for
                    # irrelevant events on every subsequent work step.
                    pending.clear()
                    return
                saved = await self.timeline.append(
                    pid, producer="B", kind="execution_blueprint_state", payload=updated,
                    supersedes_seq=self.revision(pid) or None)
                self._boards[pid] = updated
                self._prompt_days.pop(pid, None)
                self._revisions[pid] = saved["seq"]
                pending.clear()
                if self.bus:
                    self.bus.publish(pid, {"type": "execution_blueprint_updated", "revision": saved["seq"]})
            except Exception:
                log.exception("B's Execution Blueprint update failed; previous blueprint retained")
