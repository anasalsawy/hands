"""Decision-choice telemetry for L/R training, separate from Quranic curricula.

A policy choice is NOT a training label. Only independently reviewed decisions
with an external evidence reference are exported to an adapter dataset.
No messages, tool arguments, screen text, URLs or raw outputs are stored.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DECISION_MAP: dict[str, tuple[str, ...]] = {
    "task_route": ("direct_tool", "persistent_goal", "answer_without_tool"),
    "engagement": ("conversation_reserved", "execution_focused"),
    "lobe_allocation": ("conversation_only", "conversation_and_execution", "dual_execution",
                        "execute_and_plan", "execute_and_verify", "idle"),
    "execution_strategy": ("serial", "parallel"),
    "public_speech": ("speak", "stay_silent"),
    "tool_recovery": ("retry", "alternative", "blocked"),
    "task_verification": ("verified", "unverified", "unknown"),
    "task_lifecycle": ("continue", "done", "paused", "cancelled"),
}

# Whitelist bounded decision features. Unrecognized values are silently dropped;
# this prevents a caller from accidentally recording a user's text or a token.
_CONTEXT_BOOL = frozenset({"user_engaged", "requires_approval", "has_tool_evidence",
                           "has_independent_work", "was_interrupted", "has_new_information",
                           "tool_failed", "has_verifier", "role_switched"})
_CONTEXT_COUNT = frozenset({"independent_items", "tool_rounds", "attempt", "batch",
                            "beats", "verified_items"})
_CONTEXT_ENUMS = {
    "reason": frozenset({"repeated", "new_information", "tool_result", "tool_failure",
                         "no_tool", "user_request", "tool_choice", "verified",
                         "not_verified", "stale_draft", "explicit_stop",
                         "work_complete", "concurrent_work", "no_work"}),
    "resource": frozenset({"browser", "computer", "cloud", "other", "none"}),
    "mode": frozenset({"home", "inference", "task"}),
}
_EVIDENCE_KINDS = frozenset({"harness_test", "reviewed_trace", "reviewed_screenshot"})
_EVIDENCE_REF = re.compile(r"^(?:harness|review|test):[a-zA-Z0-9_.:-]{1,100}$")
_TASK_ID = re.compile(r"^[a-f0-9-]{8,64}$")


def safe_context(context: dict | None) -> dict:
    """Keep categorical and numeric facts; never keep prompt or tool contents."""
    result = {}
    for key, value in (context or {}).items():
        if key in _CONTEXT_BOOL and type(value) is bool:
            result[key] = value
        elif key in _CONTEXT_COUNT and type(value) is int and 0 <= value <= 10000:
            result[key] = value
        elif key in _CONTEXT_ENUMS and isinstance(value, str) and value in _CONTEXT_ENUMS[key]:
            result[key] = value
    return result


class DecisionJournal:
    """Append-only observations, separate reviewer decisions, filtered export.

    Async methods use a thread so logging cannot block the audio/event loop.
    The selected branch is what the system DID, not what it SHOULD have done.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS persona_decisions (
                id TEXT PRIMARY KEY, created_at REAL NOT NULL,
                persona_id INTEGER NOT NULL, task_id TEXT NOT NULL,
                point TEXT NOT NULL, options TEXT NOT NULL,
                selected TEXT NOT NULL, lobe TEXT NOT NULL,
                context TEXT NOT NULL, outcome TEXT NOT NULL DEFAULT 'unknown',
                review_status TEXT NOT NULL DEFAULT 'pending',
                evidence_kind TEXT NOT NULL DEFAULT '',
                evidence_ref TEXT NOT NULL DEFAULT ''
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS ix_persona_decisions_review "
                       "ON persona_decisions(review_status, point)")

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, *, persona_id: int, point: str, selected: str,
               task_id: str = "", lobe: str = "", context: dict | None = None) -> str:
        if point not in DECISION_MAP:
            raise ValueError("unknown decision point")
        if selected not in DECISION_MAP[point]:
            raise ValueError("choice is not an available option")
        if lobe not in ("", "L", "R") or type(persona_id) is not int:
            raise ValueError("invalid decision identity")
        ident = uuid.uuid4().hex
        task = task_id if isinstance(task_id, str) and _TASK_ID.fullmatch(task_id) else ""
        with self._db() as db:
            db.execute("""INSERT INTO persona_decisions
                (id, created_at, persona_id, task_id, point, options,
                 selected, lobe, context) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (ident, time.time(), persona_id, task, point,
                        json.dumps(DECISION_MAP[point]), selected, lobe,
                        json.dumps(safe_context(context), sort_keys=True)))
        return ident

    async def arecord(self, **kwargs) -> str:
        return await asyncio.to_thread(self.record, **kwargs)

    def review(self, decision_id: str, *, status: str, outcome: str,
               evidence_kind: str = "", evidence_ref: str = "") -> None:
        """An independent review, not an executor's self-evaluation.

        Accepted records require a reproducible external evidence identifier.
        This method deliberately does not infer acceptance from tool success.
        """
        if status not in ("accepted", "rejected") or outcome not in ("success", "failure", "unknown"):
            raise ValueError("invalid review status or outcome")
        if status == "accepted" and (evidence_kind not in _EVIDENCE_KINDS
                                      or not _EVIDENCE_REF.fullmatch(evidence_ref)):
            raise ValueError("accepted decisions require independent evidence")
        with self._db() as db:
            row = db.execute("SELECT review_status FROM persona_decisions WHERE id = ?",
                             (decision_id,)).fetchone()
            if row is None:
                raise KeyError(decision_id)
            if row[0] != "pending":
                raise ValueError("review has already been recorded")
            db.execute("""UPDATE persona_decisions
                       SET review_status=?, outcome=?, evidence_kind=?, evidence_ref=?
                       WHERE id=?""", (status, outcome,
                                       evidence_kind if status == "accepted" else "",
                                       evidence_ref if status == "accepted" else "",
                                       decision_id))

    async def areview(self, decision_id: str, **kwargs) -> None:
        await asyncio.to_thread(self.review, decision_id, **kwargs)

    def reviewed_samples(self, *, persona_id: int | None = None) -> list[dict]:
        """Accepted successes only; failures are review evidence, not SFT targets."""
        with self._db() as db:
            query = ("SELECT point, options, selected, lobe, context, outcome, "
                     "evidence_kind, evidence_ref FROM persona_decisions "
                     "WHERE review_status='accepted' AND outcome='success'")
            params: tuple = ()
            if persona_id is not None:
                query += " AND persona_id = ?"
                params = (persona_id,)
            query += " ORDER BY created_at, id"
            rows = db.execute(query, params).fetchall()
        return [{"stage": "specialist", "mode": "general", "task": "decision_policy",
                 "point": point, "candidates": json.loads(options), "selected": selected,
                 "role": lobe or "scheduler", "context": json.loads(context),
                 "outcome": outcome, "review_status": "accepted",
                 "evidence_kind": kind, "evidence_ref": reference}
                for point, options, selected, lobe, context, outcome, kind, reference in rows]

    def export_jsonl(self, path: str | Path, *, persona_id: int | None = None) -> int:
        """Export independently reviewed successes, never failures/pending choices."""
        samples = self.reviewed_samples(persona_id=persona_id)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as out:
            for sample in samples:
                out.write(json.dumps(sample, sort_keys=True) + "\n")
        return len(samples)
