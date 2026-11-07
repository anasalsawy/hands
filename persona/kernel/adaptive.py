"""Durable task state, process-safe resource ownership and adaptive strategy scores.

SQLite is the source of truth for resource ownership. The synchronous interface
remains usable in CLI/legacy code; async callers MUST use the a-prefixed
methods, which run SQLite work in a thread without blocking voice/audio loops.
"""
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    id: str
    persona_id: int
    goal: str
    status: str = "queued"
    strategy: str = "adaptive"
    checkpoint: dict = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class LeaseBusy(RuntimeError):
    """Another task, process, or unreconciled action owns this resource."""


class TaskLedger:
    """Crash-safe checkpoints and authoritative SQLite resource leases.

    A running action becomes UNKNOWN on startup, not finished. UNKNOWN leases
    deliberately block replay until a human/independent inspection reconciles
    them. This makes the resource *eventually* reclaimable without a blind retry.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        with self._lock, self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS adaptive_tasks (
              id TEXT PRIMARY KEY, persona_id INTEGER NOT NULL, goal TEXT NOT NULL,
              status TEXT NOT NULL, strategy TEXT NOT NULL, checkpoint TEXT NOT NULL,
              evidence TEXT NOT NULL, updated_at REAL NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS ix_adaptive_tasks_persona ON adaptive_tasks(persona_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS adaptive_outcomes (
              id INTEGER PRIMARY KEY, persona_id INTEGER NOT NULL, strategy TEXT NOT NULL,
              success INTEGER NOT NULL, cost REAL NOT NULL, recorded_at REAL NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS adaptive_actions (
              id TEXT PRIMARY KEY, persona_id INTEGER NOT NULL, task_id TEXT NOT NULL,
              resource TEXT NOT NULL, status TEXT NOT NULL, receipt TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS adaptive_resource_leases (
              persona_id INTEGER NOT NULL, resource TEXT NOT NULL, action_id TEXT NOT NULL,
              task_id TEXT NOT NULL, status TEXT NOT NULL, acquired_at REAL NOT NULL,
              PRIMARY KEY (persona_id, resource))""")
            # Migrate old advisory receipts conservatively. A historical action
            # that might have run must not be silently executed again.
            for aid, pid, tid, resource in db.execute(
                    """SELECT id,persona_id,task_id,resource FROM adaptive_actions
                       WHERE status IN ('running','unknown') ORDER BY rowid""").fetchall():
                db.execute("""INSERT OR IGNORE INTO adaptive_resource_leases
                              VALUES (?,?,?,?,?,?)""",
                           (pid, resource, aid, tid, "unknown", time.time()))
            db.execute("UPDATE adaptive_tasks SET status='paused' WHERE status='running'")
            db.execute("UPDATE adaptive_actions SET status='unknown' WHERE status='running'")
            db.execute("UPDATE adaptive_resource_leases SET status='unknown' WHERE status='active'")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _task(row) -> Task | None:
        if row is None:
            return None
        return Task(row[0], row[1], row[2], row[3], row[4],
                    json.loads(row[5]), json.loads(row[6]), row[7])

    def create(self, persona_id: int, goal: str, strategy: str = "adaptive") -> Task:
        if not goal.strip():
            raise ValueError("empty goal")
        task = Task(uuid.uuid4().hex, persona_id, goal.strip(), strategy=strategy)
        return self.save(task)

    def save(self, task: Task) -> Task:
        if task.status not in {"queued", "running", "paused", "blocked", "done", "cancelled"}:
            raise ValueError("invalid task status")
        task.updated_at = time.time()
        with self._lock, self._db() as db:
            db.execute("""INSERT INTO adaptive_tasks VALUES (?,?,?,?,?,?,?,?)
              ON CONFLICT(id) DO UPDATE SET status=excluded.status,
              strategy=excluded.strategy, checkpoint=excluded.checkpoint,
              evidence=excluded.evidence, updated_at=excluded.updated_at""",
              (task.id, task.persona_id, task.goal, task.status, task.strategy,
               json.dumps(task.checkpoint), json.dumps(task.evidence), task.updated_at))
        return task

    def get(self, persona_id: int, task_id: str) -> Task | None:
        with self._lock, self._db() as db:
            return self._task(db.execute(
                """SELECT id,persona_id,goal,status,strategy,checkpoint,evidence,updated_at
                   FROM adaptive_tasks WHERE persona_id=? AND id=?""",
                (persona_id, task_id)).fetchone())

    def list(self, persona_id: int) -> list[Task]:
        with self._lock, self._db() as db:
            rows = db.execute(
                """SELECT id,persona_id,goal,status,strategy,checkpoint,evidence,updated_at
                   FROM adaptive_tasks WHERE persona_id=? ORDER BY updated_at DESC""",
                (persona_id,)).fetchall()
        return [self._task(row) for row in rows]

    def transition(self, persona_id: int, task_id: str, status: str, *,
                   checkpoint: dict | None = None, proof: dict | None = None) -> Task:
        with self._lock:
            task = self.get(persona_id, task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status in {"done", "cancelled"}:
                raise ValueError("terminal task cannot be restarted")
            if status == "done" and (proof is None or not proof.get("verified")):
                raise ValueError("independent completion evidence required")
            if checkpoint is not None:
                task.checkpoint = checkpoint
            if proof is not None:
                task.evidence.append(proof)
            task.status = status
            return self.save(task)

    def begin_action(self, persona_id: int, task_id: str, resource: str) -> str:
        """Claim resource atomically across processes BEFORE issuing a tool call."""
        if not resource:
            raise ValueError("resource required")
        action_id = uuid.uuid4().hex
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                if db.execute("SELECT 1 FROM adaptive_tasks WHERE id=? AND persona_id=?",
                              (task_id, persona_id)).fetchone() is None:
                    raise KeyError(task_id)
                lease = db.execute(
                    "SELECT status FROM adaptive_resource_leases WHERE persona_id=? AND resource=?",
                    (persona_id, resource)).fetchone()
                if lease:
                    raise LeaseBusy(f"resource {resource!r} is {lease[0]}; reconcile before reuse")
                db.execute("""INSERT INTO adaptive_resource_leases VALUES (?,?,?,?,?,?)""",
                           (persona_id, resource, action_id, task_id, "active", time.time()))
                db.execute("INSERT INTO adaptive_actions VALUES (?,?,?,?,?,?)",
                           (action_id, persona_id, task_id, resource, "running", ""))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return action_id

    def finish_action(self, persona_id: int, action_id: str, receipt: str) -> None:
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT resource FROM adaptive_actions WHERE id=? AND persona_id=? AND status='running'",
                    (action_id, persona_id)).fetchone()
                if row is None:
                    raise ValueError("unknown or unreconciled action: inspect external state before retry")
                db.execute("UPDATE adaptive_actions SET status='finished', receipt=? WHERE id=?",
                           (receipt, action_id))
                db.execute("""DELETE FROM adaptive_resource_leases
                              WHERE persona_id=? AND resource=? AND action_id=? AND status='active'""",
                           (persona_id, row[0], action_id))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def mark_unknown(self, persona_id: int, action_id: str) -> None:
        """Tool failure/cancellation is ambiguous: never issue a blind retry."""
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT resource FROM adaptive_actions WHERE id=? AND persona_id=? AND status='running'",
                    (action_id, persona_id)).fetchone()
                if row is not None:
                    db.execute("UPDATE adaptive_actions SET status='unknown' WHERE id=?", (action_id,))
                    db.execute("""UPDATE adaptive_resource_leases SET status='unknown'
                                  WHERE persona_id=? AND resource=? AND action_id=?""",
                               (persona_id, row[0], action_id))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def unresolved(self, persona_id: int, task_id: str) -> list[dict]:
        with self._lock, self._db() as db:
            rows = db.execute("""SELECT id,resource,status FROM adaptive_actions
                                 WHERE persona_id=? AND task_id=? AND status IN ('running','unknown')""",
                              (persona_id, task_id)).fetchall()
        return [{"action_id": a, "resource": r, "status": s} for a, r, s in rows]

    def reconcile_action(self, persona_id: int, action_id: str, *, outcome: str,
                         source: str, basis: str) -> None:
        """Explicit reconciliation after checking external state.

        'not_executed' makes a retry permissible; 'completed' records external
        completion but still requires separate subtask verification.
        """
        if outcome not in {"not_executed", "completed"} or not source.strip() or not basis.strip():
            raise ValueError("reconciliation requires outcome and independent source/basis")
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    """SELECT task_id,resource,status FROM adaptive_actions
                       WHERE id=? AND persona_id=?""", (action_id, persona_id)).fetchone()
                if row is None or row[2] != "unknown":
                    raise ValueError("action is not unknown")
                remaining = db.execute(
                    """SELECT 1 FROM adaptive_actions WHERE persona_id=? AND resource=?
                       AND id<>? AND status IN ('running','unknown') LIMIT 1""",
                    (persona_id, row[1], action_id)).fetchone()
                db.execute("""UPDATE adaptive_actions SET status=?, receipt=? WHERE id=?""",
                           ("reconciled_" + outcome, json.dumps({"source": source, "basis": basis}), action_id))
                if remaining is None:
                    db.execute("""DELETE FROM adaptive_resource_leases
                                  WHERE persona_id=? AND resource=?""",
                               (persona_id, row[1]))
                else:
                    # Multiple legacy unknowns need individual reconciliations.
                    other = db.execute(
                        """SELECT id,task_id FROM adaptive_actions WHERE persona_id=? AND resource=?
                           AND status IN ('running','unknown') LIMIT 1""",
                        (persona_id, row[1])).fetchone()
                    db.execute("""UPDATE adaptive_resource_leases SET action_id=?,task_id=?,status='unknown'
                                  WHERE persona_id=? AND resource=?""",
                               (other[0], other[1], persona_id, row[1]))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def reconcile_subtask(self, persona_id: int, task_id: str, subtask_id: str, *,
                          outcome: str, source: str, basis: str) -> Task:
        """Manually reconcile a paused, uncertain subtask after external checks.

        Does not release unresolved action leases: callers must reconcile those
        actions first. An independent completed receipt is stored as evidence.
        """
        if outcome not in {"not_executed", "completed"} or not source.strip() or not basis.strip():
            raise ValueError("reconciliation needs checked outcome, source and basis")
        with self._lock:
            task = self.get(persona_id, task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status not in {"paused", "blocked"}:
                raise ValueError("pause task before external reconciliation")
            if self.unresolved(persona_id, task_id):
                raise ValueError("reconcile uncertain external actions first")
            target = next((row for row in task.checkpoint.get("items", [])
                           if row.get("id") == subtask_id), None)
            if target is None or target.get("state") not in {"running", "unknown"}:
                raise ValueError("subtask is not awaiting reconciliation")
            if outcome == "completed":
                target["state"] = "verified"
                task.evidence.append({"subtask": subtask_id, "verified": True,
                                      "source": source, "basis": basis,
                                      "reconciled": True})
            else:
                target["state"] = "pending"
                target["attempts"] = max(0, int(target.get("attempts", 0)) - 1)
            target.pop("last_error", None)
            return self.save(task)

    def record_outcome(self, persona_id: int, strategy: str, *,
                       success: bool, cost: float = 0) -> None:
        with self._lock, self._db() as db:
            db.execute("INSERT INTO adaptive_outcomes(persona_id,strategy,success,cost,recorded_at) VALUES(?,?,?,?,?)",
                       (persona_id, strategy, int(success), max(0.0, cost), time.time()))

    def rank(self, persona_id: int, strategies: list[str]) -> list[str]:
        """Smoothed success + bounded exploration bonus; never a weight update."""
        with self._lock, self._db() as db:
            rows = db.execute("""SELECT strategy, SUM(success), COUNT(*), SUM(cost)
                                 FROM adaptive_outcomes WHERE persona_id=? GROUP BY strategy""",
                              (persona_id,)).fetchall()
        history = {name: (wins, total, cost) for name, wins, total, cost in rows}
        total_trials = sum(r[1] for r in rows)
        def score(name: str) -> float:
            wins, attempts, cost = history.get(name, (0, 0, 0))
            laplace = (1 + wins) / (2 + attempts)
            explore = 0.12 * math.sqrt(math.log(2 + total_trials) / (1 + attempts))
            return laplace + explore - 0.001 * (cost or 0) / max(1, attempts)
        return sorted(strategies, key=lambda name: (-score(name), name))

    # Async façade: all disk I/O takes place outside the event loop. Keep
    # synchronous APIs for existing scripts and direct ledger unit tests.
    async def acreate(self, *args, **kwargs) -> Task:
        return await asyncio.to_thread(self.create, *args, **kwargs)

    async def asave(self, task: Task) -> Task:
        return await asyncio.to_thread(self.save, task)

    async def aget(self, persona_id: int, task_id: str) -> Task | None:
        return await asyncio.to_thread(self.get, persona_id, task_id)

    async def alist(self, persona_id: int) -> list[Task]:
        return await asyncio.to_thread(self.list, persona_id)

    async def atransition(self, *args, **kwargs) -> Task:
        return await asyncio.to_thread(self.transition, *args, **kwargs)

    async def abegin_action(self, *args) -> str:
        return await asyncio.to_thread(self.begin_action, *args)

    async def afinish_action(self, *args) -> None:
        await asyncio.to_thread(self.finish_action, *args)

    async def amark_unknown(self, *args) -> None:
        await asyncio.to_thread(self.mark_unknown, *args)

    async def aunresolved(self, *args) -> list[dict]:
        return await asyncio.to_thread(self.unresolved, *args)

    async def areconcile_action(self, *args, **kwargs) -> None:
        await asyncio.to_thread(self.reconcile_action, *args, **kwargs)

    async def areconcile_subtask(self, *args, **kwargs) -> Task:
        return await asyncio.to_thread(self.reconcile_subtask, *args, **kwargs)

    async def arecord_outcome(self, *args, **kwargs) -> None:
        await asyncio.to_thread(self.record_outcome, *args, **kwargs)

    async def arank(self, *args) -> list[str]:
        return await asyncio.to_thread(self.rank, *args)


@dataclass(frozen=True)
class Assignment:
    lobe: str
    role: str
    task_id: str = ""


class AdaptiveCoordinator:
    """One speaking voice; other lobe can execute when independent work exists."""

    def __init__(self):
        self._leases: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def allocate(self, *, user_engaged: bool, independent_work: int,
                 needs_verification: bool = False, current_speaker: str = "L") -> tuple[Assignment, Assignment]:
        other = "R" if current_speaker == "L" else "L"
        if user_engaged:
            return (Assignment(current_speaker, "conversation"),
                    Assignment(other, "conversation" if independent_work == 0 else "execute"))
        if independent_work >= 2:
            return Assignment("L", "execute"), Assignment("R", "execute")
        if independent_work == 1:
            return Assignment("L", "execute"), Assignment("R", "verify" if needs_verification else "plan")
        return Assignment("L", "verify" if needs_verification else "idle"), Assignment("R", "idle")

    @asynccontextmanager
    async def resource(self, key: str, owner: str):
        """Optional process-local fast fail; DB lease is authoritative."""
        if not key or not owner:
            raise ValueError("resource and owner required")
        async with self._lock:
            if key in self._leases:
                raise LeaseBusy(f"resource busy: {key}")
            self._leases[key] = owner
        try:
            yield
        finally:
            async with self._lock:
                if self._leases.get(key) == owner:
                    self._leases.pop(key, None)
