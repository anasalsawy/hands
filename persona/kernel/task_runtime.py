"""Evidence-gated task runner: L stays conversational, R1..Rn execute elastically.

All R lanes for a persona share the same durable ledger, conflict coordinator,
and persona-scoped chronological conversation/tool history.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .adaptive import AdaptiveCoordinator, Task, TaskLedger
from .execution_pool import ExecutionPool
from .conflict_coordinator import ResourceConflictCoordinator, ActionIntent, SupersededAction
from ..life.update import strategy_feedback
from ..decision_map import DecisionJournal


@dataclass(frozen=True)
class WorkItem:
    id: str
    instruction: str
    resource: str = ""
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WorkResult:
    receipt: str
    outcome: dict = field(default_factory=dict)


RunItem = Callable[[str, WorkItem], Awaitable[WorkResult]]
CheckItem = Callable[[WorkItem, WorkResult], Awaitable[dict]]


class AdaptiveTaskRunner:
    """Coordinate independent steps, verify each outcome, checkpoint every batch.

    A task cannot become done based on an executor's self-report. The separate
    verification callback must provide evidence with a basis and source.
    Unknown external action status after restart is never retried automatically.
    """

    def __init__(self, ledger: TaskLedger, coordinator: AdaptiveCoordinator | None = None,
                 *, max_attempts: int = 3, max_batches: int = 24,
                 journal: DecisionJournal | None = None):
        self.ledger = ledger
        self.coordinator = coordinator or AdaptiveCoordinator()
        self.max_attempts = max_attempts
        self.max_batches = max_batches
        self._engaged: set[int] = set()
        self._engagement_refs: dict[int, int] = {}
        self._speaker: dict[int, str] = {}
        self._running: dict[str, asyncio.Task] = {}
        self.journal = journal
        # Elastic R allocation is unconditional. No fixed worker count or
        # artificial waiting merely because earlier R lanes are occupied.
        # SAWII_BACKGROUND_LANES is obsolete and intentionally ignored.
        self.max_execution_lanes = None
        self._pools: dict[int, ExecutionPool] = {}
        self._conflicts: dict[int, ResourceConflictCoordinator] = {}
        self._decision_tasks: set[asyncio.Task] = set()

    def _pool(self, pid: int) -> ExecutionPool:
        return self._pools.setdefault(pid, ExecutionPool())

    def available_execution_slots(self, pid: int) -> int:
        return self._pool(pid).available

    def conflict_coordinator(self, pid: int) -> ResourceConflictCoordinator:
        return self._conflicts.setdefault(pid, ResourceConflictCoordinator())

    def _record(self, pid: int, point: str, selected: str, *,
                task_id: str = "", context: dict | None = None) -> None:
        """Record observed choices asynchronously without slowing the worker."""
        if self.journal is None:
            return

        async def save() -> None:
            try:
                await self.journal.arecord(persona_id=pid, task_id=task_id,
                                           point=point, selected=selected,
                                           context=context)
            except Exception:
                pass  # Telemetry must never prevent a real task from running.

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(save())
        self._decision_tasks.add(task)
        task.add_done_callback(self._decision_tasks.discard)

    async def drain_decisions(self, timeout: float = 5.0) -> None:
        """Let in-flight telemetry writes finish before the store is closed.

        `_record` is deliberately fire-and-forget: a slow disk must never slow a
        real task. But that leaves writers racing teardown, and on Windows an
        open sqlite handle makes the file undeletable - a pending telemetry
        write then surfaces as a PermissionError in unrelated cleanup. Linux
        tolerates it; Windows does not, and this app runs on Windows.
        """
        pending = [t for t in self._decision_tasks if not t.done()]
        if not pending:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True),
                                   timeout=timeout)
        except (asyncio.TimeoutError, RuntimeError):
            for task in pending:
                task.cancel()

    def set_user_engaged(self, persona_id: int, engaged: bool) -> None:
        """Keep a speaking slot until ALL active voice/chat leases have ended."""
        was_engaged = persona_id in self._engaged
        refs = max(0, self._engagement_refs.get(persona_id, 0) + (1 if engaged else -1))
        if refs:
            self._engagement_refs[persona_id] = refs
            self._engaged.add(persona_id)
        else:
            self._engagement_refs.pop(persona_id, None)
            self._engaged.discard(persona_id)
            self._speaker.pop(persona_id, None)
        currently_engaged = persona_id in self._engaged
        if was_engaged != currently_engaged:
            self._record(persona_id, "engagement",
                         "conversation_reserved" if currently_engaged else "execution_focused",
                         context={"user_engaged": currently_engaged, "mode": "home"})

    def set_conversation_speaker(self, persona_id: int, speaker: str) -> None:
        """Use whichever lobe currently has the conversational floor.

        Running actions remain in flight. The next batch picks the available
        lobe, never replays an already-issued tool call.
        """
        if speaker not in ("L", "R") or persona_id not in self._engaged:
            return
        previous = self._speaker.get(persona_id, "L")
        if previous != speaker:
            self._speaker[persona_id] = speaker
            self._record(persona_id, "lobe_allocation",
                         "conversation_and_execution",
                         context={"user_engaged": True, "role_switched": True,
                                  "mode": "home"})

    def status(self, persona_id: int, task_id: str) -> Task | None:
        return self.ledger.get(persona_id, task_id)

    def submit(self, persona_id: int, goal: str, items: list[WorkItem], *,
               execute: RunItem, verify: CheckItem, strategy: str = "adaptive") -> Task:
        if not items:
            raise ValueError("task needs executable subtasks")
        ids = [w.id for w in items]
        if len(ids) != len(set(ids)) or not all(ids):
            raise ValueError("subtask ids must be unique and nonempty")
        task = self.ledger.create(persona_id, goal, strategy=strategy)
        task.checkpoint = {"items": [{"id": w.id, "instruction": w.instruction,
                                     "resource": w.resource, "state": "pending",
                                     "attempts": 0} for w in items],
                           "batches": 0}
        self.ledger.save(task)
        self.resume(persona_id, task.id, items, execute=execute, verify=verify)
        return task

    def resume(self, persona_id: int, task_id: str, items: list[WorkItem], *,
               execute: RunItem, verify: CheckItem) -> None:
        task = self.ledger.get(persona_id, task_id)
        if task is None:
            raise KeyError(task_id)
        if task.status not in ("queued", "paused", "blocked"):
            raise ValueError("task is not resumable")
        saved = {row["id"]: row for row in task.checkpoint.get("items", [])}
        if set(saved) != {w.id for w in items}:
            raise ValueError("resumed items differ from checkpoint")
        # Any in-flight action from before a crash must be checked externally.
        # The unknown action is not silently re-executed.
        if any(row["state"] in ("running", "unknown") for row in saved.values()) or self.ledger.unresolved(persona_id, task_id):
            raise ValueError("in-flight subtask requires reconciliation")
        if task_id in self._running and not self._running[task_id].done():
            raise ValueError("task already has a worker")
        job = asyncio.create_task(self._work(persona_id, task_id, items, execute, verify),
                                  name=f"adaptive-task-{task_id[:8]}")
        self._running[task_id] = job
        job.add_done_callback(lambda _: self._running.pop(task_id, None))

    async def asubmit(self, persona_id: int, goal: str, items: list[WorkItem], *,
                      execute: RunItem, verify: CheckItem,
                      strategy: str = "adaptive") -> Task:
        if not items or not all(w.id for w in items):
            raise ValueError("task needs executable subtasks")
        if len({w.id for w in items}) != len(items):
            raise ValueError("duplicate subtask ids")
        task = await self.ledger.acreate(persona_id, goal, strategy=strategy)
        task.checkpoint = {"items": [{"id": w.id, "instruction": w.instruction,
                                     "resource": w.resource, "state": "pending",
                                     "attempts": 0} for w in items], "batches": 0}
        await self.ledger.asave(task)
        await self.aresume(persona_id, task.id, items, execute=execute, verify=verify)
        return task

    async def aresume(self, persona_id: int, task_id: str, items: list[WorkItem], *,
                      execute: RunItem, verify: CheckItem) -> None:
        task, pending = await asyncio.gather(
            self.ledger.aget(persona_id, task_id),
            self.ledger.aunresolved(persona_id, task_id))
        if task is None:
            raise KeyError(task_id)
        if task.status not in ("queued", "paused", "blocked"):
            raise ValueError("task is not resumable")
        saved = {row["id"]: row for row in task.checkpoint.get("items", [])}
        if set(saved) != {w.id for w in items}:
            raise ValueError("resumed items differ from checkpoint")
        if pending or any(row["state"] in ("running", "unknown") for row in saved.values()):
            raise ValueError("in-flight subtask requires reconciliation")
        if task_id in self._running and not self._running[task_id].done():
            raise ValueError("task already has a worker")
        job = asyncio.create_task(self._work(persona_id, task_id, items, execute, verify),
                                  name=f"adaptive-task-{task_id[:8]}")
        self._running[task_id] = job
        job.add_done_callback(lambda _: self._running.pop(task_id, None))

    def pause(self, persona_id: int, task_id: str) -> Task:
        return self.ledger.transition(persona_id, task_id, "paused")

    def cancel(self, persona_id: int, task_id: str) -> Task:
        return self.ledger.transition(persona_id, task_id, "cancelled")

    async def apause(self, persona_id: int, task_id: str) -> Task:
        return await self.ledger.atransition(persona_id, task_id, "paused")

    async def acancel(self, persona_id: int, task_id: str) -> Task:
        return await self.ledger.atransition(persona_id, task_id, "cancelled")

    async def wait(self, task_id: str) -> None:
        task = self._running.get(task_id)
        if task is not None:
            await task

    async def _work(self, pid: int, task_id: str, items: list[WorkItem],
                    execute: RunItem, verify: CheckItem) -> None:
        task = await self.ledger.atransition(pid, task_id, "running")
        known = {w.id: w for w in items}
        try:
            for _ in range(self.max_batches):
                planned_version = await self.conflict_coordinator(pid).current_version(task_id)
                task = await self.ledger.aget(pid, task_id)
                if task is None or task.status != "running":
                    return
                pending = [row for row in task.checkpoint["items"] if row["state"] not in ("verified", "unknown")
                           and row["attempts"] < self.max_attempts]
                if any(row["state"] == "unknown" for row in task.checkpoint["items"]):
                    await strategy_feedback(self.ledger, pid, task.strategy, verified=True, success=False)
                    await self.ledger.atransition(pid, task_id, "paused")
                    return
                if not pending:
                    if all(row["state"] == "verified" for row in task.checkpoint["items"]):
                        await strategy_feedback(self.ledger, pid, task.strategy, verified=True, success=True)
                        await self.ledger.atransition(pid, task_id, "done",
                                               proof={"verified": True, "source": "subtask verifier",
                                                      "basis": "Every planned subtask independently verified"})
                    else:
                        await self.ledger.arecord_outcome(pid, task.strategy, success=False)
                        await self.ledger.atransition(pid, task_id, "paused", checkpoint=task.checkpoint)
                    return
                # L is never consumed as a tool lane. Every independent item
                # can acquire a free R or immediately create a new one.
                capacity = len(pending)
                execution_lobes = ["R"] * capacity
                engaged = pid in self._engaged
                allocation = ("conversation_and_execution" if engaged else
                              "multi_execution" if capacity > 2 else
                              "dual_execution" if capacity == 2 else
                              "execute_and_plan" if capacity == 1 else "idle")
                self._record(pid, "lobe_allocation", allocation, task_id=task_id,
                             context={"user_engaged": engaged,
                                      "independent_items": len(pending),
                                      "has_independent_work": len(pending) > 1,
                                      "mode": "task",
                                      "batch": task.checkpoint["batches"]})
                # Resource classes are not scheduling locks. Independent work
                # packages start together; actual tool actions arbitrate conflicts.
                selected = pending[:capacity]
                for row in selected:
                    row["state"] = "running"
                    row["attempts"] += 1
                task.checkpoint["batches"] += 1
                await self.ledger.asave(task)

                # Capture the instruction version when the batch is planned,
                # not after a busy R slot becomes available.

                async def run_owned(lobe: str, row: dict) -> tuple[str, dict | None, str]:
                    item = known[row["id"]]
                    action = None
                    try:
                        resource = item.resource or f"work:{task_id}:{item.id}"
                        conflict = self.conflict_coordinator(pid)
                        version = planned_version
                        # Do not hold a device-class lock during model inference.
                        # Individual tool calls are guarded in GoalAgent._execute_for.
                        if version != await conflict.current_version(task_id):
                            raise SupersededAction("instruction changed before execution")
                        action = await self.ledger.abegin_action(pid, task_id, resource)
                        if version != await conflict.current_version(task_id):
                            raise SupersededAction("instruction changed during action persistence")
                        result = await execute(lobe, item)
                        await self.ledger.afinish_action(pid, action, result.receipt)
                        proof = await verify(item, result)
                        # A separate checker must give a concrete evidence source.
                        if (isinstance(proof, dict) and proof.get("verified") is True
                                and str(proof.get("source") or "").strip()
                                and str(proof.get("basis") or "").strip()):
                            return item.id, proof, ""
                        tools = (result.outcome or {}).get("tool_results")
                        if isinstance(tools, list) and not any(r.get("effect", True) for r in tools):
                            # Nothing outside changed, so another attempt is safe.
                            return item.id, None, "retry: not verified, no side effects"
                        return item.id, None, "not independently verified"
                    except SupersededAction:
                        if action is not None:
                            await self.ledger.afinish_action(
                                pid, action, "not executed: superseded before tool entry")
                        # No tool was admitted. Re-plan using the new instruction
                        # rather than falsely marking an external action unknown.
                        return item.id, None, "superseded_before_execution"
                    except BaseException as exc:  # noqa: BLE001
                        if action is not None:
                            await self.ledger.amark_unknown(pid, action)
                        return item.id, None, f"{type(exc).__name__}: {exc}"[:200]

                async def one(lobe: str, row: dict) -> tuple[str, dict | None, str]:
                    async with self._pool(pid).reserve() as actual_lobe:
                        return await run_owned(actual_lobe, row)

                results = await asyncio.gather(
                    *(one(execution_lobes[i], row) for i, row in enumerate(selected)))
                # Reload: a user may have paused/cancelled while hands were active.
                fresh = await self.ledger.aget(pid, task_id)
                if fresh is None:
                    return
                states = {row["id"]: row for row in fresh.checkpoint["items"]}
                for item_id, proof, error in results:
                    self._record(pid, "task_verification",
                                 "verified" if proof else "unverified",
                                 task_id=task_id,
                                 context={"has_verifier": True,
                                          "has_tool_evidence": bool(proof), "mode": "task",
                                          "reason": "verified" if proof else "not_verified"})
                    row = states[item_id]
                    # A tool may have changed external state even when verification
                    # failed. Never repeat it automatically; pause for review.
                    if error.startswith("retry:"):
                        row["state"] = "pending"
                        row["last_error"] = error
                        continue
                    if error == "superseded_before_execution":
                        row["state"] = "pending"
                        row["attempts"] = max(0,row["attempts"]-1)
                        continue
                    row["state"] = "verified" if proof else "unknown"
                    if proof:
                        fresh.evidence.append({"subtask": item_id, **proof})
                    if error:
                        row["last_error"] = error
                await self.ledger.asave(fresh)
                if fresh.status != "running":
                    return
            current = await self.ledger.aget(pid, task_id)
            if (current is not None and current.status == "running"
                    and all(row["state"] == "verified" for row in current.checkpoint["items"])):
                # The final permitted batch verified the last item.
                await strategy_feedback(self.ledger, pid, current.strategy, verified=True, success=True)
                await self.ledger.atransition(pid, task_id, "done",
                                       proof={"verified": True, "source": "subtask verifier",
                                              "basis": "Every planned subtask independently verified"})
                return
            if current is not None and current.status == "running":
                # Exhausted a scheduling/cost budget; not a successful task.
                await strategy_feedback(self.ledger, pid, current.strategy, verified=True, success=False)
                await self.ledger.atransition(pid, task_id, "paused", checkpoint=current.checkpoint)
        except asyncio.CancelledError:
            current = await self.ledger.aget(pid, task_id)
            if current is not None and current.status == "running":
                await self.ledger.atransition(pid, task_id, "paused", checkpoint=current.checkpoint)
            raise
        except Exception:
            current = await self.ledger.aget(pid, task_id)
            if current is not None and current.status == "running":
                await self.ledger.atransition(pid, task_id, "paused", checkpoint=current.checkpoint)
            raise

