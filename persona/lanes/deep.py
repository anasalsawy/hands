"""The workshop (deep lane): heavy multi-step work runs here, in the background,
server-side, beside the wheel - never inside a turn.

    A delegates  ->  plan (B)  ->  N parallel lanes  ->  B critiques each as it lands
                 ->  streaming reduce  ->  result returns to A (next turn / outreach)

Lane patterns (pick by task shape, Amdahl decides the speedup, not head count):
    single       one worker
    map          independent subtasks in parallel, merged (breadth work: 3-5x)
    speculative  the same task attempted in parallel, the critic keeps the best
                 (buys wall-clock with tokens on hard single steps)

A worker is an OpenHands agent (terminal + file editor in the persona's own
server-side workspace) when ``openhands-sdk`` is installed; otherwise a plain
model worker. The user's machine is never touched.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os

from ..brains.prompts import DEEP_CRITIC, DEEP_PLAN, DEEP_REDUCE
from ..llm import LLM, extract_json, strip_reasoning
from ..store import jdump, now

log = logging.getLogger(__name__)


def workspace_for(root: str, key_or_id: str) -> str:
    d = os.path.join(root, hashlib.sha256(str(key_or_id).encode()).hexdigest()[:16])
    os.makedirs(d, exist_ok=True)
    return d


class LLMWorker:
    name = "model"

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, task: str, workspace: str) -> str:
        res = await self.llm.chat([{"role": "system", "content": "You are a diligent worker. Do the task fully and "
                                    "report the result with specifics. If you cannot verify something, say so."},
                                   {"role": "user", "content": task}], temperature=0.4, max_tokens=2500)
        return strip_reasoning(res.text)


class OpenHandsWorker:
    """Runs an OpenHands SDK agent in the persona's workspace (thread, sync SDK)."""

    name = "openhands"

    def __init__(self, model: str, base_url: str, api_key: str, max_iterations: int = 40):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.max_iterations = max_iterations

    @staticmethod
    def available() -> bool:
        try:
            import openhands.sdk  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    async def run(self, task: str, workspace: str) -> str:
        return await asyncio.to_thread(self._run_sync, task, workspace)

    def _run_sync(self, task: str, workspace: str) -> str:  # pragma: no cover - needs the SDK
        from pydantic import SecretStr
        from openhands.sdk import LLM as OHLLM, Agent, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.terminal import TerminalTool

        llm = OHLLM(model=f"openai/{self.model}", base_url=self.base_url, api_key=SecretStr(self.api_key))
        agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)])
        conv = Conversation(agent=agent, workspace=workspace, max_iteration_per_run=self.max_iterations)
        conv.send_message(task + "\n\nWhen finished, end with a short plain-text summary of the result.")
        conv.run()
        final = ""
        for ev in reversed(list(conv.state.events)):
            src = getattr(ev, "source", "")
            msg = getattr(ev, "llm_message", None)
            if src == "agent" and msg is not None:
                parts = getattr(msg, "content", []) or []
                final = " ".join(getattr(p, "text", "") for p in parts if getattr(p, "text", ""))
                if final:
                    break
        return final or "Finished; see the workspace for the output files."


class DeepLane:
    def __init__(self, mind, *, planner: LLM, critic: LLM, worker_factory=None, max_lanes: int = 4):
        self.mind = mind
        self.planner = planner
        self.critic = critic
        self.max_lanes = max_lanes
        self.worker_factory = worker_factory or (lambda: LLMWorker(mind.b))
        # No fixed worker count: 0 starts every planned worker at once.
        self._sem = asyncio.Semaphore(max_lanes) if max_lanes > 0 else contextlib.nullcontext()

    async def submit(self, pid: int, task: str, mode: str = "auto") -> int:
        job_id = await self.mind.store.insert("deep_jobs", persona_id=pid, task=task[:8000], mode=mode,
                                              created_at=now())
        self.mind.spawn(self._run(pid, job_id, task, mode), name=f"deep-{job_id}")
        return job_id

    async def _plan(self, task: str, mode: str) -> tuple[str, list[str]]:
        if mode in ("single", "speculative"):
            return mode, [task]
        try:
            res = await self.planner.chat([{"role": "system", "content": DEEP_PLAN}, {"role": "user", "content": task}],
                                          temperature=0.2, max_tokens=600, json_mode=True)
            data = extract_json(res.text) or {}
            m = data.get("mode") if data.get("mode") in ("single", "map", "speculative") else "single"
            subs = [str(s) for s in (data.get("subtasks") or []) if str(s).strip()][: self.max_lanes or None]
            if m == "map" and len(subs) >= 2:
                return "map", subs
            return m, [task]
        except Exception:  # noqa: BLE001
            return "single", [task]

    async def _score(self, task: str, result: str) -> float:
        try:
            res = await self.critic.chat([{"role": "system", "content": DEEP_CRITIC},
                                          {"role": "user", "content": f"Task: {task}\n\nResult:\n{result[:6000]}"}],
                                         temperature=0.0, max_tokens=200, json_mode=True)
            return float((extract_json(res.text) or {}).get("score", 0.5))
        except Exception:  # noqa: BLE001
            return 0.5

    async def _lane(self, task: str, workspace: str, trace: list) -> tuple[str, float]:
        async with self._sem:
            worker = self.worker_factory()
            started = now()
            try:
                out = await worker.run(task, workspace)
            except Exception as exc:  # noqa: BLE001
                out = f"(worker failed: {exc})"
            score = await self._score(task, out)  # pipelined verification: judged as it lands
            trace.append({"task": task[:200], "worker": worker.name, "seconds": round(now() - started, 1),
                          "score": score})
            return out, score

    async def _run(self, pid: int, job_id: int, task: str, mode: str) -> None:
        await self.mind.store.update("deep_jobs", job_id, status="running")
        workspace = workspace_for(self.mind.s.workspace_root, f"persona-{pid}")
        trace: list = []
        try:
            plan_mode, tasks = await self._plan(task, mode)
            if plan_mode == "speculative":
                tasks = [task] * (min(3, self.max_lanes) if self.max_lanes > 0 else 3)
            results = await asyncio.gather(*(self._lane(t, workspace, trace) for t in tasks))
            if plan_mode == "speculative" or len(results) == 1:
                best = max(results, key=lambda r: r[1])[0]
            else:
                joined = "\n\n".join(f"### {t}\n{r[0]}" for t, r in zip(tasks, results))
                res = await self.planner.chat([{"role": "system", "content": DEEP_REDUCE},
                                               {"role": "user", "content": f"Task: {task}\n\n{joined[:20000]}"}],
                                              temperature=0.2, max_tokens=2500)
                best = strip_reasoning(res.text)
            await self.mind.store.update("deep_jobs", job_id, status="done", result=best[:20000],
                                         trace=jdump({"mode": plan_mode, "lanes": trace}), finished_at=now())
            await self.mind.store.insert("thoughts", persona_id=pid, text=f"(job {job_id}) {task[:160]} -> {best[:400]}",
                                         source="deep", score=0.8, ref=str(job_id), created_at=now())
            self.mind.bus.publish(pid, {"type": "deep_done", "job": job_id})
        except Exception as exc:  # noqa: BLE001
            log.exception("deep job failed")
            await self.mind.store.update("deep_jobs", job_id, status="failed", result=str(exc)[:1000],
                                         trace=jdump({"lanes": trace}), finished_at=now())

    async def unseen_results(self, pid: int) -> list[dict]:
        rows = await self.mind.store.all("SELECT id, task, result FROM deep_jobs WHERE persona_id = ? AND "
                                         "status = 'done' AND seen = 0 ORDER BY id", pid)
        for r in rows:
            await self.mind.store.update("deep_jobs", r["id"], seen=1)
        return rows


def attach(mind, settings) -> DeepLane:
    if OpenHandsWorker.available() and settings.a.configured:
        def factory():
            return OpenHandsWorker(settings.b.model, settings.b.base_url, settings.b.api_key)
    else:
        factory = None
    lane = DeepLane(mind, planner=mind.b, critic=mind.b, worker_factory=factory, max_lanes=settings.deep_max_lanes)
    mind.deep = lane
    return lane
