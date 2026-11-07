"""Parallel minds: the moment a question arrives, consultant models start
thinking about it at the same time as A. A *pulls* their work with the
``fetch_consults`` tool - with a bounded wait, never blocking its own sentence.

Whatever lands late is not lost: B reviews it after the turn and may turn it
into an afterthought ("oh, one more thing...").
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from ..llm import LLM, strip_reasoning
from .prompts import CONSULT

log = logging.getLogger(__name__)


@dataclass
class Consult:
    name: str
    task: asyncio.Task
    started: float = field(default_factory=time.monotonic)
    result: str | None = None
    error: str | None = None


class ConsultBoard:
    def __init__(self, consultants: dict[str, LLM], *, timeout: float = 45.0):
        self.consultants = consultants
        self.timeout = timeout
        self.consults: list[Consult] = []

    def start(self, question: str, context: str = "") -> None:
        for name, llm in self.consultants.items():
            self.consults.append(Consult(name=name, task=asyncio.create_task(self._run(llm, question, context))))

    async def _run(self, llm: LLM, question: str, context: str) -> str:
        msgs = [{"role": "system", "content": CONSULT}]
        if context:
            msgs.append({"role": "system", "content": "Context (background, may be partial):\n" + context[:4000]})
        msgs.append({"role": "user", "content": question})
        res = await asyncio.wait_for(llm.chat(msgs, temperature=0.4, max_tokens=600), self.timeout)
        return strip_reasoning(res.text)

    def _collect(self) -> None:
        for c in self.consults:
            if c.result is None and c.error is None and c.task.done():
                try:
                    c.result = c.task.result()
                except Exception as exc:  # noqa: BLE001
                    c.error = str(exc)[:200]

    async def fetch(self, wait_ms: int = 300) -> dict:
        pending = [c.task for c in self.consults if not c.task.done()]
        if pending and wait_ms > 0:
            await asyncio.wait(pending, timeout=wait_ms / 1000)
        self._collect()
        ready = [{"from": c.name, "said": c.result} for c in self.consults
                 if c.result and c.result.strip().upper() != "NOTHING"]
        working = [c.name for c in self.consults if not c.task.done()]
        failed = [c.name for c in self.consults if c.error]
        return {"ready": ready, "still_working": working, "failed": failed}

    async def settle(self, timeout: float = 20.0) -> list[dict]:
        """After the turn: give stragglers a little longer, then report everything."""
        pending = [c.task for c in self.consults if not c.task.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        for c in self.consults:
            if not c.task.done():
                c.task.cancel()
        self._collect()
        return [{"from": c.name, "said": c.result} for c in self.consults
                if c.result and c.result.strip().upper() != "NOTHING"]

    def render(self, data: dict) -> str:
        lines = []
        for item in data.get("ready", []):
            lines.append(f"[{item['from']}] {item['said'][:1500]}")
        if data.get("still_working"):
            lines.append("still working: " + ", ".join(data["still_working"]))
        if not lines:
            lines.append("No consult has anything yet.")
        return "\n".join(lines)
