"""Omni-TARS' real composable loop, with all inference and tools owned by Persona.

The child has no credentials or network endpoint. Its injected LLM client and
tool plugin use a private stdio channel into this R lane, not a second mind.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import shutil
from typing import Awaitable, Callable

from .link import SpeculativeEngine

ENGINE = Path(__file__).resolve().parent.parent / "engine"
SETUP = "Omni-TARS loop unavailable: install Node.js 22+ and run npm ci --prefix engine"
PLAN_NAME = "persona_prepare_next_actions"
ACTION_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "arguments": {"type": "object"},
    "_expected_outcome": {"type": "string"},
}, "required": ["name", "arguments"]}
PLAN_TOOL = {"type": "function", "function": {
    "name": PLAN_NAME,
    "description": (
        "Read-only Persona planning: prepare possible next actions BEFORE the next hand call. "
        "May precede that call in the same ordered batch. confidence is an estimate (0..1); "
        "possibilities maps exact result/status to a proposed next action (up to 4 outcomes). "
        "Optional batch actions (up to 3) "
        "need _expected_outcome. Nothing is executed speculatively: matching advice returns "
        "with the real receipts and fresh vision for R to verify."),
    "parameters": {"type": "object", "properties": {
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "possibilities": {"type": "object", "additionalProperties": ACTION_SCHEMA},
        "batch": {"type": "array", "items": ACTION_SCHEMA},
    }, "required": ["confidence", "possibilities"]},
}}


class LanePlan:
    """One-use, evidence-gated speculation. It never dispatches a predicted tool."""

    def __init__(self, tools: list[dict]):
        self.names = {t["function"]["name"] for t in tools if t["function"]["name"] != PLAN_NAME}
        self.engine = SpeculativeEngine()
        self.pending: dict | None = None
        self.advice: dict | None = None

    def prepare(self, arguments: dict) -> str:
        confidence = arguments.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("plan confidence must be a finite probability between 0 and 1")
        possibilities = arguments.get("possibilities", {})
        batch = arguments.get("batch", [])
        if not isinstance(possibilities, dict) or not isinstance(batch, list):
            raise ValueError("plan must contain an outcome map and an action batch")
        if len(possibilities) > self.engine.max_possibilities or len(batch) > self.engine.max_batch:
            raise ValueError("plan exceeds the lane's speculation budget")
        for action in [*possibilities.values(), *batch]:
            if (not isinstance(action, dict) or action.get("name") not in self.names
                    or not isinstance(action.get("arguments"), dict)):
                raise ValueError("plan contains an unadvertised action or invalid arguments")
        for action in batch:
            if not isinstance(action.get("_expected_outcome"), str) or not action["_expected_outcome"]:
                raise ValueError("each batched suggestion needs an expected outcome")
        self.pending = {"confidence": confidence, "possibilities": possibilities, "batch": batch}
        return '{"prepared":true,"executed":false}'

    async def execute(self, action: Callable[[], Awaitable[str]]) -> str:
        plan, self.pending = self.pending or {}, None
        self.advice = None
        result = await self.engine.step(
            confidence=plan.get("confidence", 0.0), action_fn=action,
            precompute_fn=lambda _: plan.get("possibilities", {}),
            batch_actions=plan.get("batch", []))
        self.advice = result["next_action"]
        return result["result"]

    def with_advice(self, messages: list[dict]) -> list[dict]:
        advice, self.advice = self.advice, None
        if not advice:
            return messages
        return [*messages, {"role": "user", "content": (
            "Evidence-matching Persona plan (advice only, NOT executed). Verify against the "
            "actual tool receipts and current screenshot before choosing any next action: "
            + json.dumps(advice))}]


async def run_tars_loop(*, goal: str, instructions: str, tools: list[dict],
                        infer: Callable[[list[dict]], Awaitable[dict]],
                        execute: Callable[[str, dict], Awaitable[str]],
                        observe: Callable[[], Awaitable[dict | None]],
                        max_steps: int = 20, timeout: float = 120.0) -> str:
    node = os.getenv("SAWII_NODE") or shutil.which("node")
    if not node or not (ENGINE / "node_modules/tsx").exists():
        raise RuntimeError(SETUP)
    names = {tool["function"]["name"] for tool in tools}
    plan = LanePlan(tools)
    # Do not pass Persona/provider keys, NODE_OPTIONS or loader injection to a motor.
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "PATH", "HOME", "SYSTEMROOT", "WINDIR", "USERPROFILE", "APPDATA",
        "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR",
    }}
    try:
        process = await asyncio.create_subprocess_exec(
            node, "--import", "tsx", str(ENGINE / "persona-loop.ts"),
            cwd=ENGINE, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            limit=16 * 1024 * 1024)
    except OSError as exc:
        raise RuntimeError(SETUP) from exc

    async def send(message: dict) -> None:
        process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
        await process.stdin.drain()

    try:
        ready = await asyncio.wait_for(process.stdout.readline(), timeout=15)
        if not ready or json.loads(ready).get("type") != "ready":
            raise RuntimeError(SETUP)
        await send({"type": "start", "goal": goal, "instructions": instructions,
                    "tools": tools, "max_steps": max_steps})
        while True:
            raw = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
            if not raw:
                raise RuntimeError("Omni-TARS loop disconnected before completing the lane")
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "done":
                return str(message["text"])
            if kind == "error":
                raise RuntimeError(str(message.get("error", "Omni-TARS loop failed"))[:500])
            if kind == "think":
                result = await asyncio.wait_for(infer(plan.with_advice(message["messages"])), timeout)
                if result.get("finish_reason", "stop") not in ("stop", "tool_calls", "function_call"):
                    raise RuntimeError("Persona response was truncated or could not complete")
                if not str(result.get("text") or "").strip() and not result.get("tool_calls"):
                    raise RuntimeError("Persona returned no completed answer or tool calls")
            elif kind == "observe":
                result = await asyncio.wait_for(observe(), timeout)
            elif kind == "tool":
                name, args = message.get("name"), message.get("args")
                if name not in names or not isinstance(args, dict):
                    raise RuntimeError("Omni-TARS requested an unadvertised tool or invalid arguments")
                # Even if Tarko batches calls, mutations on one surface stay ordered.
                if name == PLAN_NAME:
                    result = plan.prepare(args)
                else:
                    result = await asyncio.wait_for(plan.execute(lambda: execute(name, args)), timeout)
            else:
                raise RuntimeError("invalid Omni-TARS loop protocol request")
            await send({"id": message["id"], "result": result})
    finally:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
