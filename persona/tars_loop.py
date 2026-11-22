"""Agent TARS is the execution engine; Persona is the mind.

Persona supplies inference (its R lobe), the planning tool, approvals, the fresh
observation and receipt verification. Agent TARS itself executes every browser,
filesystem, terminal, search and MCP action. The child has no credentials or
network endpoint; it talks to this lane over a private stdio channel.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Awaitable, Callable

from .link import SpeculativeEngine

ENGINE = Path(__file__).resolve().parent.parent / "engine"
SETUP = "Agent TARS engine unavailable: install Node.js 22+ and run npm run build:tars --prefix engine"
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


FIRE_MAX = 3


_FAIL_HEAD = ("error", "failed", "traceback", "enoent", "eacces", "eexist", "enotdir", "eisdir",
              "command failed", "access denied", "timeout")


_FAIL_ANY = ("does not exist", "no such file", "not found", "permission denied", "access denied",
             "outside allowed", "not in /", "is not a directory", "already exists")


def _text_failed(text: str) -> bool:
    low = str(text).lstrip().lower()
    return low.startswith(_FAIL_HEAD) or any(k in low[:240] for k in _FAIL_ANY)


def receipt_failed(result) -> bool:
    """True when an Agent TARS receipt means the tool failed. TARS tools return either
    plain JSON ({"status":"error"}) or MCP content parts ([{"type":"text","text":...}],
    where a failed command also carries a part named ERROR)."""
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except (TypeError, ValueError):
        return _text_failed(result)
    if isinstance(parsed, dict):
        return bool(parsed.get("error") or parsed.get("isError")
                    or str(parsed.get("status", "")).lower() in ("error", "failed", "failure"))
    if isinstance(parsed, list):
        for part in parsed:
            if not isinstance(part, dict):
                continue
            if str(part.get("name", "")).upper() == "ERROR" or part.get("isError"):
                return True
            if _text_failed(part.get("text", "")):
                return True
    return False


def outcome_of(result: str) -> str:
    """The discrete outcome a hand receipt maps to: ``ok`` or ``error``."""
    return "error" if receipt_failed(result) else "ok"


def _matches(expected, result: str) -> bool:
    return isinstance(expected, str) and bool(expected) and expected in (
        str(result), outcome_of(result), SpeculativeEngine().speculator.outcome_key(result))


class LanePlan:
    """Persona's own spec for one lane, gated by real evidence.

    R sends confidence, possible outcomes and an ordered batch in the same reply
    as the hand call. Once the real receipt matches a predicted outcome, the
    prepared step is ready at once: with ``autofire`` and enough confidence it is
    dispatched as the next call (no extra think), otherwise it returns as advice.
    """

    def __init__(self, tools: list[dict], autofire: bool = False, stats: dict | None = None):
        self.names = {t["function"]["name"] for t in tools if t["function"]["name"] != PLAN_NAME}
        self.engine = SpeculativeEngine()
        self.pending: dict | None = None
        self.advice: dict | None = None
        self.autofire = autofire
        self.stats = stats if stats is not None else {}
        self.fire_next: dict | None = None
        self.queue: list[dict] = []
        self.fired = 0
        self.firing = False

    def _count(self, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1

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
        self.queue = []
        self._count("prepared")
        return '{"prepared":true,"executed":false}'

    def ready(self) -> dict | None:
        """At think time: the evidence-matched next call to dispatch, if any."""
        action, self.fire_next = self.fire_next, None
        if action is None:
            return None
        self.fired += 1
        self.firing = True
        self._count("fired")
        return {"text": "", "finish_reason": "tool_calls", "tool_calls": [{
            "id": f"persona-spec-{self.stats['fired']}",
            "name": action["name"], "arguments": action["arguments"]}]}

    async def execute(self, action: Callable[[], Awaitable[str]]) -> str:
        plan, self.pending = self.pending or {}, None
        self.advice = None
        self.fire_next = None
        fired, self.firing = self.firing, False
        confidence = plan.get("confidence", 0.0)
        result = await self.engine.step(
            confidence=confidence, action_fn=action,
            precompute_fn=lambda _: plan.get("possibilities", {}),
            batch_actions=plan.get("batch", []))
        raw = result["result"]
        nxt, rest = result["next_action"], []
        if plan:
            batch = plan.get("batch") or []
            if batch and nxt is batch[0] or (
                    nxt is None and batch and _matches(batch[0].get("_expected_outcome"), raw)):
                nxt, rest = batch[0], batch[1:]
            elif nxt is None:
                nxt = (plan.get("possibilities") or {}).get(outcome_of(raw))
            self._count("matched" if nxt else "missed")
        elif self.queue:
            if _matches(self.queue[0].get("_expected_outcome"), raw):
                nxt, rest = self.queue[0], self.queue[1:]
                confidence = 1.0
                self._count("matched")
            else:
                self._count("missed")
            self.queue = []
        if plan:
            self.stats["last_confidence"] = confidence
        if nxt:
            if not (self.autofire and self.fired < FIRE_MAX
                    and confidence >= self.engine.confidence_threshold):
                self._count("held")
            if (self.autofire and self.fired < FIRE_MAX
                    and confidence >= self.engine.confidence_threshold):
                self.fire_next = {"name": nxt["name"], "arguments": nxt["arguments"]}
                self.queue = rest
            else:
                self.advice = nxt
        if fired:
            return "[Persona's prepared step fired on matching evidence; verify it] " + str(raw)
        return raw

    def with_advice(self, messages: list[dict]) -> list[dict]:
        advice, self.advice = self.advice, None
        if not advice:
            return messages
        return [*messages, {"role": "user", "content": (
            "Evidence-matching Persona plan (advice only, NOT executed). Verify against the "
            "actual tool receipts and current screenshot before choosing any next action: "
            + json.dumps(advice))}]


STARTUP_TIMEOUT = float(os.getenv("SAWII_TARS_STARTUP_TIMEOUT", "60"))
TARS_DIST = ENGINE / "ui-tars-desktop/multimodal/agent-tars/core/dist/index.js"


async def _allow_all(name: str, args: dict, mutating: bool) -> dict:
    return {"allow": True}


async def run_tars_loop(*, goal: str, instructions: str, tools: list[dict],
                        infer: Callable[[list[dict]], Awaitable[dict]],
                        execute: Callable[[str, dict], Awaitable[str]],
                        observe: Callable[[], Awaitable[dict | None]],
                        approve: Callable[[str, dict, bool], Awaitable[dict]] = _allow_all,
                        on_event: Callable[[dict], None] | None = None,
                        on_tools: Callable[[list[dict]], None] | None = None,
                        handle: dict | None = None,
                        on_receipt: Callable[[str, dict, str], Awaitable[None]] | None = None,
                        tars: dict | None = None, engine=None,
                        max_steps: int = 20, timeout: float = 120.0,
                        autofire: bool = False, stats: dict | None = None) -> str:
    """``tools`` are Persona-only tools (e.g. the planning tool); ``execute`` runs
    those. Every other tool is Agent TARS's own and runs inside the engine."""
    names = {tool["function"]["name"] for tool in tools}
    plan = LanePlan(tools, autofire=autofire, stats=stats)
    process = None
    if engine is None:
        node = os.getenv("SAWII_NODE") or shutil.which("node")
        if not node or not (ENGINE / "node_modules/tsx").exists() or not TARS_DIST.exists():
            raise RuntimeError(SETUP)
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
    else:
        # A persistent engine (the hub's): same TARS, same browser; this call is one loop on it.
        process = engine.process
        engine.looping = True
        while not engine.inbox.empty():
            engine.inbox.get_nowait()

    async def send(message: dict) -> None:
        process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
        await process.stdin.drain()

    async def readline() -> bytes:
        return await (engine.inbox.get() if engine is not None else process.stdout.readline())

    calls: dict[str, asyncio.Future] = {}

    async def call(name: str, args: dict | None = None, timeout_s: float = 30.0) -> str:
        """Ask Agent TARS to run one of its own tools for Persona (read-only use,
        e.g. a screenshot for the live card). TARS executes; Persona reads."""
        cid = f"py-{len(calls) + 1}"
        fut = asyncio.get_running_loop().create_future()
        calls[cid] = fut
        await send({"type": "call", "id": cid, "name": name, "args": args or {}})
        try:
            return await asyncio.wait_for(fut, timeout_s)
        finally:
            calls.pop(cid, None)

    if handle is not None:
        handle["call"] = engine.call if engine is not None else call

    try:
        if engine is None:
            ready = await asyncio.wait_for(process.stdout.readline(), timeout=15)
            if not ready or json.loads(ready).get("type") != "ready":
                raise RuntimeError(SETUP)
            await send({"type": "start", "goal": goal, "instructions": instructions,
                        "tools": tools, "max_steps": max_steps, "tars": tars or {}})
        else:
            plan.names |= set(engine.tools)
            await send({"type": "run", "goal": goal, "instructions": instructions, "tools": tools})
        booted = engine is not None  # Agent TARS boots its browser/filesystem/commands MCP servers first
        loop = asyncio.get_running_loop()
        failure: asyncio.Future = loop.create_future()
        serving: set[asyncio.Task] = set()

        async def serve(message: dict) -> None:
            """Answer one request from the engine. Requests run concurrently with the
            reader so Persona can itself ask the engine for something (a screenshot for
            the live card) while it is answering an observation."""
            kind = message.get("type")
            try:
                if kind == "think":
                    if on_tools is not None and message.get("tools"):
                        on_tools(message["tools"])
                    result = plan.ready()
                    if result is None:
                        plan.fired = 0
                        result = await asyncio.wait_for(infer(plan.with_advice(message["messages"])), timeout)
                    if result.get("finish_reason", "stop") not in ("stop", "tool_calls", "function_call"):
                        raise RuntimeError(f"Persona response was truncated or could not complete (finish_reason={result.get('finish_reason')!r}, text_tail={str(result.get('text') or '')[-80:]!r})")
                    if not str(result.get("text") or "").strip() and not result.get("tool_calls"):
                        raise RuntimeError("Persona returned no completed answer or tool calls")
                elif kind == "observe":
                    result = await asyncio.wait_for(observe(), timeout)
                elif kind == "approve":
                    result = await asyncio.wait_for(
                        approve(str(message.get("name")), message.get("args") or {},
                                bool(message.get("mutating"))), timeout)
                elif kind == "receipt":
                    # Agent TARS already executed this tool; Persona verifies/speculates on the receipt.
                    receipt = str(message.get("result"))

                    async def _done(receipt=receipt) -> str:
                        return receipt
                    if on_receipt is not None:
                        await asyncio.wait_for(
                            on_receipt(str(message.get("name")), message.get("args") or {}, receipt), timeout)
                    result = await asyncio.wait_for(plan.execute(_done), timeout)
                elif kind == "tool":
                    name, args = message.get("name"), message.get("args")
                    if name not in names or not isinstance(args, dict):
                        raise RuntimeError("Agent TARS requested an unadvertised Persona tool or invalid arguments")
                    if name == PLAN_NAME:
                        result = plan.prepare(args)
                    else:
                        result = await asyncio.wait_for(plan.execute(lambda: execute(name, args)), timeout)
                else:
                    raise RuntimeError("invalid Agent TARS protocol request")
                await send({"id": message["id"], "result": result})
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - surfaced by the reader below
                if not failure.done():
                    failure.set_exception(exc)

        try:
            while True:
                reader = asyncio.ensure_future(readline())
                done, _ = await asyncio.wait(
                    {reader, failure}, return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout if booted else max(timeout, STARTUP_TIMEOUT))
                if failure.done():
                    reader.cancel()
                    raise failure.exception()
                if reader not in done:
                    reader.cancel()
                    raise asyncio.TimeoutError()
                raw = reader.result()
                if not raw:
                    raise RuntimeError("Agent TARS engine disconnected before completing the lane")
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "done":
                    return str(message["text"])
                if kind == "error":
                    raise RuntimeError(str(message.get("error", "Agent TARS engine failed"))[:500])
                if kind == "catalog":
                    # Agent TARS's own tools: Persona's plan may name them.
                    booted = True
                    plan.names |= {str(n) for n in message.get("tools", [])}
                elif kind == "result":
                    fut = calls.get(str(message.get("id")))
                    if fut is not None and not fut.done():
                        fut.set_result(str(message.get("result")))
                elif kind == "event":
                    if on_event is not None:
                        try:
                            on_event(message)
                        except Exception:  # noqa: BLE001 - a card must never break a lane
                            pass
                else:
                    task = asyncio.create_task(serve(message))
                    serving.add(task)
                    task.add_done_callback(serving.discard)
        finally:
            for task in list(serving):
                task.cancel()
            await asyncio.gather(*serving, return_exceptions=True)
    finally:
        if engine is not None:
            if sys.exc_info()[0] is not None and engine.process is not None:
                try:  # cancelled or failed: stop the TARS loop so the browser is free again
                    await send({"type": "abort"})
                except Exception:  # noqa: BLE001
                    pass
            engine.looping = False  # the persistent engine stays up for the next call
        else:
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
