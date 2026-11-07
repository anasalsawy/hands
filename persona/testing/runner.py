"""The test harness (READY-PLAN Step 6): real turns, real numbers.

Rewritten 2026-10-22. The earlier version faked the reply ("Mock turn
execution"), always passed an "anchor score" of 0.95 and ran on whatever
persona id it was given. This one runs the real ``Mind.respond`` - kernel,
memory, verifier, psyche - and returns what actually happened: the reply, the
server-side metrics (ttft_ms, dead_air_max_ms, overlap_ms, ...), every tool
call with its time, the kernel events, and her L/R weights before and after
(when learning is on).
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

KERNEL = {"beat", "hold", "silent", "verify", "steer", "interrupt"}


@dataclass
class TestTurnResult:
    persona_id: int
    user_message: str
    reply: str = ""
    dual: bool | None = None
    metrics: dict = field(default_factory=dict)
    tools: list[dict] = field(default_factory=list)
    kernel: list[dict] = field(default_factory=list)
    weights_before: dict | None = None
    weights_after: dict | None = None
    wall_ms: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _weights(mind, pid: int) -> dict | None:
    """L/R state for the report: per-turn count and the L-R gap. None = learning off."""
    learner = getattr(mind, "learner", None)
    if learner is None:
        return None
    from ..life import weights as W
    L, R, meta = learner.weights.load(pid, learner.backend)
    return {"turns": int(meta.get("turns", 0)), "gap": round(W.norm(W.gap(L, R)), 5),
            "backend": learner.backend.name}


async def run_test_turn(mind, *, persona_id: int, user_message: str, dual: bool | None = None,
                        home: bool = True, include_kernel: bool = True) -> TestTurnResult:
    """One real turn through the mind, instrumented."""
    res = TestTurnResult(persona_id=persona_id, user_message=user_message, dual=dual)
    res.weights_before = _weights(mind, persona_id)
    starts: dict[str, float] = {}
    t0 = time.monotonic()
    try:
        async for ev in mind.respond(persona_id, user_message, dual=dual, home=home, channel="test"):
            t = ev.get("type")
            if t == "tool_start":
                starts[ev.get("id", "")] = time.monotonic()
            elif t == "tool_done":
                s = starts.pop(ev.get("id", ""), None)
                res.tools.append({"name": ev.get("name"), "ok": ev.get("ok", True), "ms": ev.get("ms",
                                  round((time.monotonic() - s) * 1000) if s else 0)})
            elif t in KERNEL and include_kernel:
                res.kernel.append(ev)
            elif t == "error":
                res.error = str(ev.get("detail", ""))[:300]
            elif t == "done":
                res.reply = ev.get("text", "")
                res.metrics = dict(ev.get("metrics") or {})
    except Exception as exc:  # noqa: BLE001 - a harness reports, it does not crash the server
        res.error = f"{type(exc).__name__}: {exc}"[:300]
    res.wall_ms = round((time.monotonic() - t0) * 1000)
    if hasattr(mind, "drain"):
        await mind.drain(timeout=30)     # post-turn work (memory, learning) has landed
    res.weights_after = _weights(mind, persona_id)
    return res


def _stats(values: list[float]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {}
    return {"avg": round(statistics.mean(v)), "p50": round(statistics.median(v)),
            "min": round(min(v)), "max": round(max(v))}


async def run_benchmark(mind, *, persona_id: int, iterations: int = 5, dual: bool | None = None,
                        messages: list[str] | None = None) -> dict:
    """``iterations`` real turns; the numbers that prove (or disprove) the kernel."""
    prompts = messages or ["hey, how is your day going?", "what time is it for you?",
                           "remind me what we talked about last", "tell me something you noticed about me",
                           "give me a quick plan for tomorrow"]
    runs = []
    for i in range(iterations):
        r = await run_test_turn(mind, persona_id=persona_id, user_message=prompts[i % len(prompts)],
                                dual=dual, include_kernel=False)
        runs.append(r)
    ms = [r.metrics for r in runs]
    return {
        "persona_id": persona_id,
        "dual": dual,
        "iterations": iterations,
        "errors": [r.error for r in runs if r.error],
        "ttft_ms": _stats([m.get("ttft_ms") for m in ms]),
        "dead_air_max_ms": _stats([m.get("dead_air_max_ms") for m in ms]),
        "reply_ms": _stats([m.get("reply_ms") for m in ms]),
        "overlap_ms": _stats([m.get("overlap_ms") for m in ms]),
        "wall_ms": _stats([r.wall_ms for r in runs]),
        "replies": [r.reply[:160] for r in runs],
    }
