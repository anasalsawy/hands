"""Actual connected L/R/C online episode evaluator.

Runs the real Mind.respond(dual=True) kernel, not scripted role samples or
isolated model completions. C sees the complete generated answer. An external
harness/reviewer must independently confirm the goal before the outcome may
be admitted for weight training. No fake tool successes, auto-production
promotion, personal conversation storage or autonomous financial actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Awaitable, Callable
import json
import re

_EVIDENCE = re.compile(r"^(?:harness|review|test):[A-Za-z0-9_.:-]{1,100}$")


def digest(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


@dataclass
class JointOutcome:
    goal_hash: str
    response_hash: str = ""
    speaking_lobes: list[str] = field(default_factory=list)
    speech_hashes: list[dict] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    c_state: str = "UNRATED"
    c_level: str = ""
    proof: str = ""
    independently_verified: bool = False
    success: bool = False
    latency_ms: float | None = None
    user_interruptions: int = 0
    trainable: bool = False

    def record(self) -> dict:
        return {**self.__dict__, "objective": "verified_team_goal",
                "roles": ["L", "R", "C"], "synthetic": False}


ExternalVerify = Callable[[str, str, list[dict]], Awaitable[dict]]


async def run_connected_episode(mind: Any, *, persona_id: int, goal: str,
                                verify_external: ExternalVerify,
                                user_permission: bool = False) -> JointOutcome:
    """One real, consented goal episode, evidence gated, never automatically SGD.

    A model's own statements and C's GREEN review are *not* external task proof.
    The verifier receives a complete candidate once all lobes have finished;
    events are kept as hashes and categorical summaries, never raw tool data.
    """
    if not user_permission:
        raise PermissionError("Explicit authorization required for real episode capture")
    if not goal.strip() or len(goal) > 2500:
        raise ValueError("Need a bounded authorized goal")
    if verify_external is None:
        raise ValueError("Independent harness required")
    from time import monotonic
    from .brains import verifier as verification

    started = monotonic()
    result = JointOutcome(goal_hash=digest(goal))
    chunks: list[str] = []
    evidence_events: list[dict] = []
    current: str | None = None
    final_verdict = None
    async for event in mind.respond(
            persona_id, goal, channel="training", dual=True, surface="connected pilot",
            home=True):
        kind = str(event.get("type") or "")
        if kind == "beat":
            speaker = event.get("writing")
            if speaker in ("L", "R"):
                current = speaker
                result.speaking_lobes.append(speaker)
        elif kind == "delta":
            text = str(event.get("text") or "")
            chunks.append(text)
            if text.strip():
                result.speech_hashes.append({"lobe": current or "?",
                                             "sha256": digest(text)})
        elif kind in ("tool_start", "tool_done"):
            # Do not retain tool contents; execution evidence has to be
            # collected independently by the trusted harness.
            entry = {"kind": kind, "tool": str(event.get("name") or "")[:80],
                     "ok": event.get("ok") is True}
            result.tool_events.append(entry)
            evidence_events.append(entry)
        elif kind in ("hold", "silent", "verify", "steer", "interrupt"):
            if kind == "interrupt":
                result.user_interruptions += 1
            if kind in ("steer", "verify"):
                result.decisions.append({"point": kind,
                                         "status": str(event.get("state") or "")[:36]})
        elif kind == "done":
            final_verdict = event.get("verdict")
        elif kind == "error":
            result.decisions.append({"point": "runtime_error", "status": "failed"})
    candidate = "".join(chunks)
    result.response_hash = digest(candidate)
    # Do not infer valid external evidence from the kernel's own finish flag.
    c = getattr(mind, "c", None)
    if c is None:
        return result
    review = await verification.verify(
        c, context=goal, response_text=candidate, latest_request=goal,
        events=json.dumps(evidence_events)[:6000],
        on_health=getattr(mind, "health_registry", {}).get("verifier"))
    result.c_state = str(getattr(review, "state", "UNRATED"))
    result.c_level = str(getattr(review, "level", ""))
    result.latency_ms = round((monotonic() - started) * 1000, 2)
    if not (result.c_state == "READY" and result.c_level == "GREEN"):
        return result
    # The external callback must read the real system's final state, not
    # take the assistant's explanation as evidence of task completion.
    actual = await verify_external(goal, candidate, result.tool_events)
    proof = actual.get("evidence_ref")
    result.proof = proof if isinstance(proof, str) and _EVIDENCE.fullmatch(proof) else ""
    result.independently_verified = bool(actual.get("independently_verified") is True
                                          and result.proof)
    result.success = bool(result.independently_verified and actual.get("success") is True)
    # A completed team task is useful as review data; do NOT enable neural
    # SFT automatically. It first needs private-data review and role credit.
    result.trainable = False
    return result


def team_reward(outcome: JointOutcome, *, tool_budget: int = 12,
                latency_budget_ms: int = 20000) -> float | None:
    """Goal-level bounded reward, never inferred from self-reported success."""
    if not outcome.independently_verified or not outcome.proof:
        return None
    if outcome.c_state != "READY" or outcome.c_level != "GREEN":
        return 0.0
    success = 1.0 if outcome.success else 0.0
    latency = min(1.0, max(0.0, (outcome.latency_ms or 0) / latency_budget_ms))
    tool_overuse = min(1.0, max(0.0, len(outcome.tool_events) / max(tool_budget, 1)))
    interrupt_penalty = min(1.0, outcome.user_interruptions / 10)
    return round(max(0, min(1.0, success - 0.08 * latency
                            - 0.05 * tool_overuse - 0.05 * interrupt_penalty)), 4)
