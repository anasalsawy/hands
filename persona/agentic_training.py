"""Evidence-gated browser/desktop and L/R coordination training for Persona.

The live tools are in computer.py, desktop.py and gateway/browser/tools.py.
Don't train on guessed interactions, invented screen contents, or synthetic
completion claims. An episode's identity is its own goal, not a Quranic mode.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

BROWSER_ACTIONS = frozenset({
    "browser_navigate", "browser_click", "browser_type", "browser_scroll",
    "browser_back", "browser_wait_for_selector",
})
BROWSER_OBSERVE = frozenset({"browser_snapshot"})
COMPUTER_ACTIONS = frozenset({"computer_open_url", "computer_click", "computer_type_text"})
COMPUTER_OBSERVE = frozenset({"computer_screen_text", "computer_screenshot"})
DESKTOP_ACTIONS = frozenset({
    "desktop_open_app", "desktop_focus_app", "desktop_click", "desktop_click_xy",
    "desktop_type_text", "desktop_key", "desktop_scroll",
})
DESKTOP_OBSERVE = frozenset({"desktop_screen_text", "desktop_screenshot"})
READS = BROWSER_OBSERVE | COMPUTER_OBSERVE | DESKTOP_OBSERVE
ACTIONS = BROWSER_ACTIONS | COMPUTER_ACTIONS | DESKTOP_ACTIONS
COORDINATION_TASK = "lobe_coordination"
ACTION_TASKS = frozenset({"browser_control", "desktop_control", COORDINATION_TASK})


def _reject(message: str) -> None:
    from .tajweed_training import TrainingDataError
    raise TrainingDataError(message)


def _tool_messages(trajectory: list[dict]) -> list[tuple[str, dict, str, str]]:
    """Ordered observed hand operations; error texts are retained, never hidden."""
    calls: dict[str, tuple[int, str, dict]] = {}
    observations: dict[str, tuple[int, str]] = {}
    for pos, item in enumerate(trajectory):
        if item.get("role") == "assistant":
            for call in item.get("tool_calls", []):
                if not isinstance(call, dict):
                    _reject("Malformed assistant tool call")
                cid = call.get("id")
                fn = call.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    _reject("Each tool needs an actual function name")
                name = fn["name"]
                if name not in ACTIONS | READS:
                    _reject(f"Unknown screen tool: {name}")
                import json
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        _reject("Malformed tool JSON arguments")
                if not isinstance(args, dict):
                    _reject("Tool arguments must be an object")
                calls[cid] = (pos, name, args)
        elif item.get("role") == "tool":
            observations[item.get("tool_call_id")] = (pos, str(item.get("content", "")))
    if set(calls) != set(observations):
        _reject("Action/observation IDs are incomplete")
    result = []
    for cid, (pos, name, args) in sorted(calls.items(), key=lambda kv: kv[1][0]):
        obs_pos, value = observations[cid]
        if obs_pos <= pos:
            _reject("Tool result cannot precede its requested action")
        if not value.strip():
            _reject("An empty result cannot verify a screen action")
        result.append((name, args, value, cid))
    return result


def _is_failure(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in (
        '"error"', "failed:", "attempt failed", "unknown browser tool", "permission denied",
    ))


def validate_execution_evidence(sample: dict) -> dict:
    """Check Persona-specific tool discipline and L/R collaboration evidence.

    This validates an externally reviewed record; it does not personally
    attest the screenshot is true. Training still requires evaluation in the
    actual browser/desktop harness and a reproducible source evidence reference.
    """
    if sample.get("stage") != "specialist" or sample.get("mode") != "general":
        _reject("Browser/desktop/coordination data is normal-agent specialization")
    if sample.get("role") not in ("L", "R"):
        _reject("Only speaking/working L/R may learn browser/desktop coordination")
    if sample.get("task") not in ACTION_TASKS:
        _reject("Unsupported agentic execution task")
    if sample.get("review_status") != "accepted":
        _reject("Tool episode must pass independent acceptance criteria")
    if sample.get("evidence_kind") not in ("harness_test", "reviewed_screenshot", "reviewed_trace"):
        _reject("Provide reproducible tool execution evidence")
    if not isinstance(sample.get("evidence_ref"), str) or not sample["evidence_ref"].strip():
        _reject("A nonempty external evidence_ref is required")
    if not isinstance(sample.get("acceptance_criteria"), list) or not sample["acceptance_criteria"]:
        _reject("Specify external task acceptance criteria")
    if any(not isinstance(item, str) or not item.strip()
           for item in sample["acceptance_criteria"]):
        _reject("Acceptance criteria must be concrete strings")
    trace = sample.get("trajectory")
    if not isinstance(trace, list) or not trace:
        _reject("A tool execution trajectory is required")
    operations = _tool_messages(trace)
    actions = [n for n, _, _, _ in operations if n in ACTIONS]
    if not actions:
        _reject("A successful task must contain a real computer/browser action")
    if sample["task"] == "browser_control" and not any(
        n in BROWSER_ACTIONS | COMPUTER_ACTIONS for n in actions
    ):
        _reject("Browser lesson lacks browser navigation/action")
    if sample["task"] == "desktop_control" and not any(n in DESKTOP_ACTIONS for n in actions):
        _reject("Desktop lesson lacks desktop action")
    if any(_is_failure(val) for _, _, val, _ in operations) and not sample.get("recovery_verified"):
        _reject("Recorded tool failures require an independently verified recovery")
    # After a click or navigation, old indices are stale; the next click/type
    # must first have a fresh screen read in the same tool family.
    fresh: dict[str, bool] = {"browser": False, "computer": False, "desktop": False}
    last_index = None
    for name, args, outcome, _ in operations:
        family = name.split("_", 1)[0]
        if name in READS:
            fresh[family] = not _is_failure(outcome)
            continue
        if name.endswith(("_click", "_type", "_type_text")) or name == "desktop_click_xy":
            if not fresh[family]:
                _reject(f"{name} requires a fresh {family} screen observation")
            if name in ("browser_click", "computer_click", "desktop_click", "desktop_type_text",
                        "computer_type_text", "browser_type"):
                index = args.get("index")
                if type(index) is not int or index < 0:
                    _reject("An indexed action needs a nonnegative live element index")
                if name.endswith("click") and not str(args.get("text", "")).strip():
                    _reject("Indexed clicks need visible text for re-grounding")
                last_index = index
        fresh[family] = False
    # The last action cannot be called a success merely from its own receipt.
    last_action_idx = max(i for i, (name, _, _, _) in enumerate(operations) if name in ACTIONS)
    if not any(name in READS for name, _, _, _ in operations[last_action_idx + 1:]):
        _reject("Read the screen again after the last action to verify the result")
    coordination = sample.get("coordination")
    if sample["task"] == COORDINATION_TASK:
        if not isinstance(coordination, list) or len(coordination) < 2:
            _reject("Coordination needs actual cross-lobe handoff observations")
        writers = {x.get("writing") for x in coordination if isinstance(x, dict)}
        if writers != {"L", "R"}:
            _reject("Coordination must include both L and R")
        if not any(x.get("kind") == "hold" and x.get("to") in ("L", "R")
                   for x in coordination if isinstance(x, dict)):
            _reject("Coordination must include a real handoff/hold")
        if not any(x.get("kind") == "resolved" and x.get("evidence_ref")
                   for x in coordination if isinstance(x, dict)):
            _reject("A hold needs a verified resolution")
    return {"operations": len(operations), "actions": len(actions),
            "observations": len(operations) - len(actions),
            "coordination": bool(coordination)}


def coordination_report(samples: list[dict]) -> dict[str, Any]:
    """Measure verified screen use/coordination, never model accuracy from prompts."""
    counts: Counter = Counter()
    for sample in samples:
        if sample.get("task") in ACTION_TASKS:
            evidence = validate_execution_evidence(sample)
            counts[(sample["role"], sample["task"])] += 1
            counts[(sample["role"], "actions")] += evidence["actions"]
    return {
        role: {"browser": counts[role, "browser_control"],
               "desktop": counts[role, "desktop_control"],
               "coordination": counts[role, COORDINATION_TASK],
               "actions": counts[role, "actions"]}
        for role in ("L", "R")
    }
