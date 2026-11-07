"""Cursor subagent lifecycle receipts into Persona's single shared mind.

Cursor's native Task tool owns execution. Hooks report real host child starts
and stops. This module NEVER spawns a second Persona or treats a model's
proposed call as confirmation that a child was created.
"""
from __future__ import annotations

import re

_MARK = re.compile(r"\[persona-lane (ln[0-9a-f]{8})\]")


async def receive_cursor_subagent(mind, pid: int, event: dict) -> dict:
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    phase = event.get("phase")
    if phase not in ("start", "stop"):
        raise ValueError("phase must be start or stop")
    task = event.get("task") or ""
    if not isinstance(task, str) or len(task) > 12000:
        raise ValueError("invalid task")
    match = _MARK.search(task)
    if not match:
        # Cursor can start unrelated built-in subagents. Do not assign them to
        # Persona or create an R line from an untrusted/unmarked task.
        return {"ok": True, "tracked": False, "reason": "not a Persona lane"}
    lane_id = match.group(1)
    lane = await mind.lanes.get(pid, lane_id)
    if lane is None:
        return {"ok": False, "tracked": False, "reason": "unknown lane", "lane": lane_id}

    def field(name: str, max_length: int = 256) -> str:
        value = event.get(name) or ""
        return str(value)[:max_length]

    status = field("status", 32)
    if phase == "stop" and status not in ("completed", "error", "aborted"):
        raise ValueError("invalid stop status")
    receipt = {
        "lane_id": lane_id, "phase": phase, "status": status if phase == "stop" else "launch_observed",
        "subagent_id": field("subagent_id"), "subagent_type": field("subagent_type", 80),
        "tool_call_id": field("tool_call_id"), "parent_conversation_id": field("parent_conversation_id"),
        "model": field("subagent_model", 120),
        "summary": field("summary", 4000) if phase == "stop" else "",
        "modified_files": [str(v)[:240] for v in (event.get("modified_files") or [])[:30]
                           if isinstance(v, str)] if phase == "stop" and isinstance(event.get("modified_files"), list) else [],
    }
    # Cursor's subagentStart fires BEFORE actual creation; it proves the host
    # entered the spawn hook, not that the new agent successfully started.
    # SharedTimeline records actual host child receipts. This is also
    # the authoritative, queryable subagent ID mapping after a restart.
    await mind.shared_timeline.append(
        pid, producer="cursor", kind="cursor_subagent_" + phase,
        task_id=lane_id, payload=receipt)
    if phase == "start":
        if lane.status == "active":
            lane.delegated = True
            lane.host_agent_id = receipt["subagent_id"]
            lane.host_tool_call_id = receipt["tool_call_id"]
            lane.host_status = "launch_observed"
            await mind.lanes.save(pid, lane)
    elif lane.status == "active":
        lane.host_status = status
        # Host completion is NOT independent verification.
        lane.result = ("Cursor " + status + " (unverified): " + receipt["summary"])[:4000]
        lane.status = "done" if status == "completed" else ("stopped" if status == "aborted" else "failed")
        await mind.lanes.save(pid, lane)
    return {"ok": True, "tracked": True, "lane": lane_id,
            "phase": phase, "host_status": receipt["status"],
            "verified": False}
