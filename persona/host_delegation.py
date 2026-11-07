"""Host tool schema validation, action deduplication and output boundary.

This does not execute or spoof host tools. Both lobes can reason over dialogue,
observations and action options; only the host may execute its own tools.
"""
from __future__ import annotations

import json

def is_host_background_tool(tool: dict) -> bool:
    if not isinstance(tool, dict):
        return False
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = fn.get("name")
    schema = fn.get("parameters")
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(props, dict):
        return False
    if name == "delegate_task":  # Hermes: always async by default.
        tasks = props.get("tasks")
        return isinstance(tasks, dict) and tasks.get("type") == "array"
    if name in ("Task", "task", "spawn_subagent", "launch_subagent"):
        # Cursor's built-in Task tool may advertise only a prompt and an agent
        # type: background execution is configured on the selected subagent,
        # not necessarily on the tool parameters. Recognize the host capability
        # without fabricating an unsupported run_in_background argument.
        return ("run_in_background" in props or "is_background" in props
                or (name in ("Task", "task") and
                    ("prompt" in props or "subagent_type" in props)))
    return False


def route_host_output(text: str, calls: list[dict]) -> tuple[str, list[dict]]:
    """Route structured tool calls to the host, text to conversation output.

    Never rewrite names, arguments, or ids, and never invent a tool result.
    This router is an output boundary; it does not pretend the host is dual.
    """
    unique: dict[tuple[str, str], dict] = {}
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        args = str(call.get("arguments") or "{}")
        if name and (name, args) not in unique:
            unique[(name, args)] = call
    return (text or ""), list(unique.values())


def _checked_action(action: dict, tools: list[dict]) -> dict | None:
    """Check the model's requested action against the host's actual schemas."""
    if not isinstance(action, dict):
        return None
    name = str(action.get("name") or "")
    args = action.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    match = next((t for t in tools if (t.get("function") or {}).get("name") == name), None)
    if not match:
        return None
    schema = match["function"].get("parameters") or {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return None
    required = schema.get("required") or []
    if any(k not in args for k in required):
        return None
    if schema.get("additionalProperties") is False and any(k not in props for k in args):
        return None
    if name == "delegate_task":
        # Hermes supports an authoritative live-child control plane. The
        # actions list/steer/stop do not accept a tasks array; unlike spawn,
        # they MUST be allowed through if host's actual schema advertises them.
        control_action = args.get("action", "spawn")
        actions = (props.get("action") or {}).get("enum") or []
        if control_action in ("list", "steer", "stop"):
            if control_action not in actions:
                return None
            if control_action in ("steer", "stop") and not isinstance(args.get("subagent_id"), str):
                return None
            if control_action == "steer" and not str(args.get("message") or "").strip():
                return None
        elif control_action in ("spawn", None):
            tasks = args.get("tasks")
            if not isinstance(tasks, list) or not tasks or len(tasks) > 4:
                return None
            if not all(isinstance(t, dict) and isinstance(t.get("goal"), str)
                       and t["goal"].strip() for t in tasks):
                return None
        else:
            return None
    if name in ("Task", "task", "spawn_subagent", "launch_subagent"):
        if is_host_background_tool(match):
            flags = [k for k in ("run_in_background", "is_background") if k in props]
            if flags and not any(args.get(k) is True for k in flags):
                return None
    return {"id": str(action.get("id") or "call_" + __import__("uuid").uuid4().hex[:16]),
            "name": name, "arguments": json.dumps(args, ensure_ascii=False)}


def _completed_actions(messages: list[dict]) -> set[tuple[str, str]]:
    """Suppress redispatch of a host tool call whose result already arrived."""
    completed = {str(m.get("tool_call_id") or "") for m in messages
                 if m.get("role") == "tool"}
    prior = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            if c.get("id") not in completed:
                continue
            fn = c.get("function") or {}
            try:
                args = json.dumps(json.loads(fn.get("arguments") or "{}"), sort_keys=True)
            except ValueError:
                continue
            prior.add((str(fn.get("name") or ""), args))
    return prior


