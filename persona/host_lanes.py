"""Explicit Persona lane contract and asynchronous request scope."""
from contextvars import ContextVar

# The one surface whose host turns run through the cheap alias table. Set by
# the Cursor gateway adapter in api/app.py, mirrored by the tests.
CURSOR_SURFACE = "persona-cursor-deepinfra"

CURRENT_SCOPE = ContextVar("sawii_request_scope", default=None)


def parse_lane(header, body):
    values = []
    for value in (header, body):
        if value is None:
            continue
        if not isinstance(value, str) or value not in ("L", "R"):
            raise ValueError("sawii_lane must be L or R")
        values.append(value)
    if len(set(values)) > 1:
        raise ValueError("conflicting lane selectors")
    return values[0] if values else None


def lane_lock_key(pid, session_id, lane, purpose="conversation"):
    return (pid, session_id, lane, purpose) if lane else pid


def scope_key(pid):
    scope = CURRENT_SCOPE.get()
    return scope if isinstance(scope, tuple) and scope[0] == pid else pid


def use_host_tool_kernel(client_tools, lane=None):
    """Whether an inference host advertised executable native tools."""
    return bool(client_tools)


def additional_host_task(host_messages, client_tools, active_lines=()):
    """Identify an independently arriving user request while host Task is busy.

    The host remains the orchestrator. Only model-issued native Task calls are
    returned. No synthetic completion or Cursor-private API is involved.
    An unacknowledged host Task or an active delegated Persona R is evidence
    of ongoing work. Tool-result continuations are not new user jobs.
    """
    if not any((t.get("function") or {}).get("name") in ("Task", "task")
               for t in (client_tools or []) if isinstance(t, dict)):
        return False
    rows = [m for m in (host_messages or []) if isinstance(m, dict)]
    if not rows or rows[-1].get("role") != "user":
        return False
    # A new user request must follow an earlier tool-spawn action.
    if any(getattr(line, "status", None) == "active" and
           getattr(line, "delegated", False) for line in active_lines):
        return True
    active = set()
    for row in rows[:-1]:
        if row.get("role") == "assistant":
            for call in row.get("tool_calls") or []:
                if (call.get("function") or {}).get("name") in ("Task", "task"):
                    active.add(str(call.get("id") or ""))
        elif row.get("role") == "tool":
            active.discard(str(row.get("tool_call_id") or ""))
    return bool(active)



def choose_host_inference(normal, cursor_worker, *, surface, home):
    """An explicit Cursor alias changes model provider, not the Persona mind.

    No other host, Tasker, or Companion changes providers because Cursor did.
    """
    if not home and surface == CURSOR_SURFACE and cursor_worker is not None:
        return cursor_worker
    return normal


def cursor_alias_model_id(value, *, surface, cursor_worker):
    """Per-request model string -> an alias-table model id for the cursor face.

    Mirrors choose_host_inference's gate exactly: any other surface, or no
    configured Cursor worker, returns "" - the request must never resolve
    through the alias table off the persona-cursor-deepinfra face, and the
    adapter keeps rejecting every model string other than "sawii-persona"
    there. On the cursor face the only strings that resolve are exact keys of
    config.CURSOR_MODEL_ALIASES; the empty "" answer means "not a valid alias"
    and the caller must reject the request, so a host- or attacker-chosen
    model id is never passed through to the provider.
    """
    if cursor_worker is None or surface != CURSOR_SURFACE:
        return ""
    from .config import cursor_alias_for
    return cursor_alias_for(value)
