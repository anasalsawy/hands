"""Two parts at once: each lobe drives one independent part of a task.

Owner, 2026-10-30: "i want them to be able to work concurrently on two taskss
... or on two parts of same task". The person hears one voice; behind it, L
and R each own one part - its tool calls, its results, its next step - and
the parts meet again in one answer. See README.md

A round is split only when its calls need different resources. Look-ups that
share nothing already run side by side in ``_hand_groups``; the gain here is
for a screen (one browser per persona) working beside a fetch, a file read or
another machine - work that today waits in one queue.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING

from ..llm import ChatResult, ToolCall

if TYPE_CHECKING:  # pragma: no cover
    from .dual import DualLoop
    from .writings import Writing

# Never pressed by a part: these wait for the voice, which can ask first.
IRREVERSIBLE = re.compile(
    r"(?:pay|purchase|buy|book|reserve|checkout|submit|send|transfer|delete|remove|cancel|"
    r"publish|deploy|wire|charge|refund|post|write|edit|move|shell|make_plugin|order|sign.?up|"
    r"log.?out|unsubscribe)", re.I)
# The names whose arguments can carry an irreversible intent ("click Pay now").
_PRESSES = re.compile(r"(?:click|type|press|key|submit|select)", re.I)
_IRREVERSIBLE_WORDS = re.compile(
    r"\b(?:pay|buy|book|purchase|checkout|check out|place order|order now|submit|send|transfer|"
    r"delete|remove|cancel|publish|post|confirm|sign up|log out|unsubscribe)\b", re.I)
# Screen tools of the server's own computer drive the persona's one browser,
# so they share its lane unless the host says otherwise.
_SCREEN_BASES = frozenset({
    "open_url", "screen_text", "screenshot", "click", "click_xy", "type_text",
    "press_keys", "key", "scroll", "open_app", "focus_app", "wait",
})
_MARKER = re.compile(r"\u27e6[^\u27e7]*\u27e7")

PARTS_HINT = """

[two parts at once]
When what they ask has parts that do not depend on each other - a page to work
on and a fact to fetch, two different things to look up - call the tools for
each part together in the same step. Each part then runs at the same time and
you answer once, when both are back. A step that needs another step's result
waits for it."""

PART_PROMPT = (
    "[not shown to them - your part of the task]\n"
    "You are doing one part of what they asked while the other side of your mind "
    "does another part at the same time.\n"
    "Your part started with: {mine}\n"
    "The other part, not yours - do not do it: {theirs}\n"
    "If your part needs another step, call its tool now. When your part is done, "
    "write its result in one or two plain sentences, only from what the tools "
    "returned; if it failed, say what failed. No greeting, nothing about the other part.")

PARTS_NOTE = ("Both parts of the task came back; they ran at the same time.\n{lines}\n"
              "Give them one answer with each result once. Do not describe the parts "
              "or which side of you did what.")


def lane_of(name: str, tools=None) -> str:
    """The exclusive resource a call needs, or "" for a look-up that needs none."""
    custom = getattr(tools, "lane", None)
    if callable(custom):
        try:
            got = custom(name)
            if got is not None:
                return str(got)
        except Exception:  # noqa: BLE001 - a host lane map that breaks falls back
            pass
    text = name or ""
    if text.startswith("browser_"):
        return "browser"
    if text.startswith("desktop_"):
        return "desktop"
    if text.startswith("host_"):
        return "host"
    for place, lane in (("computer_", "computer"), ("cloud_", "cloud")):
        if text.startswith(place):
            base = text[len(place):]
            if base in _SCREEN_BASES:
                return "browser" if place == "computer_" else lane
            return "" if _lookup(text) else lane
    return ""


def _lookup(name: str) -> bool:
    from .dual import _parallel_safe
    return _parallel_safe(name)


def can_run_in_part(tc, tools=None) -> bool:
    """A call a part may make on its own: reversible, and either a look-up or a lane."""
    from .dual import _base_name
    name = tc.name or ""
    if IRREVERSIBLE.search(_base_name(name)):
        return False
    if _PRESSES.search(name) and _IRREVERSIBLE_WORDS.search(tc.arguments or ""):
        return False
    return bool(lane_of(name, tools)) or _lookup(name)


def split_parts(tool_calls: list, tools=None) -> list[list] | None:
    """Two parts when the calls need different resources, else None.

    Calls on the same lane stay together and in order (one browser, one
    pointer). At least one lane must be involved: look-ups alone are already
    run side by side by ``_hand_groups``.
    """
    calls = list(tool_calls or [])
    if len(calls) < 2 or not all(can_run_in_part(tc, tools) for tc in calls):
        return None
    units: dict[str, list] = {}
    for i, tc in enumerate(calls):
        lane = lane_of(tc.name, tools)
        units.setdefault(lane or f"look:{i}", []).append(tc)
    if len(units) < 2 or all(k.startswith("look:") for k in units):
        return None
    parts: list[list] = [[], []]
    for i, group in enumerate(units.values()):
        parts[i % 2].extend(group)
    return parts


def label(calls: list) -> str:
    """A short human label for a part: its first call and the argument that names it."""
    if not calls:
        return ""
    tc = calls[0]
    args = tc.args()
    hint = next((str(v) for v in args.values() if isinstance(v, str) and v.strip()), "")
    text = tc.name + (f" {hint}" if hint else "")
    if len(calls) > 1:
        text += f" (+{len(calls) - 1})"
    return text[:90]


def _same(tc) -> tuple[str, str]:
    """A call's identity for "the other part already does this"."""
    return tc.name, json.dumps(tc.args(), sort_keys=True)


def _wanted(calls: list) -> str:
    return "; ".join(f"{c.name}({c.arguments})" for c in calls)[:400]


def _allowed(schemas: list[dict], lanes: set[str], tools) -> list[dict]:
    out = []
    for s in schemas:
        name = s.get("function", {}).get("name", "")
        tc = ToolCall(id="", name=name, arguments="")
        if not can_run_in_part(tc, tools):
            continue
        lane = lane_of(name, tools)
        if (lane and lane in lanes) or (not lane and _lookup(name)):
            out.append(s)
    return out


async def run_parts(loop: "DualLoop", msgs: list[dict], res: ChatResult, parts: list[list],
                    writings: tuple["Writing", "Writing"], *, rounds: int) -> tuple[list[dict], list[dict]]:
    """Run the parts at once; same contract as ``DualLoop._hands``.

    ``msgs`` gets the assistant message that asked for every call, then one
    tool message per call. A part's later steps are folded into the tool
    message of its last opening call, so every call keeps exactly one answer
    and the shared prefix stays a valid tool-call/result sequence. The first
    returned calls line up with the tool messages; a part's later steps follow.
    """
    base = list(msgs[:-1]) if msgs and msgs[-1].get("role") == "user" else list(msgs)
    msgs.append({"role": "assistant", "content": res.text or None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in res.tool_calls]})
    tools = loop.caps.tools
    schemas = loop.caps.schemas_for("L")
    t0 = time.perf_counter()
    board = [{"writing": w.key, "label": label(p), "state": "running", "steps": 0}
             for w, p in zip(writings, parts)]
    loop._parts_live, loop._part_steers = True, []

    def show(phase: str) -> None:
        loop._event({"type": "parts", "phase": phase, "tracks": [dict(b) for b in board]})

    show("start")

    done = [{"writing": w.key, "opening": [], "later": [], "finding": "", "rounds": 0, "error": ""}
            for w in writings]
    names = {s.get("function", {}).get("name", "") for s in schemas}

    async def one(k: int) -> None:
        w, mine, out = writings[k], parts[k], done[k]
        lanes = {lane_of(c.name, tools) for c in mine} - {""}
        offered = [s for s in _allowed(schemas, lanes, tools)]
        allowed = {s["function"]["name"] for s in offered}
        convo = list(base)
        theirs = {_same(c) for c in parts[1 - k]}
        prompt = PART_PROMPT.format(mine=_wanted(mine), theirs=_wanted(parts[1 - k]))
        heard = 0
        step = ChatResult(text="", tool_calls=list(mine))
        while True:
            calls, _ = await loop._hands(convo, step)
            if not out["opening"] and not out["later"]:
                out["opening"] = [c for c in calls if not c.get("auto")]
                out["later"].extend(c for c in calls if c.get("auto"))
            else:
                out["later"].extend(calls)
            board[k]["steps"] += 1
            show("step")
            note = prompt
            steers = loop._part_steers[heard:]
            heard += len(steers)
            if steers:
                note += ("\nThey just said, while you work: " + " / ".join(steers)[:300]
                         + ". If it changes your part, adjust; if it cancels your part, stop and "
                           "write what you have.")
            convo.append({"role": "user", "content": note})
            reply = await loop.llm.chat(convo, tools=offered or None, temperature=w.temperature,
                                        max_tokens=loop.max_tokens, model=loop._instance(w))
            convo.pop()
            text = _MARKER.sub("", reply.text or "").strip()
            # The other part's calls are its own: repeating one only does its work twice.
            wanted = [c for c in (reply.tool_calls or []) if c.name in names and _same(c) not in theirs]
            if not wanted:
                out["finding"] = text[:600]
                return
            fit = [c for c in wanted if c.name in allowed and can_run_in_part(c, tools)]
            if len(fit) != len(wanted) or out["rounds"] >= rounds:
                # Outside its lane, needs approval, or out of steps: the voice
                # takes the next step from the shared mind after the merge.
                left = ", ".join(c.name for c in wanted if c not in fit) or "more steps"
                out["finding"] = ((text + " ") if text else "") + f"(next step left for after: {left[:160]})"
                return
            out["rounds"] += 1
            step = ChatResult(text=text, tool_calls=fit)

    async def guarded(k: int) -> None:
        t = time.perf_counter()
        try:
            await asyncio.wait_for(one(k), timeout=loop.part_timeout)
            board[k]["state"] = "done"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one part failing leaves the other standing
            done[k]["error"] = f"{type(exc).__name__}: {exc}"[:200] or "failed"
            board[k]["state"] = "failed"
        board[k]["took_ms"] = round((time.perf_counter() - t) * 1000)

    try:
        await asyncio.gather(guarded(0), guarded(1))
    finally:
        loop._parts_live = False
    show("end")

    by_id: dict[str, dict] = {}
    for k, got in enumerate(done):
        for i, tc in enumerate(parts[k]):
            opened = got["opening"][i] if i < len(got["opening"]) else None
            by_id[tc.id] = opened or {
                "name": tc.name, "args": tc.args(),
                "result": f"the {tc.name} attempt failed: {got['error'] or 'no result'}"}
    calls: list[dict] = []
    tool_msgs: list[dict] = []
    last_of = {parts[k][-1].id: k for k in range(2) if parts[k]}
    lines = []
    for tc in res.tool_calls:
        c = dict(by_id[tc.id], part=writings[0 if tc in parts[0] else 1].key)
        content = str(c["result"])
        k = last_of.get(tc.id)
        if k is not None:
            got = done[k]
            later = "; ".join(f"{x['name']}({json.dumps(x['args'], ensure_ascii=False)[:120]}) -> "
                              f"{str(x['result'])[:1200]}" for x in got["later"])
            finding = got["finding"] or (f"failed: {got['error']}" if got["error"] else "")
            if later:
                content += f"\n[then, for this part] {later}"
            if finding:
                content += f"\n[this part's result] {finding}"
            lines.append(f"- {board[k]['label']}: {finding or content[:300]}")
        calls.append(c)
        tool_msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content})
    for k, got in enumerate(done):
        calls.extend(dict(x, part=writings[k].key) for x in got["later"])
    msgs.extend(tool_msgs)
    loop._parts_rounds = sum(got["rounds"] for got in done)
    loop._parts_note = PARTS_NOTE.format(lines="\n".join(lines))
    loop._event({"type": "hands", "kind": "parts", "phase": "end", "n": len(res.tool_calls),
                 "took_ms": round((time.perf_counter() - t0) * 1000),
                 "rounds": loop._parts_rounds})
    return calls, tool_msgs
