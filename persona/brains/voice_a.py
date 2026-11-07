"""A - the voice. Streams to the person, uses its tools mid-sentence, and
declares its own feeling (the chord) in a private suffix that is removed
before anyone sees it. A is the one who *experiences* the emotional spectrum:
its declared chord becomes the key that pulls mood-congruent memories.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

from .. import emotion
from ..llm import LLM, ToolCall
from .prompts import FEEL_CLOSE, FEEL_OPEN

_MARKERS = (FEEL_OPEN, "[[feel:", "[feel:")
_FEEL_RE = re.compile(r"(?:⟦|\[\[|\[)feel:(.*?)(?:⟧|\]\]|\]|$)", re.S | re.I)

A_TOOLS = [
    {"type": "function", "function": {
        "name": "fetch_consults",
        "description": "See what your parallel minds have worked out about the current question so far. "
                       "Returns whatever is ready (waits at most a moment) and who is still working.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "check_clock",
        "description": "The same stopwatch already in front of you: place, local time, and how long "
                       "this work has been open. Call it when you want the face again mid-thought.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "delegate_task",
        "description": "Hand heavy multi-step work (research, reading, coding, files) to your workshop. It runs "
                       "in the background; the result comes back to you later.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "What to do, fully specified."}},
            "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "The only memory search tool. Search long-term facts, conversation history, "
                       "entities and saved document passages together. Combines semantic embeddings "
                       "and exact-word matching, including original conversation records. "
                       "Returns grounded, source-labeled evidence rather than guesses.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Specific question, name, phrase or keyword to search everywhere."}},
            "required": ["query"]}}},
    # work_in_background / stop_background_work / nudge_background_work /
    # answer_background_work are gone. They addressed brains/director.py: one
    # queued worker, tools only through the hosted gateway webhook, its own
    # private history. That is a second, rival R system, and it only works for
    # one host. Work is delegated to the concurrent lines in inference_lanes.py,
    # which run the caller's own tools over the shared history, so every lobe
    # sees the same transcript and the host sees every tool call. The voice has
    # no delegation tool at all: the router opens lines from what was asked.
    {"type": "function", "function": {
        "name": "adaptive_goal",
        "description": "Start a durable, evidence-verified goal on your real browser/computer/cloud tools. "
                       "Both lobes can work on independent parts while you keep talking. "
                       "Use when the person asks you to execute substantial work. Report the task id "
                       "and running status; never announce completion until independently verified.",
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "The complete authorized result to achieve."}},
            "required": ["goal"]}}},
    {"type": "function", "function": {
        "name": "adaptive_goal_status",
        "description": "Read verified progress of a durable adaptive goal while you continue speaking.",
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string", "description": "Task identifier returned by adaptive_goal."}},
            "required": ["task_id"]}}},
    {"type": "function", "function": {
        "name": "stop_background_work",
        "description": "Abort the work that is running right now. Use when the person says to stop, "
                       "or when you have decided to pivot before the work finishes.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    # Not a memory tool. A tool running somewhere else can ask you something and
    # then wait; this is the only way the waiter resumes. The question arrives in
    # your brief with its id, so you are never guessing which one you mean.
    {"type": "function", "function": {
        "name": "reply_to_question",
        "description": "Answer a question a running tool is waiting on, using the id from your brief. "
                       "Until you do, that work is paused; if you would rather it just proceed on its "
                       "own default, say so in your reply and the default will apply.",
        "parameters": {"type": "object", "properties": {
            "question_id": {"type": "string", "description": "The id exactly as the brief gave it."},
            "answer": {"type": "string", "description": "What to tell the waiting tool."}},
            "required": ["question_id", "answer"]}}},
]


class _NoFiller(Exception):
    """Internal: the caller asked for the plain baseline (no filler line)."""


class FeelFilter:
    """Streaming filter: passes text through, swallows the private feel suffix."""

    def __init__(self):
        self.buf = ""
        self.captured = ""
        self.hidden = False

    def feed(self, piece: str) -> str:
        if self.hidden:
            self.captured += piece
            return ""
        self.buf += piece
        for m in _MARKERS:
            pos = self.buf.find(m)
            if pos != -1:
                out, self.captured, self.hidden = self.buf[:pos], self.buf[pos:], True
                self.buf = ""
                return out
        # hold back a tail that could be the start of a marker
        hold = 0
        for m in _MARKERS:
            for k in range(min(len(m) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(m[:k]):
                    hold = max(hold, k)
                    break
        out = self.buf[:len(self.buf) - hold] if hold else self.buf
        self.buf = self.buf[len(out):]
        return out

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out

    def chord(self) -> dict:
        m = _FEEL_RE.search(self.captured)
        return emotion.parse_declaration(m.group(1)) if m else {}


def strip_feel(text: str) -> tuple[str, dict]:
    f = FeelFilter()
    out = f.feed(text) + f.flush()
    return out.rstrip(), f.chord()


ToolExec = Callable[[ToolCall], Awaitable[str]]

_ENDS = (".", "!", "?", "…", "\n")


_BOUNDARY = re.compile(r"[.!?…][\"')\]*_»”’]*(?=\s|$)|\n")


def at_boundary(text: str) -> bool:
    """True when the visible text has just finished a sentence (closing quotes/brackets allowed)."""
    t = text.rstrip(" \t").rstrip("\"')*_»”’")
    return bool(t) and (t.endswith(_ENDS) or text.endswith("\n"))


def first_boundary(piece: str) -> int:
    """Index just after the first sentence end inside ``piece`` (-1 if none). Tokens often carry the end of
    one sentence and the start of the next, so the cut can fall inside a piece."""
    m = _BOUNDARY.search(piece)
    return m.end() if m else -1


def last_boundary(text: str) -> int:
    ends = [m.end() for m in _BOUNDARY.finditer(text)]
    return ends[-1] if ends else -1


def inner_voice(items: list[dict]) -> str:
    lines = ["[inner voice - not from the person; they cannot see this]"]
    for it in items:
        lines.append(it["text"])
    lines.append(
        "Carry on with the SAME reply from exactly where you stopped - no greeting, do not repeat "
        "what you already said, never mention this note or where it came from. If it is a correction "
        "of a fact you stated, use the right fact from here on, plainly, without announcing a "
        "correction; never correct what the person said. If it is a memory, use it the way someone who just remembered would. "
        "If it is a report from your background work, mention it as something you just noticed "
        "yourself, never as a message from anyone else. "
        "Otherwise just finish your thought."
    )
    return "\n".join(lines)


@dataclass
class VoiceResult:
    text: str = ""
    chord: dict = field(default_factory=dict)
    tools_used: list[str] = field(default_factory=list)
    heard: list[dict] = field(default_factory=list)   # inner-voice items A was interrupted with


async def speak(llm: LLM, messages: list[dict], *, tools: list[dict] | None, run_tool: ToolExec,
                result: VoiceResult, max_rounds: int = 4, model: str | None = None,
                temperature: float = 0.8, inner: "asyncio.Queue | None" = None,
                on_text: Callable[[str], None] | None = None, max_interruptions: int = 3,
                max_tokens: int | None = None, filler: bool = True,
                host_tools: "set[str] | None" = None,
                host_sent: "Callable[[], bool] | None" = None,
                orient: "Callable[[str], str] | None" = None) -> AsyncIterator[str]:
    """Stream A's answer, executing tool calls between rounds. Yields visible text.

    ``inner`` is the inner-voice queue the other lobes write into while A speaks. When it holds
    something and A has just finished a sentence, A's stream is stopped there, A hears the note, and
    continues the same reply. ``on_text`` sees the visible text so far after every piece.

    A speaks as long as A has something to say. ``max_tokens=None`` is the default and sends
    no cap to the provider — the model's own context ceiling applies. Pass an int only if you
    explicitly want to truncate.
    """
    msgs = list(messages)
    feel = FeelFilter()
    tool_rounds = 0
    interruptions = 0

    def _with_face(doing: str = "") -> list[dict]:
        """Refresh the stopwatch on the system line. Tool bodies stay untouched."""
        if orient is None or not msgs or msgs[0].get("role") != "system":
            return msgs
        try:
            line = str(orient(doing) or "").strip()
        except Exception:  # noqa: BLE001
            return msgs
        if not line:
            return msgs
        content = str(msgs[0].get("content") or "")
        cut = content.rfind("\n\n[stopwatch]")
        if cut != -1:
            content = content[:cut]
        stamped = {**msgs[0], "content": content + "\n\n" + line + "\n[not shown to them]"}
        return [stamped, *msgs[1:]]

    while True:
        round_text = ""
        said = ""        # what the person has actually seen this round
        calls: list[ToolCall] = []
        interrupted = False
        use_tools = tools if tool_rounds < max_rounds - 1 else None
        kwargs = {"tools": use_tools, "temperature": temperature, "max_tokens": max_tokens}
        if model:
            kwargs["model"] = model
        stream = llm.stream(_with_face(), **kwargs)
        try:
            async for ev in stream:
                if ev["type"] == "text":
                    round_text += ev["text"]
                    visible = feel.feed(ev["text"])
                    can_stop = (inner is not None and not inner.empty() and interruptions < max_interruptions
                                and not feel.hidden)
                    if can_stop and visible:
                        cut = 0 if at_boundary(result.text) and visible[:1].isspace() else first_boundary(visible)
                        if cut != -1:
                            visible, interrupted = visible[:cut], True
                    if visible:
                        result.text += visible
                        said += visible
                        yield visible
                        if on_text is not None:
                            on_text(result.text)
                    if not interrupted and can_stop and at_boundary(result.text + feel.buf):
                        interrupted = True
                    if interrupted:
                        break
                elif ev["type"] == "tool_call":
                    calls.append(ev["call"])
        finally:
            await stream.aclose()
        if interrupted:
            heard = []
            while not inner.empty():
                heard.append(inner.get_nowait())
            interruptions += 1
            result.heard += heard
            msgs.append({"role": "assistant", "content": said})
            msgs.append({"role": "user", "content": inner_voice(heard)})
            if not result.text.endswith((" ", "\n")):
                result.text += " "
                yield " "
            feel = FeelFilter()
            continue
        if not calls:
            break
        tool_rounds += 1
        tail = feel.flush()
        if tail:
            result.text += tail
            yield tail
        feel = FeelFilter() if not feel.hidden else feel
        msgs.append({"role": "assistant", "content": round_text or None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments or "{}"}}
            for c in calls]})

        # TALK-WHILE-DOING: kick each tool off as an asyncio Task, then stream a
        # short filler line to the person while the tools run. The dual-lobe
        # design is literally built for this - A keeps the voice; the tool
        # runs on a parallel task. Single-model agents (ChatGPT, Hermes) stall
        # here because they have one thread; we don't.
        async def _one_tool(call):
            result.tools_used.append(call.name)
            try:
                return call, await run_tool(call)
            except Exception as exc:  # noqa: BLE001
                return call, json.dumps({"error": str(exc)[:200]})

        tool_tasks = [asyncio.create_task(_one_tool(c)) for c in calls]

        # A visiting agent's tool (READY-PLAN Step 3): the agent runs it, not us.
        # End the turn right here so the agent gets the call at once - no filler,
        # no second round asking the model about a result that is not back yet.
        if host_tools and any(c.name in host_tools for c in calls):
            done_pairs = await asyncio.gather(*tool_tasks)
            if host_sent is None or host_sent():
                break
            # Nothing went out (the director stopped it): the model decides again.
            for c, out in done_pairs:
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            continue

        # Short filler stream - the model acknowledges what it's doing without
        # waiting on the tool. Capped at a few tokens so it's a line, not a
        # paragraph. Cuts short the moment every tool resolves.
        filler_names = ", ".join(c.name for c in calls)
        filler_system = {
            "role": "system",
            "content": (
                "You just chose to use these tools: " + filler_names + ". While they run, "
                "say ONE short natural line to the user acknowledging what you're doing "
                "(e.g. 'one sec, let me check', 'looking at that now'). Under 12 words. "
                "Do not pretend you have the result yet. Do not say the tool names out "
                "loud. No feel suffix. Keep it in the same voice as the rest of your reply."
            ),
        }
        filler_feel = FeelFilter()
        try:
            if not filler:
                # Plain single-model baseline (the D0 compare switch): silent
                # while the tools run, like an ordinary one-thread agent.
                await asyncio.gather(*tool_tasks)
                raise _NoFiller
            filler_stream = llm.stream(
                _with_face("while " + filler_names + " runs") + [filler_system],
                tools=None, temperature=0.7, max_tokens=40,
                **({"model": model} if model else {}),
            )
            emitted_any = False
            try:
                async for ev in filler_stream:
                    if ev["type"] == "text":
                        visible = filler_feel.feed(ev["text"])
                        if visible:
                            emitted_any = True
                            result.text += visible
                            yield visible
                            if on_text is not None:
                                on_text(result.text)
                    if all(t.done() for t in tool_tasks):
                        break
            finally:
                await filler_stream.aclose()
            flush_tail = filler_feel.flush()
            if flush_tail:
                result.text += flush_tail
                yield flush_tail
                emitted_any = True
            if emitted_any and not result.text.endswith((" ", "\n")):
                result.text += " "
                yield " "
        except Exception as exc:  # noqa: BLE001
            # Filler is a nicety, not load-bearing. If it fails, A still gets
            # the tool result in the next round - the user just doesn't hear
            # the "one sec" line. Log and continue.
            pass

        # Collect tool outputs (likely already resolved while filler streamed)
        done_pairs = await asyncio.gather(*tool_tasks)
        for c, out in done_pairs:
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
    tail = feel.flush()
    if tail:
        result.text += tail
        yield tail
    result.text = result.text.strip()
    result.chord = feel.chord()
