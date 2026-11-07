"""Lanes for any agent: the voice talks while every line of work keeps moving.

This is the default runtime for every host that brings its own tools; there is no mode.

Persona is inference. In one reply to any host, the voice (L) starts streaming its
words at once while, at the same moment, the router and every active line of work
(R1..Rn) decide their next actions. The reply is the voice's text followed by the
tool calls of every moving line; the host runs them all and sends the results
back, and the next reply moves every line again. A tool-call id carries its line,
so each result goes back to the line that asked for it.

The voice never learns that lines or workers exist: it reads one first-person
story. The router and the lines are told the truth, because knowing who owns
which work is what keeps them from doing it twice. Only the router starts,
steers or stops a line; a line only takes the next step of its own work.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import logging
import re
import secrets
import time
from dataclasses import dataclass, field

from .brains.voice_a import FeelFilter
from .host_delegation import _checked_action, _completed_actions, is_host_background_tool
from .llm import strip_reasoning

log = logging.getLogger(__name__)

_LINE_IN_ID = re.compile(r"^call_(ln[0-9a-f]{8})_")
# Zero means no fixed lane ceiling. Optional operator limit protects constrained deployments.
# Lane creation never waits for a slot; a configured limit rejects new work explicitly.
MAX_ACTIVE_LINES = max(0, int(os.getenv('SAWII_MAX_ACTIVE_LINES', '0')))
MAX_CALLS_PER_LINE = 3
QUIET = "(quiet)"
_PROMISE = re.compile(r"\b(?:i'll|i will|let me|i'm going to|i am going to|on it|starting|"
                      r"i'm (?:checking|looking|starting|working)|checking|looking into)\b", re.I)


@dataclass
class Line:
    id: str
    label: str
    topic: str
    instruction: str
    status: str = "active"          # active | done | failed | stopped
    result: str = ""
    notes: list[str] = field(default_factory=list)
    told: bool = False              # the voice has told the person how it ended
    steps: int = 0
    parent: str = ""                # a piece fanned out from this line
    children: list[str] = field(default_factory=list)
    waiting: bool = False           # waiting for its pieces to report back
    delegated: bool = False         # running inside one of the host's background agents
    host_agent_id: str = ""
    host_tool_call_id: str = ""
    host_status: str = ""


class LaneBook:
    """Every line of every conversation, by id; durable through the shared timeline."""

    def __init__(self, timeline):
        self.timeline = timeline
        self._lines: dict[tuple[int, str], Line] = {}
        self._count: dict[int, int] = {}
        self._recovered: set[int] = set()
        self._recovery_locks: dict[int, asyncio.Lock] = {}

    async def load_active(self, pid: int) -> None:
        """Recover the latest durable status for every lane after a restart.

        The persisted record is not proof that a host child remains alive;
        background-agent liveness still requires host acknowledgement.
        """
        if self.timeline is None or pid in self._recovered:
            return
        async with self._recovery_locks.setdefault(pid, asyncio.Lock()):
            if pid in self._recovered:
                return
            await self.timeline.initialize()
            rows = await self.timeline.store.all(
                "SELECT t.task_id, t.payload FROM shared_timeline t "
                "JOIN (SELECT task_id, MAX(seq) AS seq FROM shared_timeline "
                "WHERE persona_id=? AND kind='line' AND task_id!='' GROUP BY task_id) x "
                "ON x.seq=t.seq WHERE t.persona_id=? ORDER BY t.seq",
                pid, pid)
            for row in rows:
                try:
                    data = json.loads(row["payload"])
                    lid = str(row["task_id"])
                    label = str(data.get("label") or "R?")
                    if label.startswith("R") and label[1:].isdigit():
                        self._count[pid] = max(self._count.get(pid, 0), int(label[1:]))
                    if data.get("status") != "active" or (pid, lid) in self._lines:
                        continue
                    self._lines[(pid, lid)] = Line(
                        id=lid, label=label, topic=str(data.get("topic") or ""),
                        instruction=str(data.get("instruction") or ""),
                        status="active", result=str(data.get("result") or ""),
                        notes=list(data.get("notes") or []),
                        told=bool(data.get("told")), steps=int(data.get("steps") or 0),
                        parent=str(data.get("parent") or ""),
                        children=list(data.get("children") or []),
                        waiting=bool(data.get("waiting")),
                        delegated=bool(data.get("delegated")),
                        host_agent_id=str(data.get("host_agent_id") or ""),
                        host_tool_call_id=str(data.get("host_tool_call_id") or ""),
                        host_status=str(data.get("host_status") or ""))
                except (ValueError, TypeError, KeyError):
                    log.exception("invalid durable R lane record")
            self._recovered.add(pid)

    async def start(self, pid: int, topic: str, instruction: str) -> Line:
        n = self._count[pid] = self._count.get(pid, 0) + 1
        line = Line(id="ln" + secrets.token_hex(4), label=f"R{n}", topic=topic[:80],
                    instruction=instruction[:2000])
        self._lines[(pid, line.id)] = line
        await self.save(pid, line)
        return line

    async def save(self, pid: int, line: Line) -> None:
        if self.timeline is None:
            return
        try:
            await self.timeline.append(pid, producer=line.label, kind="line", task_id=line.id,
                                       payload={"line": line.id, "label": line.label, "topic": line.topic,
                                                "instruction": line.instruction, "status": line.status,
                                                "result": line.result[:4000], "notes": line.notes[-5:],
                                                "told": line.told, "steps": line.steps,
                                                "parent": line.parent, "children": line.children,
                                                "waiting": line.waiting, "delegated": line.delegated,
                                                "host_agent_id": line.host_agent_id,
                                                "host_tool_call_id": line.host_tool_call_id,
                                                "host_status": line.host_status})
        except Exception:  # noqa: BLE001
            log.exception("could not record line %s", line.id)

    def active(self, pid: int) -> list[Line]:
        """The lines running right now. Live commands steer and stop these."""
        return [line for (persona, _), line in self._lines.items()
                if persona == pid and line.status == "active"]

    def counts(self) -> dict[int, int]:
        """How many lines each persona has running. For the health endpoint."""
        out: dict[int, int] = {}
        for (persona, _), line in self._lines.items():
            if line.status == "active":
                out[persona] = out.get(persona, 0) + 1
        return out

    async def stop(self, pid: int, line_id: str) -> bool:
        """One line, by id. True when it was running and is stopped now."""
        line = await self.get(pid, line_id)
        if line is None or line.status != "active":
            return False
        line.status, line.result = "stopped", line.result or "stopped at their request"
        await self.save(pid, line)
        return True

    async def note(self, pid: int, line_id: str, text: str) -> bool:
        """Mid-work guidance: the line reads this before its next step.

        The line is not restarted and its work is not thrown away - a person
        adding a wrinkle mid-job is the normal case, not an exception.
        """
        line = await self.get(pid, line_id)
        if line is None or line.status != "active" or not text.strip():
            return False
        line.notes.append(text.strip()[:500])
        await self.save(pid, line)
        return True

    async def stop_all(self, pid: int) -> int:
        """Stop every line this persona is running. The person said stop, or the
        voice decided to pivot; nothing may keep working after that."""
        stopped = 0
        for (persona, _), line in list(self._lines.items()):
            if persona == pid and line.status == "active":
                line.status, line.result = "stopped", line.result or "stopped at their request"
                await self.save(pid, line)
                stopped += 1
        return stopped

    async def get(self, pid: int, line_id: str) -> Line | None:
        line = self._lines.get((pid, line_id))
        if line is not None or self.timeline is None:
            return line
        # After a restart: the newest record of this line.
        rows = await self.timeline.store.all(
            "SELECT payload FROM shared_timeline WHERE persona_id = ? AND kind = 'line' AND task_id = ? "
            "ORDER BY seq DESC LIMIT 1", pid, line_id)
        if not rows:
            return None
        p = json.loads(rows[0]["payload"])
        line = Line(id=line_id, label=p.get("label", "R?"), topic=p.get("topic", ""),
                    instruction=p.get("instruction", ""), status=p.get("status", "active"),
                    result=p.get("result", ""), notes=list(p.get("notes") or []),
                    told=bool(p.get("told")), steps=int(p.get("steps") or 0),
                    parent=str(p.get("parent") or ""), children=list(p.get("children") or []),
                    waiting=bool(p.get("waiting")), delegated=bool(p.get("delegated")),
                    host_agent_id=str(p.get("host_agent_id") or ""),
                    host_tool_call_id=str(p.get("host_tool_call_id") or ""),
                    host_status=str(p.get("host_status") or ""))
        self._lines[(pid, line_id)] = line
        return line


def line_of(call_id: str) -> str:
    m = _LINE_IN_ID.match(call_id or "")
    return m.group(1) if m else ""


def _call_id(line: Line) -> str:
    return f"call_{line.id}_{secrets.token_hex(3)}"


def _short(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _json(text: str) -> dict:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        m = re.search(r"\{[\s\S]*\}", text or "")
        if not m:
            return {}
        try:
            value = json.loads(m.group())
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2)


_TARGET_KEYS = ("path", "file", "filepath", "file_path", "filename", "target",
                "dir", "directory", "url", "notebook")
# A line can change a file without naming it in a path argument, by running code
# that writes it. Ownership has to see that too, or two lines clobber each other.
_WRITE_MODE = re.compile(
    r"""open\(\s*(['"])(?P<open>[^'"]+)\1\s*,\s*['"][wax+]          # open(path, 'w')
      | (['"])(?P<chain>[^'"]+)\3\s*\)\.(?:write_text|write_bytes|writelines|write)\(
      | (['"])(?P<pend>[^'"]+)\4\s*,\s*['"]w['"]                   # os.open-ish ("path", "w")
      """, re.VERBOSE)
_SHELL_REDIRECT = re.compile(r">>?\s*([\w./\\-]+)")
_CODE_KEYS = ("code", "command", "script", "shell", "bash", "python")
_LOOKS_LIKE_TARGET = re.compile(r"^(?:https?://|\.{0,2}[/\\]|[\w.-]+[/\\]|[\w.-]+\.\w{1,8}$)")
_CHANGING = ("write", "edit", "patch", "create", "delete", "remove", "move", "rename",
             "save", "insert", "update")
_RUNNING = ("run", "exec", "shell", "bash", "python", "command", "terminal")


def _as_target(value) -> str:
    text = value.strip().rstrip("/\\").lower() if isinstance(value, str) else ""
    return text if _LOOKS_LIKE_TARGET.match(text) else ""


def write_targets(name: str, arguments) -> list[str]:
    """What a call changes. A read changes nothing and owns nothing.

    Two lines may run at once only while they do not change the same thing, so
    this is checked from the arguments, never from a model's assurance.
    """
    tool = str(name or "").lower()
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except (ValueError, TypeError):
        return []
    if not isinstance(args, dict):
        return []
    changes = any(word in tool for word in _CHANGING)
    out: list[str] = []
    for key in _TARGET_KEYS:
        target = _as_target(args.get(key))
        if target and (changes or key == "url"):
            out.append(target)
    if any(word in tool for word in _RUNNING):
        for key in _CODE_KEYS:
            value = args.get(key)
            if not isinstance(value, str):
                continue
            for match in _WRITE_MODE.finditer(value):
                found = next((g for g in match.groupdict().values() if g), "")
                if found:
                    out.append(found.strip("/\\").lower())
            out += [m.group(1).strip("/\\").lower() for m in _SHELL_REDIRECT.finditer(value)]
    return [t for t in out if t]


def claims_only(name: str, arguments) -> bool:
    """A run call whose whole code only prints words: it says nothing about the world.

    In live testing a line spent its step on run_python("print('Fixed parser.py')")
    and the reply then repeated that text as if it were a result. Printing a value
    the code computed is real work; printing words the model made up is not, so a
    step like that is refused before it runs.
    """
    tool = str(name or "").lower()
    if not any(word in tool for word in _RUNNING) or write_targets(name, arguments):
        return False
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except (ValueError, TypeError):
        return False
    if not isinstance(args, dict):
        return False
    # A runner with no code argument at all (run_tests, a shell call with only a cwd)
    # is a real command; only a code argument that carries no code is words-only.
    present = [args[k] for k in _CODE_KEYS if k in args]
    if not present:
        return False
    code = next((v for v in present if isinstance(v, str)), "")
    if not code.strip():
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if not tree.body:
        return True
    for statement in tree.body:
        # Only print(<literals>) counts as saying nothing. Anything else - an
        # import, a read, a call inside print - may be doing something real.
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id in {"print", "echo"}):
            return False
        printed = list(statement.value.args) + [k.value for k in statement.value.keywords]
        if not printed or any(not isinstance(p, ast.Constant) or not isinstance(p.value, str)
                              for p in printed):
            return False
    return True


def same_work(a: str, b: str, threshold: float = 0.6) -> bool:
    """Mechanical duplicate check, independent of any model's judgment."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


@dataclass
class Transcript:
    messages: list[dict]
    calls: dict[str, tuple[str, str]]       # call id -> (tool, arguments)
    answered: set[str]                      # call ids that have a result
    fresh: list[dict]                       # results since the last reply
    new_message: str                        # what the person just said, if they spoke
    line_ids: list[str]                     # lines seen in this conversation, oldest first


def read_transcript(messages: list[dict]) -> Transcript:
    calls: dict[str, tuple[str, str]] = {}
    answered: set[str] = set()
    line_ids: list[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls[str(tc.get("id"))] = (str(fn.get("name") or ""), str(fn.get("arguments") or "{}"))
                lid = line_of(str(tc.get("id")))
                if lid and lid not in line_ids:
                    line_ids.append(lid)
        elif m.get("role") == "tool":
            answered.add(str(m.get("tool_call_id")))
    fresh: list[dict] = []
    for m in reversed(messages):
        if m.get("role") != "tool":
            break
        fresh.insert(0, m)
    new_message = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            break
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            new_message = str(content or "")
            break
    return Transcript(messages, calls, answered, fresh, new_message, line_ids)


def story(messages: list[dict], labels: dict[str, Line], *, voice: bool, budget: int = 14000) -> str:
    """The conversation and every action in it. ``voice`` tells it in the first person."""
    out: list[str] = []
    names: dict[str, str] = {}
    for m in messages:
        role = m.get("role")
        if role == "user":
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            out.append(f"{'They' if voice else 'The person'} said: {_short(content, 1500)}")
        elif role == "assistant":
            if m.get("content"):
                out.append(f"{'I' if voice else 'The voice'} said: {_short(m['content'], 1200)}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                line = labels.get(line_of(str(tc.get("id"))))
                tag = f"[{line.topic}] " if line else ""
                who = "I" if voice else (f"line {line.label}" if line else "the voice")
                names[str(tc.get("id"))] = tag
                out.append(f"{tag}{who} used {fn.get('name')}({_short(fn.get('arguments'), 300)})")
        elif role == "tool":
            tag = names.get(str(m.get("tool_call_id")), "")
            out.append(f"{tag}result: {_short(m.get('content'), 900)}")
    kept: list[str] = []
    used = 0
    for text in reversed(out):
        if used + len(text) > budget:
            kept.append(f"(earlier: {len(out) - len(kept)} older moments)")
            break
        kept.append(text)
        used += len(text) + 1
    return "\n".join(reversed(kept))


VOICE = """Right now you are talking with them while your work keeps moving; they see every action
you take as it happens. Say something short and natural - one or two sentences - in the first person:
- If they just said something, answer it. If it asks for something to be done, say briefly what you
  are doing about it. Never state a finding you do not have yet.
- Otherwise tell them what just happened in your work that matters: something finished, something
  you found, something stuck. Skip routine steps.
- Only state facts that are in your story below. If nothing new is worth saying, reply exactly: (quiet)
- A line shown as working is NOT finished. Never say finished, fixed, done, written, created or
  corrected about it - say what you are doing. Only a line listed as finished or failed may be
  described as ended, and then say what its result actually says. Never repeat what you already said."""

# A claim about work that has not ended. The voice is a model, and a model that has
# just said "I'm fixing it" will happily say "I fixed it" one turn early; a person
# who is told a file was changed when it was not is worse off than one told nothing.
_CLAIMS_ENDED = re.compile(
    r"\b(?:fix(?:ed|es)?|done|finished|wrote|written|created|added|corrected|resolved|completed|"
    r"all set|covered|updated|saved|made the change|it works now|edited|edits?|changed|patched|"
    r"replaced|converted|verified|tested|ran|executed|deleted|removed|renamed|passed)\b", re.I)
_IN_PROGRESS = re.compile(
    r"\b(?:i'?m|we'?re|i'?ll|still|going to|about to|starting|beginning|checking|reading|looking|"
    r"writing|trying|waiting|working|let me|one moment|shortly|next)\b", re.I)


def _wanted(sentence: str, finished: bool) -> bool:
    return finished or not _CLAIMS_ENDED.search(sentence) or bool(_IN_PROGRESS.search(sentence))


class ClaimGate:
    """Words out as they come, except a sentence that claims work has ended.

    A sentence whose words are all safe streams straight through, so the first
    words reach the person at the usual speed. As soon as a claim word appears
    the sentence is held back to its full stop - whole sentences only, so a
    dropped one leaves no dangling fragment - and if no line has ended it is
    never said.
    """

    def __init__(self, finished: bool = False):
        self.finished = finished
        self.said = 0        # characters let through, for the first-word timing
        self.pending = ""
        self.held = False

    def _take(self, text: str) -> str:
        self.said += len(text)
        return text

    def feed(self, piece: str) -> str:
        out = ""
        self.pending += piece
        while self.pending:
            if self.held or _CLAIMS_ENDED.search(self.pending):
                self.held = True
                end = re.search(r"[.!?][\"')\]]*(\s|$)", self.pending)
                if not end:
                    break          # wait for the sentence to finish
                sentence, self.pending = self.pending[: end.end()], self.pending[end.end():]
                if _wanted(sentence, self.finished):
                    out += self._take(sentence)
                self.held = False
                continue
            # Safe so far: let it out, but keep the last partial word, since the
            # next chunk can still extend it into a claim ("wrot" -> "written").
            cut = max(self.pending.rfind(" "), self.pending.rfind("\n"))
            if cut < 0:
                break
            out += self._take(self.pending[: cut + 1])
            self.pending = self.pending[cut + 1:]
        return out

    def flush(self) -> str:
        """At the end of the voice's turn: a finished sentence may still be kept."""
        text, self.pending = self.pending, ""
        if not text.strip():
            return ""
        return self._take(text) if _wanted(text, self.finished) else ""

VOICE_END = """Your work on this has just finished. Tell them what you did and what came of it, in the
first person: concise but complete, with the concrete results (names, numbers, files, links) from the
record below and nothing that is not in it. If a part failed or was stopped, say which and why."""

ROUTER = """You are the router for a persona working as an agent. A separate voice talks with the person;
you never speak. You decide what work exists, and it starts the moment you decide. Every line you start
runs at the same time as every other line: there is no queue and no limit on lines.

A line is a chain of steps where each step needs the previous step's result. Make separate lines
whenever pieces can proceed at the same time without waiting on each other and without changing the
same file, page or form - that is what makes the work fast. Keep one line when one action covers it.
Each line's instruction must be complete on its own. Give each new line its first tool call.

Decide only from the history and the lines below:
- If the person asked for something to be done, or the voice told them it would do something, and
  no line is doing it, start it.
- If a line's result revealed more independent work (for example several files or pages that each
  need handling), start those as new lines and steer that line so nothing is done twice.
- If the person changed what they want, steer the line doing it. If they said stop, stop it.
- Steer exactly ONE line per change: pick the line whose recent work the message best matches. Never
  send the same instruction to several lines, and never steer a line for plain conversation.
- Never start what is already running or finished. Never start work for plain conversation or for
  something answered by talking. Never retry a failed line the same way.
- A line's first call must act on the real world: read, search, write, run. Never a call whose
  arguments only restate an answer, and never a second line that changes something a running line
  already changes - if the answer will come from another line's work, start nothing.
- Change a file with the host's own write or edit tool, never by running code that writes it: the
  host has to see which file each line changes. Use run for what only run can do - executing the
  tests, computing a number, checking the result.

Give every start an "id". In "after" list the ids whose results it needs first, and in "writes" what
it changes (files, pages, forms). Code puts dependent work and work on the same thing in one lane, in
order, and runs everything else at once - so list each piece of work separately.

Return only JSON:
{"start": [{"id": "a", "after": [], "writes": [], "topic": "2-5 word label", "do": "complete instruction for the line",
            "call": {"name": "tool name", "arguments": {}}}],
 "steer": [{"line": "line id", "note": "the change, in one sentence"}],
 "stop": ["line id"]}
Use empty lists when nothing should change. Use only these tools:
"""

LINE = """You are line {label} of the work, running at the same time as other lines. Your line: {instruction}
{notes}Other lines own: {others}. Never do their work and never repeat an action already in the history.
You share one timeline with the voice lobe: plain things the person says are for the voice to answer,
not instructions to you. Act on user input ONLY when it is addressed to your lane ('ln={id}=...') or is
already part of your line.
Take the next step of YOUR line only: call the tool or tools it needs next (at most {most}, and several
only if they do not depend on each other), grounded in the results so far. If its next step is several
independent pieces, call fan_out with them instead: each runs at once on its own worker and you continue
when all report back. If your line is complete,
reply with its final result - the concrete facts, files or changes - and no tool call. If it cannot be
done, reply with exactly what blocked it and no tool call. Never invent a result."""


def _schema_view(tools: list[dict]) -> str:
    view = [{"name": t["function"]["name"],
             "description": str(t["function"].get("description", ""))[:400],
             "parameters": t["function"].get("parameters", {})}
            for t in tools if isinstance(t, dict) and isinstance(t.get("function"), dict)]
    return json.dumps(view, ensure_ascii=False)[:14000]


def _canonical(name: str, arguments) -> tuple[str, str]:
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except ValueError:
        args = {}
    return name, json.dumps(args, sort_keys=True)


async def _speak(worker, messages: list[dict], max_tokens: int):
    """Stream the voice; hold back a reply that is only "(quiet)"."""
    feel = FeelFilter()
    held = ""
    released = False
    stream = worker.stream(messages, tools=None, temperature=0.6, max_tokens=max_tokens)
    try:
        async for ev in stream:
            if ev.get("type") != "text":
                continue
            piece = feel.feed(ev["text"])
            if not piece:
                continue
            if released:
                yield piece
                continue
            held += piece
            if len(held.strip()) > len(QUIET) or not QUIET.startswith(held.strip()[: len(QUIET)]):
                released = True
                yield held
    finally:
        await stream.aclose()
    tail = feel.flush()
    if not released:
        held += tail
        if held.strip() and held.strip() != QUIET:
            yield held
    elif tail:
        yield tail


MAX_OWN_ROUNDS = 4
_LANE_MARK = re.compile(r"\[persona-lane (ln[0-9a-f]{8})\]")


def lane_mark(line: Line) -> str:
    return f"[persona-lane {line.id}]"


def marked_lane(messages: list[dict]) -> str:
    """A host background agent working one of our lanes carries the lane's mark in its task."""
    for m in messages:
        if m.get("role") in ("system", "user"):
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            found = _LANE_MARK.search(str(content or ""))
            if found:
                return found.group(1)
    return ""
FAN_OUT = {"type": "function", "function": {
    "name": "fan_out",
    "description": ("Split the next step of your line into independent pieces that run at the same time, "
                    "each on its own worker. Only for pieces that do not need each other's results and do "
                    "not change the same thing. Your line continues once every piece has reported back."),
    "parameters": {"type": "object", "properties": {"pieces": {"type": "array", "items": {
        "type": "object", "properties": {"topic": {"type": "string"}, "do": {"type": "string"}},
        "required": ["do"]}}}, "required": ["pieces"]}}}


def _pieces(arguments) -> list[dict]:
    data = _json(arguments) if isinstance(arguments, str) else (arguments or {})
    rows = data.get("pieces") if isinstance(data, dict) else None
    return [{"topic": str(r.get("topic") or r["do"])[:80], "do": str(r["do"]).strip()[:2000]}
            for r in (rows or []) if isinstance(r, dict) and str(r.get("do") or "").strip()]


def group_starts(rows: list) -> tuple[list[dict], int]:
    """The router proposes work; code decides the lanes.

    Work that needs another's result runs in that one's lane, in order. Work that
    changes the same thing shares a lane. Everything else gets its own lane. A
    dependency loop cannot be ordered, so it is dropped. Returns (lanes, dropped).
    """
    rows = [r for r in rows if isinstance(r, dict) and str(r.get("do") or "").strip()]
    keys = [str(r.get("id") or i) for i, r in enumerate(rows)]
    index = {k: i for i, k in enumerate(keys)}
    deps = {i: [index[str(a)] for a in (rows[i].get("after") or []) if str(a) in index and index[str(a)] != i]
            for i in range(len(rows))}
    cyclic: set[int] = set()
    state: dict[int, int] = {}

    def visit(i: int, path: list[int]) -> None:
        if state.get(i) == 1:
            cyclic.update(path[path.index(i):])
            return
        if state.get(i) == 2:
            return
        state[i] = 1
        path.append(i)
        for d in deps[i]:
            visit(d, path)
        path.pop()
        state[i] = 2

    for i in range(len(rows)):
        visit(i, [])
    root = list(range(len(rows)))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    owner: dict[str, int] = {}
    for i, row in enumerate(rows):
        if i in cyclic:
            continue
        for d in deps[i]:
            if d not in cyclic:
                root[find(i)] = find(d)
        call = row.get("call") if isinstance(row.get("call"), dict) else {}
        targets = {str(t).strip().lower() for t in (row.get("writes") or []) if str(t).strip()}
        targets.update(write_targets(call.get("name", ""), call.get("arguments") or {}))
        for t in targets:
            if t in owner:
                root[find(i)] = find(owner[t])
            else:
                owner[t] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        if i not in cyclic:
            groups.setdefault(find(i), []).append(i)
    out: list[dict] = []
    for members in groups.values():
        order: list[int] = []
        left = list(members)
        while left:
            ready = [i for i in left if all(d in order or d in cyclic or d not in members for d in deps[i])]
            nxt = ready[0] if ready else left[0]
            order.append(nxt)
            left.remove(nxt)
        first = rows[order[0]]
        if len(order) == 1:
            out.append(first)
            continue
        out.append({"topic": str(first.get("topic") or first["do"])[:80],
                    "do": "Do these in order, each with the results of the ones before:\n" + "\n".join(
                        f"{n + 1}. {str(rows[i]['do']).strip()}" for n, i in enumerate(order)),
                    "call": first.get("call")})
    return out, len(cyclic)


class Round:
    """One reply to the host: the voice, the router and every ready line, together."""

    def __init__(self, mind, pid: int, worker, system: str, host_system: str,
                 messages: list[dict], tools: list[dict], own_tools: list[dict] | None = None,
                 run_own=None):
        self.m, self.pid, self.worker = mind, pid, worker
        # Persona's own tools run here, inside the step; the host's go back to it.
        self.own = [t for t in (own_tools or []) if isinstance(t, dict) and "function" in t]
        self.own_names = {t["function"]["name"] for t in self.own}
        self.run_own = run_own
        self.as_line = marked_lane(messages)
        self.system, self.host_system = system, host_system
        self.messages, self.tools = messages, tools
        self.book: LaneBook = mind.lanes
        self.tr = read_transcript(messages)
        self.lines: dict[str, Line] = {}
        self.trace: dict = {"lines_started": 0, "lines_moved": 0, "lines_finished": 0,
                            "calls": 0, "duplicates_refused": 0, "words_only_refused": 0}
        self.collisions: list[tuple[Line, str]] = []
        # Stateful dedupe across the incremental batches: one call per action, whether
        # it arrives in the first batch while the voice speaks or the last after it.
        self._seen: set[tuple[str, str]] = set()
        self._emitted: list[dict] = []

    async def load(self) -> None:
        await self.book.load_active(self.pid)
        for lane in self.book.active(self.pid):
            self.lines[lane.id] = lane
        for lid in self.tr.line_ids:
            line = await self.book.get(self.pid, lid)
            if line is not None:
                self.lines[lid] = line
        for line in list(self.lines.values()):
            if line.parent and line.parent not in self.lines:
                parent = await self.book.get(self.pid, line.parent)
                if parent is not None:
                    self.lines[parent.id] = parent
        for line in list(self.lines.values()):
            for cid in line.children:
                if cid not in self.lines:
                    child = await self.book.get(self.pid, cid)
                    if child is not None:
                        self.lines[cid] = child
        for m in self.tr.fresh:
            line = self.lines.get(line_of(str(m.get("tool_call_id"))))
            if line is not None and self.m.shared_timeline is not None:
                name = self.tr.calls.get(str(m.get("tool_call_id")), ("", ""))[0]
                await self.m.shared_timeline.append(
                    self.pid, producer=line.label, kind="tool_result", task_id=line.id,
                    payload={"tool": name, "call_id": m.get("tool_call_id"), "part": line.topic,
                             "result": {"text": str(m.get("content") or "")[:2000]}})

    def ready(self, line: Line) -> bool:
        """Every call this line made has its result back."""
        mine = [cid for cid in self.tr.calls if line_of(cid) == line.id]
        return (line.status == "active" and not line.waiting and not line.delegated and bool(mine)
                and all(cid in self.tr.answered for cid in mine))

    def voice_context(self, finished: list[Line]) -> str:
        parts = [story(self.messages, self.lines, voice=True)]
        working = [line for line in self.lines.values() if line.status == "active" and not line.parent]
        if working:
            parts.append("What you are working on right now: " + "; ".join(line.topic for line in working))
        if finished:
            parts.append("Finished since you last spoke:\n" + "\n".join(
                f"- {line.topic}: {line.status}. {_short(line.result, 700)}" for line in finished))
        if self.tr.new_message:
            parts.append("They just said: " + _short(self.tr.new_message, 3000))
        else:
            parts.append("They have not said anything new; your work moved on.")
        return "\n\n".join(parts)

    def checked_for_line(self, line: Line, action: dict) -> dict | None:
        """Attach a durable R identity and request background execution mechanically.

        A model can omit a marker or background flag, but the host contract
        must not depend on it following a prompt. Only add arguments declared
        by the host tool's own schema; do not invent private Cursor options.
        """
        if not isinstance(action, dict):
            return None
        name = str(action.get("name") or "")
        match = next((t for t in self.tools if (t.get("function") or {}).get("name") == name), None)
        if match is None or not is_host_background_tool(match):
            return _checked_action(action, self.tools)
        arguments = action.get("arguments")
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        except (TypeError, ValueError):
            return None
        if not isinstance(args, dict):
            return None
        props = (match.get("function") or {}).get("parameters", {}).get("properties") or {}
        for flag in ("run_in_background", "is_background"):
            if flag in props:
                args[flag] = True
        mark = lane_mark(line)
        if name == "delegate_task" and isinstance(args.get("tasks"), list):
            for task in args["tasks"]:
                if isinstance(task, dict) and isinstance(task.get("goal"), str):
                    if mark not in task["goal"]:
                        task["goal"] = mark + " " + task["goal"]
        else:
            for field in ("prompt", "task", "goal"):
                if isinstance(args.get(field), str):
                    if mark not in args[field]:
                        args[field] = mark + " " + args[field]
                    break
        return _checked_action({"name": name, "arguments": args}, self.tools)

    def cursor_subagent_call(self, line: Line) -> dict | None:
        """Use Cursor's advertised Task tool to hand an R lane to the host.

        Persona owns one mind and the lane identity. Cursor supplies the hands.
        Never synthesize a tool that Cursor did not advertise, or invent an
        argument its actual JSON schema does not permit.
        """
        if self.as_line:
            return None  # A child continues its own R lane; it must not spawn peers.
        for tool in self.tools:
            fn = tool.get("function") or {}
            if fn.get("name") not in ("Task", "task") or not is_host_background_tool(tool):
                continue
            schema = fn.get("parameters") or {}
            props = schema.get("properties") or {}
            required = schema.get("required") or []
            # Background subagent definition is installed at
            # .cursor/agents/persona-r.md, inherits the parent's model.
            args = {}
            prompt = lane_mark(line) + " " + line.instruction
            if "prompt" in props:
                args["prompt"] = prompt
            elif "task" in props:
                args["task"] = prompt
            else:
                continue
            if "description" in props:
                args["description"] = line.topic[:80] or "Persona R execution"
            # Use the host's REAL advertised agent types. If the project has
            # persona-r installed, prefer it; otherwise use an available general
            # execution agent instead of silently dropping the native Task call.
            for field in ("subagent_type", "agent"):
                if field not in props:
                    continue
                spec = props[field] if isinstance(props[field], dict) else {}
                accepted = spec.get("enum") or []
                preferred = ("persona-r", "generalPurpose", "general-purpose",
                             "general", "default", "explore")
                if accepted:
                    selected = next((x for x in preferred if x in accepted), None)
                    if selected is None:
                        if field in required:
                            break
                        continue
                    args[field] = selected
                elif field in required:
                    args[field] = "persona-r"
            if any(k not in args for k in required):
                continue
            for flag in ("run_in_background", "is_background"):
                if flag in props:
                    args[flag] = True
            if not all(name in args for name in required):
                continue
            return self.checked_for_line(line, {"name": fn["name"], "arguments": args})
        return None

    async def step(self, line: Line) -> tuple[str, object]:
        others = [f"{o.label} {o.topic}" for o in self.lines.values() if o.status == "active" and o.id != line.id]
        notes = ("Changes the person asked for on your line: " + "; ".join(line.notes[-5:]) + "\n"
                 if line.notes else "")
        rules = LINE.format(label=line.label, instruction=line.instruction, notes=notes,
                            id=line.id,
                            others=", ".join(others) or "nothing else", most=MAX_CALLS_PER_LINE)
        system = (self.host_system + "\n\n" if self.host_system else "") + rules
        spawn = [t["function"]["name"] for t in self.tools if is_host_background_tool(t)]
        if spawn and not self.as_line:
            system += (f"\nFor long work that can run on its own, hand your whole line to a background agent "
                       f"with {spawn[0]}; begin its task text with {lane_mark(line)} exactly.")
        msgs = [{"role": "system", "content": system}] + self.messages + [
            {"role": "user", "content": f"Next step of your line ({line.label}: {line.topic})."}]
        offer = list(self.tools) + self.own + ([] if line.parent else [FAN_OUT])
        try:
            for _ in range(MAX_OWN_ROUNDS):
                res = await self.worker.chat(msgs, tools=offer, temperature=0.2, max_tokens=1500)
                proposed = list(res.tool_calls or [])
                fan = next((tc for tc in proposed if tc.name == "fan_out"), None)
                if fan is not None and not line.parent:
                    pieces = _pieces(fan.arguments)
                    if len(pieces) >= 2:
                        return "fan", pieces
                own = [tc for tc in proposed if tc.name in self.own_names]
                if not own or self.run_own is None:
                    break
                outs = await asyncio.gather(*(self.run_own(tc) for tc in own), return_exceptions=True)
                self.trace["own_calls"] = self.trace.get("own_calls", 0) + len(own)
                msgs = msgs + [{"role": "assistant", "content": res.text or "", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in own]}] + [{"role": "tool", "tool_call_id": tc.id, "content": str(o)[:4000]}
                                        for tc, o in zip(own, outs)]
                if any(tc.name not in self.own_names and tc.name != "fan_out" for tc in proposed):
                    break
        except Exception as exc:  # noqa: BLE001
            return "failed", f"could not think about the next step: {type(exc).__name__}"
        host = [tc for tc in (res.tool_calls or []) if tc.name not in self.own_names and tc.name != "fan_out"]
        if host:
            calls = [self.checked_for_line(line, {"name": tc.name, "arguments": tc.arguments})
                     for tc in host[:MAX_CALLS_PER_LINE]]
            calls = [c for c in calls if c]
            real = [c for c in calls if not claims_only(c["name"], c["arguments"])]
            if len(real) != len(calls):
                self.trace["words_only_refused"] += len(calls) - len(real)
                line.notes.append("printing words about the work is not doing it: use the real tool")
            if real:
                return "calls", real
            if calls:
                return "failed", "printed words about the work instead of doing it"
            return "failed", "proposed an action the tools do not accept"
        text = strip_reasoning(res.text or "").strip()
        return "done", text or "finished with nothing to report"

    async def route(self, note: str = "") -> dict:
        lines = [{"line": line.id, "label": line.label, "topic": line.topic, "doing": line.instruction[:300],
                  "status": line.status, "result": _short(line.result, 300),
                  "host_agent_id": line.host_agent_id, "host_status": line.host_status}
                 for line in self.lines.values()]
        context = (
            (("The host's own instructions (excerpt):\n" + self.host_system[:4000] + "\n\n")
             if self.host_system else "")
            + "History:\n" + story(self.messages, self.lines, voice=False)
            + "\n\nLines in this conversation: " + (json.dumps(lines, ensure_ascii=False) if lines else "none")
            + ("\n\n" + note if note else ""))
        try:
            res = await self.worker.chat([{"role": "system", "content": ROUTER + _schema_view(self.tools)},
                                          {"role": "user", "content": context}],
                                         temperature=0.1, max_tokens=1500, json_mode=True)
        except Exception:  # noqa: BLE001
            log.exception("router failed")
            return None
        decision = _json(res.text)
        return decision if decision or (res.text or "").strip().startswith("{") else None

    async def apply(self, decision: dict) -> list[tuple[Line, dict | None]]:
        """Stops, steers and starts. Returns new lines with their first call, if valid."""
        for lid in decision.get("stop") or []:
            line = self.lines.get(str(lid))
            if line is not None and line.status == "active":
                line.status, line.result = "stopped", line.result or "stopped at their request"
                await self.book.save(self.pid, line)
        for row in decision.get("steer") or []:
            if not isinstance(row, dict):
                continue
            line = self.lines.get(str(row.get("line") or ""))
            note = str(row.get("note") or "").strip()
            if line is not None and line.status == "active" and note:
                line.notes.append(note[:500])
                await self.book.save(self.pid, line)
        started: list[tuple[Line, dict | None]] = []
        running = [line.instruction for line in self.lines.values() if line.status == "active"]
        active = sum(1 for line in self.lines.values() if line.status == "active")
        grouped, dropped = group_starts(decision.get("start") or [])
        self.trace["plan_cycles_dropped"] = self.trace.get("plan_cycles_dropped", 0) + dropped
        for row in grouped:
            if not isinstance(row, dict) or not str(row.get("do") or "").strip():
                continue
            do = str(row["do"]).strip()
            if any(same_work(do, other) for other in running):
                self.trace["duplicates_refused"] += 1
                continue
            if MAX_ACTIVE_LINES and active >= MAX_ACTIVE_LINES:
                break
            line = await self.book.start(self.pid, str(row.get("topic") or do)[:80], do)
            self.lines[line.id] = line
            running.append(do)
            active += 1
            call = row.get("call")
            # A new foreground work request becomes a Cursor background R
            # subagent immediately when the native Task tool is available.
            # No extra LLM decision or foreground tool execution is required.
            checked = self.cursor_subagent_call(line)
            if checked is None:
                checked = self.checked_for_line(line, call) if isinstance(call, dict) else None
            if checked and lane_mark(line) in checked["arguments"]:
                # A router can supply the first background-agent call directly,
                # bypassing step()/settle(). Persist its delegated ownership too.
                line.delegated = True
                await self.book.save(self.pid, line)
                self.trace["delegated_to_host_agents"] = self.trace.get("delegated_to_host_agents", 0) + 1
            started.append((line, checked))
            self.trace["lines_started"] += 1
        return started

    async def run(self):
        """Yields ("text", piece) while the voice speaks, then ("calls", [...])."""
        if self.as_line:
            line = await self.book.get(self.pid, self.as_line)
            if line is not None and line.status == "active":
                # The host's background agent is this lane's R: it takes the
                # lane's next step with the lane's instruction and notes.
                self.lines[line.id] = line
                kind, value = await self.step(line)
                line.steps += 1
                self.trace["background_agent_steps"] = 1
                if kind == "calls":
                    yield "calls", [{"id": _call_id(line), "name": c["name"], "arguments": c["arguments"],
                                     "line": line.label, "topic": line.topic} for c in value]
                    await self.book.save(self.pid, line)
                    return
                await self.settle(line, "failed" if kind == "failed" else "done",
                                  value if kind != "fan" else "split requested inside a background agent")
                yield "text", str(line.result)
                yield "calls", []
                return
        await self.load()
        finished_before = [line for line in self.lines.values() if line.status != "active" and not line.told]
        moving = [line for line in self.lines.values() if self.ready(line)]
        router = asyncio.create_task(self.route()) if self.tools else None
        steps = {line.id: asyncio.create_task(self.step(line)) for line in moving}
        self.trace["lines_moved"] = len(moving)
        said = ""
        voice = [{"role": "system", "content": self.system + "\n\n" + VOICE},
                 {"role": "user", "content": self.voice_context(finished_before)}]
        gate = ClaimGate(finished=bool(finished_before))
        heard = ""
        # Work must never wait for speech (ARCHITECTURE.md section 2), and what the
        # user sees on screen is the measurement (section 8): a finished line's calls
        # go out the moment it finishes, so the host paints tools under the voice
        # instead of in one burst after it.
        pending = set(steps.values())
        by_task = {task: lid for lid, task in steps.items()}
        stops_read = False
        router_handled = False
        early_started = False

        async def _stop_ahead():
            """A stop must beat the work it cancels to the screen.

            Calls used to be handed over once, after the router had been read, so a
            line the router stopped lost its calls. Streaming must not lose that: the
            first batch waits for the router's decision (never for the voice) and
            applies its stops before anything goes out.
            """
            nonlocal stops_read
            if router is None or stops_read:
                return
            stops_read = True
            try:
                decision = await router or {}
            except Exception:  # noqa: BLE001
                return
            for lid in decision.get("stop") or []:
                line = self.lines.get(str(lid))
                if line is not None and line.status == "active":
                    line.status, line.result = "stopped", line.result or "stopped at their request"
                    await self.book.save(self.pid, line)

        async def _take(finished):
            """Settle finished steps and return the deduped calls they produced."""
            out: list[dict] = []
            for task in finished:
                line = self.lines[by_task[task]]
                kind, value = await task
                line.steps += 1
                out.extend(self.dedupe(await self.settle(line, kind, value)))
            return out

        speak = _speak(self.worker, voice, 400)
        try:
            # One race per beat: the next spoken piece or the next finished line,
            # whichever comes first. Neither waits for the other.
            speaking: asyncio.Future | None = asyncio.ensure_future(anext(speak))
            while speaking is not None or pending or (router is not None and not router_handled):
                racing = set(pending) | ({speaking} if speaking is not None else set())
                if router is not None and not router_handled:
                    racing.add(router)
                done, _still = await asyncio.wait(racing, return_when=asyncio.FIRST_COMPLETED)
                if router is not None and router in done and not router_handled:
                    router_handled = True
                    try:
                        decision = router.result() or {}
                    except Exception:
                        log.exception("router task failed")
                        decision = {}
                    if not decision and self.tr.new_message and not self.lines:
                        # Even with a broken router, a tool-bearing work request
                        # must not silently collapse into normal chat.
                        decision = {"start": [{"topic": _short(self.tr.new_message, 80),
                                               "do": self.tr.new_message}]}
                    started = await self.apply(decision)
                    early_started = bool(started)
                    immediate = []
                    for line, call in started:
                        if call is not None:
                            immediate.append((line, call))
                        else:
                            task = asyncio.create_task(self.step(line))
                            pending.add(task)
                            by_task[task] = line.id
                    if immediate:
                        calls = self.dedupe(immediate)
                        if calls and self.tools:
                            yield "calls", calls
                finished = {t for t in done if t in pending}
                pending -= finished
                batch = await _take(finished) if finished else []
                if batch and self.tools:
                    await _stop_ahead()
                    batch = [c for c in batch if self.lines.get(line_of(c["id"])) is None
                             or self.lines[line_of(c["id"])].status == "active"]
                if batch and self.tools:
                    # Guarded on the host having tools: a voice-only turn must never
                    # put host tool-call events on screen.
                    yield "calls", batch
                if speaking is not None and speaking in done:
                    try:
                        piece = speaking.result()
                    except StopAsyncIteration:
                        speaking = None
                    except Exception:  # noqa: BLE001
                        log.exception("voice failed during lanes round")
                        speaking = None     # the work keeps going; it never waits for speech
                    else:
                        speaking = asyncio.ensure_future(anext(speak))
                        said += piece
                        words = gate.feed(piece)
                        if words:
                            heard += words
                            yield "text", words
        except Exception:  # noqa: BLE001
            log.exception("voice failed during lanes round")
        finally:
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            if speaking is not None:          # teardown only: the voice's next piece was never asked for
                if not speaking.done():
                    speaking.cancel()
                try:
                    await speaking
                except BaseException:  # noqa: BLE001
                    pass
            try:
                await speak.aclose()
            except BaseException:  # noqa: BLE001
                pass
        tail = gate.flush()
        if tail:
            heard += tail
            yield "text", tail
        if heard.strip():
            # Only mark them told once words about them actually reached the person;
            # a sentence the gate dropped was never heard.
            for line in finished_before:
                line.told = True
                await self.book.save(self.pid, line)
        decision = {} if early_started else (await router if router is not None else {})
        if decision is None:
            # The router gave no usable decision. A new request with tools still
            # gets one line, so the work is never silently dropped.
            decision = {}
            if (self.tr.new_message and not self._emitted
                    and not any(line.status == "active" for line in self.lines.values())):
                decision = {"start": [{"topic": _short(self.tr.new_message, 80), "do": self.tr.new_message}]}
        if (not early_started and not decision.get("start") and _PROMISE.search(said) and not self._emitted
                and not any(line.status == "active" for line in self.lines.values()) and self.tools):
            # The voice said it would do something; make sure something is doing it.
            decision = await self.route("The voice just told them: " + _short(said, 600)) or {}
        started = [] if early_started else await self.apply(decision)
        first_steps = {line.id: asyncio.create_task(self.step(line)) for line, call in started if call is None}
        late: list[tuple[Line, dict]] = []
        for line, call in started:
            if call is not None:
                late.append((line, call))
        for lid, task in first_steps.items():
            line = self.lines[lid]
            kind, value = await task
            line.steps += 1
            late.extend(await self.settle(line, kind, value))
        out = self.dedupe(late)   # also records into self._emitted
        if not self._emitted:
            ended = [line for line in self.lines.values() if line.status != "active" and not line.told]
            if ended:
                record = "\n".join(f"- {line.topic} ({line.status}): {line.result}" for line in ended)
                closing = [{"role": "system", "content": self.system + "\n\n" + VOICE_END},
                           {"role": "user", "content": record[:12000]}]
                lead = " " if heard and not heard.endswith((" ", "\n")) else ""
                recap = ""
                try:
                    async for piece in _speak(self.worker, closing, 900):
                        recap += piece
                except Exception:  # noqa: BLE001
                    log.exception("closing words failed")
                # The voice may have already covered it while the work ran; a
                # second telling of the same result reads as a stutter.
                from .kernel.dual import echoes
                if recap.strip() and not echoes(heard, recap):
                    yield "text", lead + recap
                for line in ended:
                    line.told = True
                    await self.book.save(self.pid, line)
        self.trace["calls"] = len(self._emitted)
        for line, target in self.collisions:
            note = f"{target} is being changed by another line right now; do not write it, read it instead"
            if note not in line.notes:
                line.notes.append(note)
                await self.book.save(self.pid, line)
        if out:
            # The last batch (the router's new lines). Lines that finished while the
            # voice spoke already went out; the host keeps the index across batches.
            yield "calls", out

    async def settle(self, line: Line, kind: str, value, depth: int = 0) -> list[tuple[Line, dict]]:
        """Apply one step's outcome. A fan-out starts every piece on its own worker
        at once; the last piece to finish hands the line its next step."""
        out: list[tuple[Line, dict]] = []
        if kind == "calls":
            out = [(line, c) for c in value]
            mark = lane_mark(line)
            if any(mark in json.dumps(c.get("arguments")) for c in value):
                line.delegated = True
                self.trace["delegated_to_host_agents"] = self.trace.get("delegated_to_host_agents", 0) + 1
        elif kind == "fan":
            line.waiting = True
            kids: list[Line] = []
            for piece in value:
                child = await self.book.start(self.pid, piece["topic"], piece["do"])
                child.parent, child.told = line.id, True
                line.children.append(child.id)
                self.lines[child.id] = child
                kids.append(child)
            self.trace["fanned_out"] = self.trace.get("fanned_out", 0) + len(kids)
            await self.book.save(self.pid, line)
            results = await asyncio.gather(*(self.step(k) for k in kids))
            for kid, (k_kind, k_value) in zip(kids, results):
                kid.steps += 1
                out.extend(await self.settle(kid, k_kind, k_value, depth + 1))
            return out
        else:
            line.status, line.result = ("done" if kind == "done" else "failed"), str(value)[:4000]
            self.trace["lines_finished"] += 1
        await self.book.save(self.pid, line)
        if line.status != "active" and line.parent and depth < 6:
            out.extend(await self.join(line.parent, depth))
        return out

    async def join(self, parent_id: str, depth: int = 0) -> list[tuple[Line, dict]]:
        parent = self.lines.get(parent_id)
        if (parent is None or parent.status != "active" or not parent.waiting
                or any(self.lines.get(c) is None or self.lines[c].status == "active" for c in parent.children)):
            return []
        parent.waiting = False
        parent.notes.append("Your parallel pieces reported back: " + " | ".join(
            f"{self.lines[c].topic} ({self.lines[c].status}): {_short(self.lines[c].result, 600)}"
            for c in parent.children))
        parent.children = []
        kind, value = await self.step(parent)
        parent.steps += 1
        return await self.settle(parent, kind, value, depth + 1)

    def dedupe(self, calls: list[tuple[Line, dict]]) -> list[dict]:
        """One call per action, and one line per thing being changed.

        Lines run at the same time, so two of them must not act on the same file,
        directory or page at once: the second would overwrite the first's work.
        This is checked from the arguments, never from a model's assurance.
        """
        done = _completed_actions(self.messages)
        active = {line.id for line in self.lines.values() if line.status == "active"}
        owners: dict[str, str] = {}
        for cid, (name, arguments) in self.tr.calls.items():
            other = line_of(cid)
            if other and other in active:
                for target in write_targets(name, arguments):
                    owners.setdefault(target, other)
        # Batches go out one at a time while the voice speaks, so ownership is
        # rebuilt from what this round already emitted, not from a single shared pass.
        for emitted in self._emitted:
            for target in write_targets(emitted["name"], emitted["arguments"]):
                owners.setdefault(target, line_of(emitted["id"]))
        out: list[dict] = []
        self.collisions = []
        for line, call in calls:
            if line.status != "active":
                continue
            key = _canonical(call["name"], call["arguments"])
            if key in done or key in self._seen:
                self.trace["duplicates_refused"] += 1
                continue
            mine = write_targets(call["name"], call["arguments"])
            clash = next((t for t in mine if owners.get(t, line.id) != line.id), "")
            if clash:
                self.trace["duplicates_refused"] += 1
                self.collisions.append((line, clash))
                continue
            for target in mine:
                owners[target] = line.id
            self._seen.add(key)
            out.append({"id": _call_id(line), "name": call["name"], "arguments": call["arguments"],
                        "line": line.label, "topic": line.topic})
        # Remember what this round already put in front of the host: the next batch
        # must not repeat it, nor write something the earlier batch already owns.
        self._emitted.extend(out)
        return out


async def run_lanes(mind, pid: int, worker, system: str, host_system: str,
                    messages: list[dict], tools: list[dict], trace: dict,
                    own_tools: list[dict] | None = None, run_own=None):
    """One reply to any host: the voice's text as it comes, then every line's next calls."""
    started = time.monotonic()
    rnd = Round(mind, pid, worker, system, host_system, messages, tools, own_tools, run_own)
    first = None
    async for kind, value in rnd.run():
        if kind == "text":
            if first is None:
                first = time.monotonic() - started
            yield {"type": "delta", "text": value}
        else:
            if mind.shared_timeline is not None:
                for c in value:
                    await mind.shared_timeline.append(
                        pid, producer=c["line"], kind="tool_call", task_id=line_of(c["id"]),
                        payload={"tool": c["name"], "args": _short(c["arguments"], 1500),
                                 "call_id": c["id"], "part": c["topic"]})
            yield {"type": "calls", "calls": [{k: c[k] for k in ("id", "name", "arguments")} for c in value]}
    trace.update(rnd.trace, first_text_s=round(first, 3) if first is not None else None,
                 wall_s=round(time.monotonic() - started, 3),
                 active_lines=sum(1 for line in rnd.lines.values() if line.status == "active"))
