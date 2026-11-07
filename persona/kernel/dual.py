"""The dual loop.

One voice, two writings, one model, both in flight for the whole answer.

The lobe on the mouth answers what was asked and is the only one that presses
a tool. The other lobe is already drafting what that answer is leaving out:
a memory that bears on it, the wider context, the other way of seeing the same
facts. The mouth lets the second half out at a sentence boundary. If the quiet
lobe has nothing to add, it says nothing.

Why this is faster and not merely more elaborate: today the mind does memory,
feeling and verification *before* it can speak, so the costs add up.

    serial    T = t_memory + t_feel + t_speak + t_verify
    dual      T = max(t_speak, t_thought + t_io)

The extra thinking hides inside speech that was going to take that time anyway.

Two rules hold the whole thing together:

1. **The message prefix is byte-identical across a handoff.** Only the writing's
   tail is appended, and only as the final message. Break this and the provider's
   prefix cache dies on every swap, which turns a latency win into a latency
   loss.
2. **A beat ends on a sentence.** The seam is a sentence boundary, so it is
   invisible. Stitching mid-clause is what makes a two-model system sound like
   two models.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

from ..llm import LLM, ChatResult, ToolCall
from .caps import CapabilityBundle, NullCapabilities, VerificationResult
from .floor import starting_lobe
from .parts import PARTS_HINT, run_parts, split_parts
from .screen_plan import ScreenPlan, marked_done, parse_next_action, parse_plan
from .screen_prove import latest_screen_text, prove_or_wait, screen_proves
from .writings import HANDOFF_PROTOCOL, Writing, writings as default_writings

# Private channel. Same trick the mind already uses for its feel marker - the
# model writes to itself and the text never reaches the person.
# Models close the bracket either way (or just end the line).
HOLD_RE = re.compile(r"\u27e6\s*hold\s*:\s*([^\u27e6\u27e7\u27e9\n]+?)\s*(?:[\u27e7\u27e9\u27e6\n]|$)",
                     re.IGNORECASE)

# A beat must clear this to be worth handing over. Below it the pass produced a
# filler syllable, and switching would only add a seam.
MIN_BEAT_CHARS = 80
FIRST_THOUGHT_CHARS = 16
BRIDGE_AFTER_S = 0.7
# Zero-gap hand-off (README.md): while the other lobe has no
# words ready, the speaker keeps the last STRETCH_CHARS of its segment back and
# lets them out a word every DRIP_S, so the floor never goes quiet.
STRETCH_CHARS = 90
FIRST_SHOW = 40
DRIP_S = 0.12
# What a lobe says when the floor reaches it before its thought is ready.
BRIDGES = {"L": ("Right — ", "Okay — ", "So — ", "Yeah — "),
           "R": ("Hmm — ", "Well — ", "And — ", "Mm — ")}
# The lobe's own opener after a bridge ("Hmm — Wait — ...") is dropped, so the
# bridge never stacks with a second hesitation.
_INTERJECTION = re.compile(
    r"^(?:\s*(?:wait|actually|hmm+|mm+|well|okay|ok|right|so|yeah|and|oh|um+|uh+)\b"
    r"\s*(?:[—–-]+|,|…|\.{3})\s*)+", re.IGNORECASE)
_OPENER_CHARS = 12


_TERSE = re.compile(
    r"\b(?:only|just)\b[^.?!]{0,24}\b(?:number|digits?|word|answer|yes|no|name|value|result)\b"
    r"|\bone[- ]word\b|\byes or no\b|\bno explanation\b", re.I)


def asks_for_terse(text: str) -> bool:
    """The person asked for a bare answer; a bridge would only clutter it."""
    return bool(_TERSE.search(text or ""))


_ACTION = re.compile(
    r"\b(?:look(?:\s+it)?\s+up|check|find|search|fetch|open|get\s+me|download|install|run|send|"
    r"book|buy|scan|list|show\s+me|what(?:'s|\s+is)\s+the\s+(?:weather|temperature|price|time|news)|"
    r"do\s+(?:we|i|you)\s+have|is\s+there|are\s+there)\b", re.I)
# Where the live part of an acting beat ends: a finished sentence or a dash.
_FIRST_CLAUSE = re.compile(r"[.!?…](?=\s)|\s[—–]\s")


def asks_for_action(text: str) -> bool:
    """The person asked her to do or look something up, so a tool is likely."""
    return bool(_ACTION.search(text or ""))


def drop_dangling_bridge(text: str) -> str:
    """A bridge that nothing followed is taken off the end of the reply."""
    t = text.rstrip()
    for b in (*BRIDGES["L"], *BRIDGES["R"]):
        if t.endswith(b.strip()):
            return t[: -len(b.strip())].rstrip()
    return text


_STARTERS = frozenset("the so it it's this that there they we you and but a an one what yes no well "
                      "now right okay let let's here his her its".split())


def _after_bridge(text: str) -> str:
    """"Hmm - The wind" reads as two sentences; "Hmm - the wind" is one voice."""
    m = re.match(r"([A-Za-z']+)", text)
    if m and m.group(1).lower() in _STARTERS and text[:1].isupper():
        return text[:1].lower() + text[1:]
    return text


def strip_interjection(text: str) -> str:
    """"Wait — actually, it's Friday" after a bridge is just "it's Friday"."""
    return _INTERJECTION.sub("", text or "", count=1).lstrip()


def _enough_to_react(said: str) -> bool:
    """The other lobe starts once there is a whole thought to react to: one
    finished sentence, or enough words that a cut would not land mid-word."""
    said = said.strip()
    if len(said) >= MIN_BEAT_CHARS:
        return True
    return len(said) >= FIRST_THOUGHT_CHARS and re.search(r"[.!?](\s|$)", said) is not None

# Steer mode: what the expensive model is told. It never speaks to the person;
# it only points the cheap model at the right thing, in as few words as possible.
STEER_PROTOCOL = """

[you are the director, not the speaker]
A cheaper pass of this same mind does all the talking and all the tool work.
You only correct its course. Reply with ONE short directive (max 25 words):
what to do or say next, what to avoid, which tool/argument is right, or the fact
it is about to get wrong. No preamble, no explanation. If its course is already
right, reply: none"""

# Lifted onto STEER_PROTOCOL for the one call that must also carry a plan. The
# director stays terse about the directive; the plan is a structure the kernel
# parses, so it is exempt from the word limit rather than crushed by it.
PLAN_EXEMPTION = """

[one exception to the word limit, this turn only]
You were also asked for a subtasks plan. The plan is not a directive and does
not count against the 25 words: write the JSON line in full, then the directive."""

# Extra room for that plan: six goal lines plus the JSON frame.
_PLAN_TOKENS = 240


class HoldFilter:
    """Streaming filter: text passes through, the private hold suffix is swallowed.

    The marker has to be caught *while* streaming. Stripping it afterwards is
    useless - by then the characters are already on the wire, and the reply
    cannot un-send them (the emitter is monotonic on purpose, so a person never
    watches text disappear). Same shape as the mind's existing feel filter.
    """

    OPEN = "\u27e6hold:"
    CLOSE = "\u27e9"
    # Every private suffix a host's prompt may ask for. The companion's A writes
    # a feel suffix; it must never reach the person either.
    OPENS = ("\u27e6hold:", "\u27e6feel:", "[[feel:", "[feel:", "\u27e6done", "\u27e6subtask",
             "\u27e6handoff:", "\u27e6stay", "\u27e6pass", "\u27e6task")

    # The speaker's dense plan for the other lobe; stripped wherever it sits.
    PLAN = "\u27e6plan:"
    FLOOR = "\u27e6floor:"
    TODO = "\u27e6todo:"

    def __init__(self):
        self.buf = ""
        self.captured = ""
        self.hidden = False
        self.plan = ""
        self.floor = ""
        self.todo = ""

    def feed(self, piece: str) -> str:
        if self.hidden:
            self.captured += piece
            return ""
        self.buf += piece
        while True:
            headers = [(p, marker) for marker in (self.PLAN, self.FLOOR, self.TODO)
                       if (p := self.buf.find(marker)) != -1]
            if not headers:
                break
            g, marker = min(headers)
            # A header inside a private suffix is not public inline metadata.
            if any(0 <= self.buf.find(o) < g for o in self.OPENS):
                break
            # Models close the bracket either way (or just end the line).
            ends = [e for e in (self.buf.find(c, g) for c in ("\u27e7", self.CLOSE, "\n")) if e != -1]
            end = min(ends) if ends else -1
            if marker == self.TODO and end != -1 and not self.buf[g + len(marker):end].strip():
                # A list written on the lines below its header runs to the bracket.
                closes = [e for e in (self.buf.find(c, g) for c in ("\u27e7", self.CLOSE)) if e != -1]
                if closes and min(closes) - g <= 600:
                    end = min(closes)
                elif len(self.buf) - g <= 600:
                    end = -1
            if end == -1:
                out, self.buf = self.buf[:g], self.buf[g:]
                return out
            value = self.buf[g + len(marker):end].strip()
            if marker == self.PLAN:
                self.plan = self.plan or value
            elif marker == self.TODO:
                self.todo = value
            elif value in ("stay", "pass", "end"):
                self.floor = value
            self.buf = self.buf[:g] + self.buf[end + 1:]
        hits = [p for p in (self.buf.find(o) for o in self.OPENS) if p != -1]
        pos = min(hits) if hits else -1
        if pos != -1:
            out = self.buf[:pos]
            self.captured = self.buf[pos:]
            self.hidden = True
            self.buf = ""
            return out
        # Hold back any tail that could still turn out to be the start of a marker.
        keep = 0
        for o in (*self.OPENS, self.PLAN, self.FLOOR, self.TODO):
            for k in range(min(len(o) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(o[:k]):
                    keep = max(keep, k)
                    break
        out = self.buf[:len(self.buf) - keep] if keep else self.buf
        self.buf = self.buf[len(out):]
        return out

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        if any(out.startswith(o) or o.startswith(out) for o in (*self.OPENS, self.PLAN, self.FLOOR, self.TODO)):
            return ""
        return out

    def handoff(self) -> str:
        if m := re.search(r"⟦handoff:\s*([^⟧⟩\n]*)", self.captured):
            return m.group(1).strip()[:200]
        return ""

    def task(self) -> str:
        found = re.findall(r"⟦task:\s*(done|ask)", self.captured, re.I)
        return found[-1].lower() if found else ""

    def intent(self) -> str:
        if "⟦handoff:" in self.captured:
            return "pass"
        suffix = re.findall(r"⟦(?:floor:\s*(stay|pass|end)|(stay|pass))\s*[⟧⟩]", self.captured)
        return next((a or b for a, b in reversed(suffix)), self.floor)

    def hold(self) -> str:
        # Parse with the regex so a sloppy marker (odd spacing, no close bracket,
        # "Hold :" instead of "hold:") still yields the thought rather than junk.
        if m := HOLD_RE.search(self.captured):
            return m.group(1).strip()
        if self.OPEN not in self.captured:
            return ""
        body = self.captured.split(self.OPEN, 1)[-1]
        return re.split(r"[\u27e6\u27e7\u27e9\n]", body, maxsplit=1)[0].strip()

_TERMINAL = ".?!…"
# A dot after one of these is not the end of a sentence. "Mr. Smith left."
# is one sentence; the cap must not throw away everything after "Mr."
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc"}
_ABBREV_TAIL = ("e.g", "i.e", "a.m", "p.m")


def _abbrev_dot(piece: str, i: int) -> bool:
    tail = piece[max(0, i - 3):i].lower()
    if tail in _ABBREV_TAIL:
        return True
    k = i
    while k > 0 and piece[k - 1].isalpha():
        k -= 1
    return piece[k:i].lower() in _ABBREV


def _split_image(result: str, *, see: bool) -> tuple[str, str]:
    """A tool result carrying a screenshot (JSON with ``image_b64``): the base64
    never goes into text (it would flood the context). Returns (text, image)."""
    if '"image_b64"' not in result:
        return result, ""
    try:
        data = json.loads(result)
    except Exception:  # noqa: BLE001
        return result, ""
    img = str(data.pop("image_b64", "") or "")
    data["screenshot"] = ("attached below" if see else
                          "taken, but this model cannot see images - use screen_text to read the screen")
    return json.dumps(data), (img if see else "")


QUIET_RECOVERY = ("\n{names} did not work. Do not apologize, do not say it failed, was a mistake "
                  "or that you will try again - just take the next path (another element, another "
                  "page, another way) and keep talking about the task itself.")

# A screen result buys the next beat. Capped so a browse cannot run away.
_PAGE_BUDGET_CAP = float('inf')
_STATUS_NUDGES = 3
# Persistent task mode: a turn that started work keeps going until a lobe
# marks it done or asks the person. Bounded so a stuck task cannot run away.
_TASK_NUDGES = float('inf')

TASK_PROTOCOL = """

[working on their task]
Keep going with tools until all of it is done.
- Your private plan line is the to-do list, one item per part they asked for,
  updated every step: ⟦plan: [x] done; [ ] next; [ ] what remains⟧
- If a step fails, try another way. Never invent a result you did not see.
- Say each new result once, as it comes in; do not repeat earlier ones.
- Done means every part they asked for has a tool result showing it. Then give
  the answer once, then ⟦task: done⟧.
- Stop only for a choice only they can make, a login, or approval to pay or
  send: ask once, then ⟦task: ask⟧."""

KEEP_GOING = ("The task is not finished and you did not ask them anything. Take the next "
              "step with a tool now. If all of it is done and the result was already said, "
              "write only ⟦task: done⟧; do not say it again.")

# A long task trims what it carries: once the shared history passes this many
# characters, tool results older than the newest few shrink to a short head.
_TRIM_AT = float('inf')
_TRIM_KEEP = 6
_TRIM_HEAD = 600


def parse_todo(value: str) -> list[dict]:
    """``[x] done; [ ] next`` -> [{"text", "done"}]. A bare item is open."""
    items = []
    for part in re.split(r"\s*(?:;|\n)\s*", value or ""):
        part = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", part.strip())
        m = re.match(r"^(?:\[\s*([xX✓✔ ]?)\s*\]|([✓✔]))?\s*(.+)$", part)
        if m and m.group(3).strip():
            items.append({"text": m.group(3).strip()[:160],
                          "done": bool((m.group(1) or "").strip() or m.group(2))})
    return items[:20]


# The kernel's own to-do list, for a model that never writes checkboxes: the
# parts of the request (or of the first plan line), ticked off by tool results.
_PART_SPLIT = re.compile(
    r"\s*(?:[;\n•]|→|->|=>|(?<=[.!?])\s+(?=[A-Z])|(?:^|\s)\(?\d+[.)]\s+|,?\s+(?:and\s+)?then\s+"
    r"|,\s*(?:and\s+)?(?:after that|finally|also|plus)\s+|,\s*and\s+|,)\s*", re.I)
_PART_LEAD = re.compile(r"^(?:(?:and|then|also|finally|first|next|please|now|so|"
                        r"can you|could you|would you|will you)\b[\s,]*)+", re.I)
_PART_STOP = frozenset((
    "the a an and or of to for on in at by with from into me my you your it its this that these those "
    "each every all both one ones some any what which who how now right just then also please can could "
    "would will be is are was were do does get give tell find open read look show make take go see check "
    "pull up out about most more top new first last sentence sentences summary summarize summarise them "
    "their they there here hi hey hello thanks thank ok okay so two three four five six seven eight nine "
    "ten matters matter").split())
_SAY = re.compile(r"\b(summar\w*|tell|explain|give|compare|decide|which|recommend|write|list|say|answer|"
                  r"describe|rank|pick|choose|report|conclude)\b", re.I)
_EACH = re.compile(r"\b(each|every|all of them|both)\b", re.I)
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}


def task_parts(text: str) -> list[str]:
    """The separately checkable parts of a request or a plan line."""
    parts = []
    for raw in _PART_SPLIT.split(text or ""):
        part = _PART_LEAD.sub("", (raw or "").strip(" .!?-*")).strip(" .!?")
        if part and (_part_tokens(part) or _SAY.search(part) or _EACH.search(part)):
            parts.append(part[:1].upper() + part[1:120])
    return parts[:8]


def _part_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (text or "").lower())
            if w not in _PART_STOP and not w.isdigit()]


def _part_count(text: str) -> int:
    """How many of the thing the request names (top 3, three stories, both)."""
    low = (text or "").lower()
    if re.search(r"\bboth\b", low):
        return 2
    for m in re.finditer(r"\b(\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\b", low):
        g = m.group(1)
        n = int(g) if g.isdigit() else _COUNT_WORDS[g]
        if 2 <= n <= 10:
            return n
    return 0


def tool_checklist(parts: list[str], calls: list[dict], *, request: str = "",
                   finished: bool = False) -> list[dict]:
    """``parts`` ticked off by this turn's tool results.

    A part is done when a successful result covers most of its words, a part
    about each of N things when N different pages were opened, and a part that
    is said rather than fetched (summarize, tell, pick) when the work ends.
    """
    # Each result belongs to a page: its own url, the link it clicked, or the
    # last page opened before it.
    seen: list[tuple[str, str]] = []
    page = ""
    for c in calls:
        args = c.get("args") or {}
        clicked = str(c.get("name", "")).endswith("click") and not _failed(c.get("result"))
        page = str(args.get("url") or "") or (
            f"click:{json.dumps(args, sort_keys=True)}" if clicked else "") or page
        if not _failed(c.get("result")):
            seen.append((page, f"{c.get('name', '')} {json.dumps(args, ensure_ascii=False)} "
                               f"{_result_text(c.get('result'), 3000)}".lower()))
    used: set[str] = set()
    want = _part_count(request)
    items = []
    for part in parts:
        words = _part_tokens(part)
        opens = bool(re.match(r"^(?:open|read|visit|fetch|go|look|check)\b", part, re.I))
        says = bool(_SAY.search(part)) and not opens and not re.match(r"^(?:find|search)\b", part, re.I)
        count = (_part_count(part) if opens else 0) or (want if _EACH.search(part) else 0)
        text, done = part, False
        if says:
            done = finished
        elif count:
            opened = list(dict.fromkeys(p for p, _ in seen if p and p not in used))
            text, done = f"{part} ({min(len(opened), count)}/{count})", len(opened) >= count
            if done:
                used.update(opened)
        else:
            need = max(1, (len(words) + 1) // 2)
            hit = next((p for p, s in seen if (not p or p not in used)
                        and (not words or sum(w in s for w in words) >= need)), None)
            if hit is not None:
                done = True
                if hit:
                    used.add(hit)
        items.append({"text": text, "done": done})
    return items


def tool_evidence(calls: list[dict], *, last: int = 6, each: int = 1500, total: int = 6000) -> str:
    """The turn's newest tool results, for a check of what was said about them."""
    lines = [f"{c.get('name', '')}({json.dumps(c.get('args') or {}, ensure_ascii=False)[:200]}) -> "
             f"{_result_text(c.get('result'), each)}" for c in calls[-last:]]
    while len(lines) > 1 and sum(len(x) + 2 for x in lines) > total:
        lines.pop(0)
    return "\n\n".join(lines)[:total]


_HTML_DROP = re.compile(r"<(script|style|noscript|svg|template)\b.*?</\1\s*>", re.I | re.S)


def _result_text(result, limit: int) -> str:
    """A result's readable text. A fetched page arrives as raw HTML, whose
    first 1,500 characters are head tags - live, C saw no article at all."""
    text = str(result or "")
    if not re.search(r"<(?:html|body|head|div|p|meta|!doctype)\b", text[:4000], re.I):
        return text[:limit]
    import html as _html
    text = text[:80_000].replace("\\n", " ").replace("\\t", " ").replace('\\"', '"')
    text = re.sub(r"<[^>]*>", " ", _HTML_DROP.sub(" ", text))
    return " ".join(_html.unescape(text).split())[:limit]


def _takes(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


def trim_history(prefix: list[dict]) -> int:
    """Shrink old tool results in place; returns how many were trimmed.

    Only content changes, so every tool call keeps its answer and the
    tool-call/result pairing a strict provider checks stays valid.
    """
    def size(m: dict) -> int:
        c = m.get("content")
        return len(c) if isinstance(c, str) else len(json.dumps(c)) if c else 0

    if sum(size(m) for m in prefix) <= _TRIM_AT:
        return 0
    results = [i for i, m in enumerate(prefix)
               if m.get("role") == "tool" or (m.get("role") == "user" and (
                   isinstance(m.get("content"), list)
                   or str(m.get("content") or "").startswith("[the screen right after that")))]
    trimmed = 0
    for i in results[:-_TRIM_KEEP]:
        m = prefix[i]
        c = m.get("content")
        if isinstance(c, list):
            prefix[i] = {**m, "content": "[an earlier screenshot, trimmed from a long task]"}
            trimmed += 1
        elif isinstance(c, str) and len(c) > _TRIM_HEAD + 40:
            prefix[i] = {**m, "content": c[:_TRIM_HEAD] + " ... [older result trimmed in a long task]"}
            trimmed += 1
    return trimmed


PEER_REVIEW = ("Before finishing, go through what they asked part by part (every item, every "
               "page they asked to open) and find the tool result above that shows each was done; "
               "a fact with no result behind it does not count. Do any part without a result now "
               "with a tool. When every part has one and the result was already said, write only "
               "⟦task: done⟧; if it was not said yet, say it once, then ⟦task: done⟧.")
# A sentence whose only job is to say the hands are in motion. A year, a URL,
# or a fact after it is an answer; this is not.
_STATUS_SENTENCE = re.compile(
    r"^(?:i(?:'ll| will| am|'m)?\s+)?(?:just\s+|still\s+)?"
    r"(?:pulling (?:up|it up)|opening|reading|checking|looking(?: at)?|loading|"
    r"navigating|bringing up|start(?:ing)? on|let me (?:open|read|check|look(?: at)?))"
    r"(?:\s+\S+){0,8}[.!?]?\s*$",
    re.IGNORECASE,
)


ADJUST_SCREEN = frozenset({
    "open_url", "screen_text", "screenshot", "click", "type_text",
    "press_keys", "scroll", "open_app", "focus_app", "wait",
})
# The ones that OBSERVE rather than act. An action's confirmation ("clicked
# button 'Bluetooth'") is not a reading of the screen, and the proof gate must
# not mistake one for the other.
READS_SCREEN = frozenset({"screen_text", "screenshot", "snapshot", "open_url", "wait"})


def _screen_tool(name: str) -> bool:
    """A hand that changes or reads a page or desktop. A shell command is not one.

    Three channels reach a screen. The server's own browser (``browser_``) and
    the in-process desktop hand (``desktop_``) are screens by their whole
    prefix. The hands runner on the person's own computer and in her cloud
    sandbox (``computer_``/``cloud_``) is different: its screen tools sit
    alongside its shell and file tools, so they are named one by one. Computer
    use lives on that runner - missing three of its names here meant a turn
    that opened an app or scrolled did not count as a screen turn at all.
    """
    text = name or ""
    if text.startswith(("browser_", "desktop_")):
        return True
    for place in ("computer_", "cloud_"):
        if text.startswith(place):
            return text[len(place):] in ADJUST_SCREEN
    return False


def _reads_screen(name: str) -> bool:
    """A hand whose RESULT is an observation of the screen, not a receipt."""
    text = name or ""
    for place in ("browser_", "desktop_", "computer_", "cloud_"):
        if text.startswith(place):
            return text[len(place):] in READS_SCREEN
    return False


# C11. Tools that only look something up can run side by side. Anything that
# touches a screen, a file or the outside world runs alone and in order.
_PARALLEL_SAFE = re.compile(
    r"(^|_)(read|list|search|get|fetch|find|lookup|query|recall|weather|news|calc|calculator|"
    r"time|define|translate|web)(_|$)")
_KNOWN_PLACES = ("computer_", "cloud_", "host_", "desktop_", "browser_")


def _base_name(name: str) -> str:
    for place in _KNOWN_PLACES:
        if name.startswith(place):
            return name[len(place):]
    return name


def _parallel_safe(name: str) -> bool:
    base = _base_name(name or "")
    if _screen_tool(name) or base.startswith(("write", "edit", "delete", "remove", "move", "send", "post")):
        return False
    return bool(_PARALLEL_SAFE.search(base))


def _hand_groups(tool_calls: list) -> list[list]:
    """Consecutive look-ups share a group (run at once); every other call is alone."""
    groups: list[list] = []
    for tc in tool_calls:
        if groups and _parallel_safe(tc.name) and all(_parallel_safe(x.name) for x in groups[-1]):
            groups[-1].append(tc)
        else:
            groups.append([tc])
    return groups


def _reground_name(tool_calls: list, available: set[str]) -> str:
    """The reading to take after a screen action that was not already followed by one."""
    last = None
    for tc in tool_calls:
        if _screen_tool(tc.name):
            last = tc.name
    if last is None or _reads_screen(last):
        return ""
    # A browser navigation/click is only an action receipt. Observe the real
    # DOM afterward, just as we do for desktop clicks, before making another
    # decision or accepting a subtask as completed.
    if last.startswith("browser_") and "browser_snapshot" in available:
        return "browser_snapshot"
    for place in ("computer_", "cloud_", "desktop_"):
        if last.startswith(place):
            for cand in (place + "screen_text", place + "snapshot"):
                if cand in available:
                    return cand
    return ""


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _last_sentence(spoken: str) -> str:
    """The sentence she was in when they cut in (A19)."""
    parts = [x.strip() for x in _SENTENCE_END.split((spoken or "")[-400:]) if x.strip()]
    return parts[-1][:240] if parts else ""


_EXPECT = re.compile(r"^\s*EXPECT:\s*(.+)$", re.I | re.M)


def _expectation(ready: str) -> str:
    """What the silent side predicted the tool would bring back (D9)."""
    m = _EXPECT.search(ready or "")
    return m.group(1).strip()[:160] if m else ""


def _status_only(text: str) -> bool:
    """True when the beat only says she is opening or reading.

    That sentence is not the answer. The page came back so she can take the
    next step or say what the page shows. A recited heading, a year, or a
    second clause is an answer and does end the beat.
    """
    raw = " ".join((text or "").split())
    if not raw or len(raw) > 160 or "," in raw or ";" in raw:
        return False
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if not parts or len(parts) > 2:
        return False
    return all(_STATUS_SENTENCE.match(p) for p in parts)


def _failed(result) -> bool:
    """A hand that came back with an error (JSON ``error`` or a failure line)."""
    text = str(result or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return bool(obj.get("error")) or obj.get("ok") is False
    low = text[:200].lower()
    return low.startswith(("error", "the ")) and (" failed" in low or "error" in low[:20])


def _sentence_end(piece: str, *, allow_final: bool = True) -> int:
    """Index just past the first sentence end in ``piece``, or -1.

    A dot inside ``flights.example`` is not an end. A terminator that is still
    the last character is not an end either unless ``allow_final``: the next
    chunk may be the rest of a URL.
    """
    for i, ch in enumerate(piece):
        if ch not in _TERMINAL:
            continue
        j = i + 1
        while j < len(piece) and piece[j] in "\"')\u201d\u2019":
            j += 1
        if j < len(piece) and not piece[j].isspace():
            continue
        if ch == "." and _abbrev_dot(piece, i):
            continue
        if j == len(piece) and not allow_final:
            return -1
        return j
    return -1


def _take_sentences(text: str, n: int, *, allow_final: bool = True) -> tuple[str, bool]:
    """Text through the nth finished sentence, and whether that many were reached."""
    if n <= 0:
        return text, False
    end = 0
    found = 0
    while found < n and end < len(text):
        i = _sentence_end(text[end:], allow_final=allow_final)
        if i == -1:
            return text, False
        end += i
        found += 1
    return text[:end], True


_FILLER = frozenset(
    "a an the to of and or for in on at it is was be am are you your i we they "
    "this that with from as so if but not just my me our".split()
)


def restates(already: str, draft: str) -> bool:
    """The quiet pass repeated the sentence already spoken, so it adds nothing.

    A real addition shares a name and then says the thing that was left out.
    A restatement is the same words again. That pass should have been a private
    done marker; when the model writes the sentence anyway, it is not spoken.
    """
    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower())
                if w not in _FILLER and len(w) > 1]

    said_nums = set(re.findall(r"\d+(?:\.\d+)?", already or ""))
    new_nums = re.findall(r"\d+(?:\.\d+)?", draft or "")
    if len(new_nums) >= 3 and sum(n in said_nums for n in new_nums) / len(new_nums) >= 0.7:
        return True  # the same readings again, in other words
    if reasks(already, draft):
        return True
    spoken = set(words(already))
    nxt = words(draft)
    if len(nxt) < 3 or not spoken:
        return False
    covered = sum(1 for w in nxt if w in spoken)
    if covered / len(nxt) >= 0.75:
        return True
    # The same list again with a new sentence wrapped round it.
    parts = [p for p in re.split(r"(?<=[.!?;:])\s+|,\s+|\s+(?:and|or)\s+", draft or "") if len(words(p)) >= 2]
    if len(parts) < 3:
        return False
    repeated = sum(1 for p in parts
                   if sum(1 for w in words(p) if w in spoken) / len(words(p)) >= 0.75)
    return repeated / len(parts) >= 0.6


_ASK_FILLER = _FILLER | frozenset(
    "what which how why when where who do does did should would could can will "
    "want like think mean".split())


def _questions(text: str) -> list[set[str]]:
    """Each clause of each question, as content words: "which option, and
    which machine?" asks two things."""
    clauses = [c for q in re.findall(r"[^.?!]*\?", text or "")
               for c in re.split(r"[,;:—–]|\s+and\s+", q)]
    return [{w for w in re.findall(r"[a-z0-9']+", c.lower()) if w not in _ASK_FILLER and len(w) > 1}
            for c in clauses]


def reasks(already: str, draft: str) -> bool:
    """The draft asks a question the reply has already asked."""
    asked = [q for q in _questions(already) if q]
    for q in _questions(draft):
        for p in asked:
            small = min(len(q), len(p))
            if small >= 2 and len(q & p) / small >= 0.6:
                return True
    return False


_RECAP_WORDS = re.compile(
    r"\b(?:done|delivered|confirmed|answered|covered|both|answers?|parts?|all|it|that'?s|"
    r"nothing|else|more|missing|left|to|add|asked|ask|two|three|from|the|and)\b")


def recaps(already: str, draft: str) -> bool:
    """The draft's first sentence only announces that what was said is done
    ("391 and Al-Baqarah - done.", "Both answers delivered."): no new fact."""
    first = re.split(r"(?<=[.!?])\s+", (draft or "").strip(), maxsplit=1)[0]
    low = first.lower()
    if not (already or "").strip() or len(low.split()) > 14:
        return False
    if not re.search(r"\b(?:done|delivered|confirmed|answered|covered|nothing (?:else |more )?"
                     r"(?:missing|left|to add)|that'?s (?:it|all))\b", low):
        return False
    said = set(re.findall(r"[a-z0-9]+", (already or "").lower()))
    rest = [w for w in re.findall(r"[a-z0-9]+", _RECAP_WORDS.sub(" ", low)) if w not in _FILLER]
    return sum(w in said for w in rest) >= 0.75 * len(rest) if rest else True


def says_again(already: str, draft: str) -> bool:
    """The draft's first sentence is a sentence already said, word for word up
    to where the shorter one ends ("X wrote it." then "X wrote it (d. 676)")."""
    def toks(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    head = toks(re.split(r"(?<=[.!?])\s+", (draft or "").strip(), maxsplit=1)[0])
    for said in re.split(r"(?<=[.!?])\s+", already or ""):
        st = toks(said)
        short = min(len(st), len(head))
        if short < 5:
            continue
        n = 0
        while n < short and st[n] == head[n]:
            n += 1
        if n >= 0.8 * short:
            return True
    return False


def paraphrases(already: str, draft: str) -> bool:
    """The draft's first sentence says what was said in other words: nearly all
    of its content words were already spoken ("al-Bukhari compiled it, a Persian
    scholar" after "al-Bukhari, a 9th-century Persian scholar, compiled it")."""
    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                if (len(w) > 2 or w.isdigit()) and w not in _FILLER]

    head = words(re.split(r"(?<=[.!?])\s+", (draft or "").strip(), maxsplit=1)[0])
    if len(head) < 5:
        return False
    said = {w[:5] for w in words(already)}
    if any(w.isdigit() and w not in said for w in head):
        return False  # a new number is a new fact
    return sum(w[:5] in said for w in head) >= 0.75 * len(head)


_META_PREFIXES = (
    "nothing missing", "nothing else", "nothing more", "nothing left", "you asked",
    "you'd asked", "you'd only asked", "you only asked", "you just asked", "you had asked",
    "done.", "done!", "both answers", "both parts", "that's the plain answer",
)
_META_CONFIRM = re.compile(r"\b(?:only|just|twice|both times|too|didn'?t ask|not ask|"
                           r"and i (?:said|answered))\b")


def _meta_lead(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lstrip().lower().replace("\u2019", "'"))


def is_meta(sentence: str) -> bool:
    """A sentence only about the asking or about being done ("Nothing missing.",
    "You asked 'where' only.", "You'd asked this earlier too."): no content."""
    low = _meta_lead(sentence).strip()
    if not low or re.search(r"\d", low):
        return False
    if low.startswith(("nothing missing", "nothing else", "nothing more", "nothing left",
                       "that's the plain answer")):
        return True
    if low.rstrip(".!") == "done":
        return True
    if low.startswith(("both answers", "both parts")):
        return bool(re.search(r"\b(?:done|delivered|covered|answered|given)\b", low))
    m = re.match(r"you(?:'d| had)? (?:only |just )?asked\b", low)
    return bool(m and (_META_CONFIRM.search(low) or m.group(0) != "you asked"))


class MetaFilter:
    """Streaming: a sentence that may be a remark about the asking is held
    until it ends, and dropped if it is one. Other sentences pass through as
    soon as their first words rule that out."""

    def __init__(self):
        self.buf = ""
        self.start = True
        self.dropped = 0

    def feed(self, piece: str) -> str:
        out, text = "", piece or ""
        while text:
            if not self.start:
                k = _sentence_end(text)
                if k == -1:
                    return out + text
                out, text, self.start = out + text[:k], text[k:], True
                continue
            self.buf, text = self.buf + text, ""
            lead = _meta_lead(self.buf)
            if not lead:  # spacing between sentences passes; only words are held
                out, self.buf = out + self.buf, ""
                return out
            if len(self.buf) > 240 or not any(p.startswith(lead) or lead.startswith(p)
                                              for p in _META_PREFIXES):
                text, self.buf, self.start = self.buf, "", False
                continue
            k = _sentence_end(self.buf)
            if k == -1:
                return out
            sentence, text, self.buf = self.buf[:k], self.buf[k:], ""
            if is_meta(sentence):
                self.dropped += 1
            else:
                out += sentence
        return out

    def flush(self) -> str:
        rest, self.buf, self.start = self.buf, "", True
        if is_meta(rest):
            self.dropped += 1
            return ""
        return rest


def echoes(already: str, draft: str) -> bool:
    """The draft's first sentence opens the way a spoken sentence already did
    ("Yes - and the part about time..." twice), or restates it outright."""
    def toks(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", (text or "").lower())

    first = re.split(r"(?<=[.!?])\s+", (draft or "").strip(), maxsplit=1)[0]
    head = toks(first)
    if len(head) < 4:
        return False
    for said in re.split(r"(?<=[.!?])\s+", already or ""):
        st = toks(said)
        n = 0
        while n < min(len(st), len(head)) and st[n] == head[n]:
            n += 1
        if n >= 5 or n >= 0.6 * len(head):
            return True
    return restates(already, first)


def ends_on_sentence(text: str) -> bool:
    """True when ``text`` stops on finished thought rather than mid-clause.

    This gates the handoff, not the text. Whatever has been streamed to the
    person has to stay streamed - trimming it afterwards would make the reply
    visibly shrink. So a beat that rambles past its length is spoken whole and
    simply does not trade the microphone, because switching mid-clause is what
    makes a two-pass mind sound like two minds.
    """
    text = (text or "").rstrip()
    if not text:
        return False
    return text.rstrip("\"')\u201d\u2019")[-1:] in _TERMINAL


@dataclass
class Beat:
    """One pass of one writing. The unit the whole design is measured in."""

    writing: str
    text: str = ""            # what the person heard
    hold: str = ""            # what it deliberately did not say yet
    raw: str = ""             # everything the model emitted
    ms: float = 0.0
    ttft: float = 0.0         # first token, for the side-by-side test
    tool_calls: list[dict] = field(default_factory=list)
    verdict: str = ""
    note: str = ""            # what the silent pass handed over
    private: str = ""         # every private suffix it wrote (hold, feel)


@dataclass
class DualResult:
    text: str = ""
    beats: list[Beat] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    ms: float = 0.0
    # What the person said while she was mid-turn, in the order it arrived, and
    # how each one was handled ("steer" = re-aimed without stopping, "stop" =
    # the reply ended and the new words became the question, "end" = stopped).
    interrupts: list[dict] = field(default_factory=list)

    @property
    def switches(self) -> int:
        return sum(1 for a, b in zip(self.beats, self.beats[1:]) if a.writing != b.writing)

    def trace(self) -> str:
        """Human-readable record of who said what, in what order."""
        out = []
        for i, b in enumerate(self.beats, 1):
            out.append(f"  [{i}] {b.writing} {b.ms:7.0f}ms ttft={b.ttft:6.0f}ms  {b.text[:70]!r}")
            if b.hold:
                out.append(f"       \u27e6hold\u27e7 {b.hold[:60]}")
            if b.note:
                out.append(f"       \u2190 {b.note[:60]}")
            if b.verdict:
                out.append(f"       \u26a0 {b.verdict[:60]}")
            for c in b.tool_calls:
                out.append(f"       \u270b {c['name']} -> {str(c['result'])[:60]}")
        return "\n".join(out)


Emitter = Callable[[object], Awaitable[None]]

# How fast a reply is heard aloud, for pacing narration and for the
# talked-during-tools metric (about 180 words a minute).
SPOKEN_CHARS_PER_S = 15.0


def spoken_seconds(text: str) -> float:
    return len((text or "").strip()) / SPOKEN_CHARS_PER_S


class DualLoop:
    """The kernel. Needs an LLM, a system prompt, and nothing else.

    ``caps`` is whatever the host *could* supply; every field is optional and the
    loop behaves correctly with none of them.
    """

    # BR3 on long waits: after the first clause, at most this many more while
    # the hands are still out, each once the last has had time to be heard.
    narrate_repeats = 5
    narrate_gap = 1.0

    def __init__(
        self,
        llm: LLM,
        *,
        system: str,
        caps: CapabilityBundle | None = None,
        pair: tuple[Writing, Writing] | None = None,
        max_beats: float = float('inf'),
        whole_handoff: bool = False,
        min_switch_chars: int = 220,
        max_tool_rounds: float = float('inf'),
        max_tokens: int | None = None,
        thought_timeout: float = 2.5,
        memory_timeout: float = 1.5,
        silent_thinks: bool = True,
        tool_prefetch: bool = True,
        on_event: Callable[[dict], None] | None = None,
        steer_llm: LLM | None = None,
        steer_max_tokens: int = 90,
        steer_every_beat: bool = False,
        narrate_tools: bool = True,
        narrate_max_tokens: int = 48,
        early_handoff: bool = True,
        couple_timeout: float = 0.12,
        final_verify_wait: float = 1.5,
        host_tools: "set[str] | None" = None,
        host_sent: "Callable[[], bool] | None" = None,
        steer_on_interrupt: bool = False,
        interrupt_grace: float = 0.35,
        prefetch_grace: float = 0.5,
        final_hands_wait: float = 15.0,
        vision: bool = False,
        models: dict | None = None,
        orient: "Callable[[str], str] | None" = None,
        reground: bool = True,
        anticipated: dict | None = None,
        on_anticipate: "Callable[[dict], None] | None" = None,
        on_late_hands: "Callable[[list], None] | None" = None,
        persist: bool = False,
        task_cap: float = float('inf'),
        bridges: bool = True,
        act_gate: bool = False,
        parts: bool = True,
        part_rounds: float = float('inf'),
        part_timeout: float | None = None,
        public_lane: str | None = None,
    ):
        if public_lane not in (None, "L", "R"):
            raise ValueError("public_lane must be L or R")
        self.public_lane = public_lane
        self.llm = llm
        self.kv_generation_worker = None
        self.persist = persist
        self.bridges = bridges
        # On a request for action, a beat that may press a tool shows only its
        # first clause live; the rest waits for the stream to end and is dropped
        # if the beat calls a tool - so no result is said before it returns.
        self.act_gate = act_gate
        self._acting = False
        # Two parts at once (README.md): in task mode a
        # round whose calls need different resources runs as two parts, one per
        # lobe, each with its own next steps; one voice answers when both land.
        self.parts = parts and public_lane is None
        self.part_rounds = part_rounds
        self.part_timeout = part_timeout
        self._parts_live = False
        self._part_steers: list[str] = []
        self._parts_rounds = 0
        self._parts_note = ""
        self.page_cap = max(task_cap, _PAGE_BUDGET_CAP) if persist else _PAGE_BUDGET_CAP
        self._task_live = False
        self._todo: list[dict] = []
        self._todo_by_model = False
        self._todo_parts: list[str] = []
        self._last_plan = ""
        # L speaks from AL, R from AR. None means the server's one model.
        self.system = system
        self.caps = caps or NullCapabilities()
        self.left, self.right = pair or default_writings()
        self.max_beats = max_beats
        self.whole_handoff = whole_handoff
        self.min_switch_chars = min_switch_chars
        self.max_tool_rounds = max_tool_rounds
        self.max_tokens = max_tokens
        self.thought_timeout = thought_timeout
        self.memory_timeout = memory_timeout
        self.silent_thinks = silent_thinks and public_lane is None
        self.tool_prefetch = tool_prefetch and public_lane is None
        # Kernel events for hosts that want to SHOW the two lobes (beat, hold,
        # silent, verify). Never allowed to break a turn.
        self.on_event = on_event
        # Steer mode (decided 2026-10-21): a cheap model does all the speaking
        # and tool work; an expensive model only writes short directives in the
        # silent pass at the critical moments - the opening beat, every tool
        # wait, and after a failed check. Everything else stays on the cheap one.
        self.steer_llm = steer_llm
        self.steer_max_tokens = steer_max_tokens
        self.steer_every_beat = steer_every_beat
        # K2 / BR3: while the hands are gone, the other writing keeps talking -
        # one short clause about what is happening, never an invented result.
        self.narrate_tools = narrate_tools
        self.narrate_max_tokens = narrate_max_tokens
        # The quiet lobe starts drafting when the mouth does. Its words stay
        # gated until the sentence ends, so the shift has no gap. A tool call
        # still finishes before that draft is allowed to speak.
        self.early_handoff = early_handoff and public_lane is None
        # How long the mouth waits, before its first token, for a clause from
        # the other lobe. Past this, the direct answer starts anyway.
        self.couple_timeout = couple_timeout
        # How long the end of a reply waits for C, to allow one correction beat.
        self.final_verify_wait = final_verify_wait
        # Tools the CALLER runs (a visiting agent's). Calling one ends the reply
        # at once: the agent needs the call now, and the result only comes with
        # its next request - there is nothing to narrate over or prepare for.
        self.host_tools = set(host_tools or ())
        self.host_sent = host_sent
        self.final_hands_wait = final_hands_wait
        # Computer use: screenshots go to the model as images only if it can see.
        self.vision = vision
        self.models = models or {}
        # Fresh time and place, called on every line A writes. None in tests.
        self.orient = orient
        # After a screen action, the screen is read again inside the same
        # tool wait, so the next beat never acts on a stale view (C10).
        self.reground = reground
        # A20: what the silent side guessed they would ask next, and its answer.
        self.anticipated = anticipated or None
        self.on_anticipate = on_anticipate
        # C12: a job still running when the reply ends reports here when done.
        self.on_late_hands = on_late_hands
        # ---- interruption (Issue #40) -------------------------------------
        # A host calls interrupt() while run() is in flight. The loop notices at
        # its seams - the places it already checks for finished background work -
        # so an interrupt never lands mid-clause of the emitter's monotonic text.
        # With steer_on_interrupt the same words become a directive to the silent
        # writing instead of a new turn: the task keeps going, re-aimed. That is
        # the interruption a single pass cannot offer - it has no silent side to
        # receive the words while its one voice is busy.
        self.steer_on_interrupt = steer_on_interrupt
        self.interrupt_grace = interrupt_grace
        self.prefetch_grace = prefetch_grace
        self._interrupted = False
        self._pending_words: list[str] = []
        self._pending_content: list[dict] = []
        self._interrupt_content: list[dict] = []
        self._user_content: list[dict] = []
        self._beats_used = 0
        # Hard cap set by _apply_verdict on a RED. One correction beat is allowed
        # (so A can say what she actually did and could not do), then the turn ends.
        # Without this, the recourse language in the verdict note is advisory - the
        # model reads it as "I was wrong, let me do another beat correcting it," and
        # the correction itself gets flagged, leading to self-correction cycles like
        # "Wait - I got that wrong" repeated five times. The cap gives the recourse
        # mechanical teeth without touching the content of the correction.
        self._red_stop = False
        self._interrupt_at = 0.0
        self._cut_target = None
        self._speaking_id = None
        self._int_evt = None
        # B (and the Director) speak into this while a sentence is still going.
        # The note joins the shared mind at the next sentence, the way a person
        # says "oh wait" without throwing away the sentence they were in.
        self._inner_notes: list[str] = []
        self._inner_evt = None
        self.steer_calls = 0
        self.steer_chars = 0
        self._t0 = time.perf_counter()
        self._steer_next = False
        # asyncio only keeps weak references to running tasks.
        self._pending: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    def interrupt(self, text: str = "", *, user_content: list[dict] | None = None) -> None:
        """The person spoke while this turn was in flight.

        Cheap and synchronous on purpose: a host calls it from its own task (the
        voice barge-in, the chat box, a hotkey) at whatever moment the words
        arrive, and must not await anything to be heard. The loop notices at its
        next seam.

        With ``steer_on_interrupt`` the words re-aim the task that is running
        instead of ending it; otherwise the reply ends and their words become the
        next question - which is all a single-pass mind can ever do.
        """
        self._interrupted = True
        if user_content is not None:
            # Structured content is authoritative: a caller's label is not speech.
            self._pending_content.extend(deepcopy(user_content))
            text = self._content_text(user_content)
        if text and text.strip():
            self._pending_words.append(text.strip())
            if self._parts_live:
                self._part_steers.append(text.strip())
        # Graceful, immediate: the writing holding the mic finishes its current
        # SENTENCE and stops (not the whole beat), and the loop wakes now - not
        # at the next seam - so the silent writing starts on the new command
        # while that last sentence is still being spoken.
        self._cut_target = getattr(self, "_speaking_id", None)
        self._interrupt_at = time.perf_counter()
        evt = getattr(self, "_int_evt", None)
        if evt is not None:
            evt.set()

    def hear_inner(self, text: str) -> None:
        """The other side of the mind found something while this sentence is going."""
        text = (text or "").strip()
        if not text:
            return
        self._inner_notes.append(text[:2000])
        evt = getattr(self, "_inner_evt", None)
        if evt is not None:
            evt.set()

    def _drain_inner(self) -> list[str]:
        notes, self._inner_notes = self._inner_notes, []
        return notes

    def _keep_alerts(self, text: str, note: str) -> str:
        """A near match must survive a silent pass that would rather say nothing."""
        text = (text or "").strip()
        if "<similarity_alerts>" not in (note or ""):
            return text[:400]
        if "<similarity_alerts>" in text:
            return text[:800]
        if not text or text.lower().startswith("none"):
            return note[:800]
        return (text.rstrip() + "\n" + note)[:800]

    def _take_interrupt(self) -> list[str] | None:
        """Drain whatever the person said since the last seam. None if nothing."""
        if not self._interrupted:
            return None
        self._interrupted = False
        words, self._pending_words = self._pending_words, []
        self._interrupt_content, self._pending_content = self._pending_content, []
        if self._interrupt_content:
            self._user_content = [*self._user_content, *deepcopy(self._interrupt_content)]
        return words

    @staticmethod
    def _content_text(content: list[dict]) -> str:
        """Only explicit text parts; media and recording metadata never become words."""
        return " ".join(str(p.get("text", "")) for p in content
                        if isinstance(p, dict) and p.get("type") == "text").strip()

    def _has_media(self) -> bool:
        """Original non-text input (e.g. audio) is part of this turn's shared mind."""
        return any(isinstance(p, dict) and p.get("type") != "text"
                   for p in getattr(self, "_user_content", None) or [])

    def _with_content(self, prompt: str):
        content = getattr(self, "_user_content", None)
        return [{"type": "text", "text": prompt}, *deepcopy(content)] if content else prompt

    def _interruption_message(self, text: str):
        return deepcopy(self._interrupt_content) if self._interrupt_content else text

    # ------------------------------------------------------------------
    async def stream(self, user_text: str, *, history: list[dict] | None = None) -> AsyncIterator[str]:
        """Yield the reply as it grows.

        The buffer only ever gets longer, which is why the person never sees a
        restart at a handoff - the emitter receives the whole reply each time.
        """
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self.run(user_text, history=history, emit=q.put))
        finished = False
        try:
            while True:
                item = await q.get()
                if item is None:
                    finished = True
                    break
                yield item
        finally:
            if not finished and not task.done():
                await self._cancel(task)
            else:
                await task

    async def run(self, user_text: str = "", *, history: list[dict] | None = None,
                  starter: str | None = None, ready: set[str] | None = None,
                  user_content: list[dict] | None = None,
                  emit: Emitter | None = None) -> DualResult:
        """Think about ``user_text`` and produce one reply in several passes."""
        t0 = time.perf_counter()
        self._t0 = t0
        self._user_content = deepcopy(user_content) if user_content is not None else []
        if user_content is not None:
            user_text = self._content_text(user_content)
        self._respond_after = None
        self._terse = asks_for_terse(user_text)
        self._acting = asks_for_action(user_text)
        first = self.public_lane or starting_lobe(user_text, starter=starter, ready=ready or ())
        out = _monotonic(emit, self._on_public) if emit else _silent
        # The past is fetched before the first word, but with a deadline: a slow
        # database must not become a slow reply. A mind that answers late is
        # worse than a mind that answers with less.
        recall = asyncio.create_task(self._recall(user_text))
        # The prefix. Everything here is visible to BOTH writings, so it is the
        # shared mind. Nothing above the tail changes mid-run.
        system = self.system
        found = await self._collect(recall, self.memory_timeout)
        if found:
            # Folded INTO the first system message rather than added as a second
            # one: strict OpenAI-compatible providers reject a system message
            # anywhere but position 0. Still written once, above the tail, so it
            # stays byte-identical across every handoff.
            system = (system + "\n\n[what you remember about this, not shown to them as a list]\n"
                      + found)
        if self._splits():
            system += PARTS_HINT
        prefix: list[dict] = [{"role": "system", "content": system}]
        # A host that passes system rows inside history would trip the same
        # rule, so they join the head instead of the middle.
        stray = [str(m.get("content", "")) for m in (history or []) if m.get("role") == "system"]
        if stray:
            prefix[0]["content"] = prefix[0]["content"] + "\n\n" + "\n\n".join(stray)
        prefix.extend(m for m in (history or []) if m.get("role") != "system")
        prefix.append({"role": "user", "content": deepcopy(user_content)
                       if user_content is not None else user_text})

        result = DualResult()
        spoken = ""
        current, other = ((self.right, self.left) if first == "R" else (self.left, self.right))
        pending_note = ""
        pending_hold = ""
        tool_rounds = 0
        guess = self.anticipated or {}
        if guess.get("a") and screen_proves(str(guess.get("q", "")), user_text):
            # A20: the silent side answered this before they asked it.
            pending_note = ("You already thought about this before they asked: "
                            f"{str(guess['a'])[:400]} Use it if it still fits, in your own words.")
            self._event({"type": "silent", "writing": self.right.key, "kind": "anticipated", "phase": "end",
                         "name": str(guess.get("q", ""))[:120], "hit": True, "took_ms": 0})
        owed_answer = False
        self._task_live = False
        self._todo = []
        self._todo_by_model, self._todo_parts, self._last_plan = False, [], ""
        task_nudges, task_rounds_seen, reviewed, task_finished = 0, 0, False, False
        task_outcome = ""
        terse_from: int | None = 0  # where the bare answer they asked for starts
        nudged_page = False
        status_nudges = 0
        screen_plan: ScreenPlan | None = None
        fast_fires = 0

        prestart = None          # the next writing's call, already running (see _seam_prestart)
        verify_task = None       # C checks in the background; never holds the mic
        late_thought = None      # a silent thought that missed its seam goes to the beat after
        carry_note, carry_hold = "", ""  # what a pre-started beat could not see yet
        bg_hands = None          # tools asked for after the mic already moved on
        budget = self.max_beats  # grows when a re-course needs room to land
        try:
            index = -1
            while True:
                index += 1
                if self.persist and tool_rounds:
                    self._task_live = True
                    self._kernel_todo(user_text, result.tool_calls)
                    if tool_rounds != task_rounds_seen:
                        task_nudges, task_rounds_seen, reviewed = 0, tool_rounds, False
                        if trimmed := trim_history(prefix):
                            self._event({"type": "trim", "n": trimmed})
                if index >= budget:
                    if (self._task_live and not task_finished and budget < self.page_cap
                            and task_nudges < _TASK_NUDGES and not self._interrupted):
                        task_nudges += 1
                        budget += 1
                        pending_note = "\n".join(x for x in (pending_note, KEEP_GOING) if x)
                    else:
                        break
                # RED hard-cap: _apply_verdict sets _red_stop on RED and lets
                # one correction beat land; after that the turn ends, no matter
                # what the budget says. Prevents the self-correction cycle.
                if self._red_stop:
                    break
                if terse_from is None:
                    terse_from = len(spoken)  # a mid-beat steer: the answer starts here
                elif self._terse and not owed_answer and spoken[terse_from:].strip():
                    break  # they asked for a bare answer and it has been given
                self._beats_used = index + 1
                beat = Beat(writing=current.key)
                started = time.perf_counter()
                announced = prestart is not None and prestart.get("announced") and prestart["writing"] == current.key
                if not announced:
                    # chars = where this writing's words start in the reply
                    self._event({"type": "beat", "writing": current.key, "index": index,
                                 "chars": len(spoken) + (1 if spoken else 0)})

                # C's verdict on what was already said lands whenever it is ready.
                if verify_task is not None and verify_task.done():
                    self._apply_verdict(verify_task, prefix, result)
                    verify_task = None
                if bg_hands is not None and bg_hands.done():
                    landed = self._land_hands(bg_hands, prefix, result) or ""
                    bg_hands = None
                    if landed and any(i["kind"] in ("steer", "tool_recourse") for i in result.interrupts):
                        # The person re-aimed the task; what the hands found for
                        # the OLD direction is context, not an instruction.
                        landed = "Findings from the search they interrupted: " + landed
                    pending_note = "\n".join(x for x in (pending_note, landed) if x)
                if late_thought is not None and late_thought.done():
                    extra = self._done_text(late_thought)
                    late_thought = None
                    if extra:
                        pending_note = "\n".join(x for x in (pending_note, extra) if x)

                # An interrupt is noticed HERE, at the seam, and nowhere else: the
                # emitter is monotonic, so words already sent can never be taken
                # back, and an interrupt landing mid-clause would either be
                # dropped or stitched into a half sentence.
                words = self._take_interrupt()
                if words is not None:
                    said = " ".join(words)[:300]
                    if not self.steer_on_interrupt:
                        # The plain kind: the reply ends, their words become the
                        # next question. This is all a single-pass mind can do.
                        result.interrupts.append({"words": said, "kind": "stop",
                                                  "beats": self._beats_used})
                        self._event({"type": "interrupt", "kind": "stop", "words": said})
                        break
                    if tool_rounds < self.max_tool_rounds:
                        tool_rounds += 1
                    budget += 1  # a re-course is a new direction; give it room
                    self._terse = asks_for_terse(said)
                    terse_from = len(spoken)
                    heard_new = said or "new original audio (attached above; no transcript)"
                    pending_note = "\n".join(x for x in (
                        pending_note,
                        f"They interrupted with: {heard_new}. Fold that into what you are "
                        f"doing now - adjust the plan, do not start over.") if x)
                    cut = "" if self._terse else _last_sentence(spoken)
                    if cut:
                        # A19: the thread they cut is kept, not lost.
                        pending_note += (f"\nYou were in the middle of saying: \"{cut}\". If that still "
                                         "matters after you answer them, come back to it in one short "
                                         "clause; if their words made it moot, drop it.")
                        self._event({"type": "silent", "writing": other.key, "kind": "cut_thread",
                                     "phase": "end", "name": cut[:120], "took_ms": 0})
                    if said or self._interrupt_content:
                        prefix.append({"role": "user", "content": self._interruption_message(said)})
                    pending_hold = ""
                    result.interrupts.append({"words": said, "kind": "steer",
                                              "beats": self._beats_used})
                    self._event({"type": "interrupt", "kind": "steer", "words": said})
                    # Anything already running was aimed at the OLD direction.
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    if late_thought is not None:
                        await self._cancel(late_thought)
                        late_thought = None
                    if bg_hands is not None and bg_hands.done():
                        self._land_hands(bg_hands, prefix, result)
                        bg_hands = None

                # The silent pass works WHILE this pass talks. That overlap is
                # the entire design; everything else is bookkeeping.
                steer_now = (index == 0) or self._steer_next or self.steer_every_beat
                self._steer_next = False
                thought = self._track(self._think(other, spoken, user_text, remembered=found,
                                                          steer=steer_now))

                # The seam: the next writing streams onto EXACTLY the text it
                # will be joined to (with the space), or the monotonic emitter
                # sees a buffer that is not an extension and drops the stream -
                # which showed up as ~640 ms of silence and a missing space.
                lead = spoken + " " if spoken else ""
                visible_done = asyncio.Event()
                prep_ready = asyncio.Event()
                shift_ready = asyncio.Event()
                declined_quiet = False
                live: list[str] = []
                plan_box: dict = {}
                self._speaking_id = ("beat", index)
                can_prepare = self.early_handoff and index + 1 < budget and not self.host_tools

                def _other_ready() -> bool:
                    # The floor may pass only to words already in hand.
                    p = prestart
                    if p is None:
                        return False
                    if p["task"].done():
                        return True
                    prev = p.get("previous") or {}
                    return bool("".join(p["live"]).strip() or "".join(prev.get("live") or []).strip())

                def _hand_floor(p: dict) -> None:
                    if can_prepare and p.get("hold_ref") is not None:
                        p["hold_ref"]["f"] = _other_ready
                # A check, a tool result, or a new request landed after this
                # draft was started. What was merely said is already in its
                # tail, so that alone must not throw the draft away: doing so
                # opens a gap while a fresh call starts.
                if (prestart is not None and not prestart.get("recourse")
                        and prestart.get("writing") == current.key
                        and self._draft_is_stale(prestart, prefix)):
                    await self._end_prestart(prestart, dropped=True)
                    prestart = None
                # The draft was written beside the mouth's sentence, and the mouth
                # may have borrowed its opening. Released now, it would say the
                # same thing twice; a fresh call says the next thing instead.
                if (prestart is not None and not prestart.get("recourse")
                        and prestart.get("writing") == current.key and not prestart["gate"].open
                        and echoes(lead, "".join(prestart["live"]))):
                    self._event({"type": "silent", "writing": current.key, "kind": "echo", "phase": "end"})
                    await self._end_prestart(prestart, dropped=True)
                    prestart = None
                if prestart is not None and prestart.get("recourse") and prestart["writing"] == current.key:
                    # Re-aimed at the person's new command the moment they gave
                    # it; its first words were prepared during the cut sentence.
                    msgs, speak_task, gate = prestart["msgs"], prestart["task"], prestart["gate"]
                    visible_done, live = prestart["visible_done"], prestart["live"]
                    prep_ready = prestart.get("prep_ready") or prep_ready
                    shift_ready = prestart.get("shift_ready") or shift_ready
                    plan_box = prestart.get("plan_box") or plan_box
                    _hand_floor(prestart)
                    # Their new instruction joins the shared mind, so every beat
                    # from here on follows it - not just the one that switched.
                    prefix.append({"role": "user", "content": prestart.get("user_content", prestart["words"])})
                    pending_note, pending_hold = "", ""
                    prestart["lead_ref"]["v"] = lead
                    self._speaking_id = prestart["speak_id"]
                    self._event({"type": "beat", "writing": current.key, "index": index, "chars": len(lead),
                                 "recourse": True})
                    held_text = (lead + "".join(live)) if live else None
                    # Only words that actually reach the person count as her answer.
                    self._respond_after = len(lead.rstrip())
                    self._event({"type": "silent", "writing": current.key, "kind": "recourse", "phase": "end",
                                 "took_ms": round((time.perf_counter() - prestart["t0"]) * 1000)})
                    prestart = None
                    await gate.release(held_text)
                    announced = True
                elif prestart is not None and prestart["writing"] == current.key:
                    # Started while the previous writing was still writing its
                    # private note: the person never waits for that note.
                    msgs, speak_task, gate = prestart["msgs"], prestart["task"], prestart["gate"]
                    visible_done, live = prestart["visible_done"], prestart["live"]
                    prep_ready = prestart.get("prep_ready") or prep_ready
                    shift_ready = prestart.get("shift_ready") or shift_ready
                    plan_box = prestart.get("plan_box") or plan_box
                    _hand_floor(prestart)
                    self._speaking_id = prestart.get("speak_id", self._speaking_id)
                    # Prepared against a partial sentence. Join it to the finished
                    # one, or the monotonic emitter drops the whole half.
                    ref = prestart.get("lead_ref")
                    if ref is not None and not prestart.get("committed"):
                        ref["v"] = lead
                    fresh = "".join(live)
                    await self._end_prestart(prestart, dropped=False)
                    prestart = None
                    await gate.release((lead + fresh) if fresh and not gate.open else None)
                else:
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    # The other lobe starts with this one. Its draft is in the
                    # mouth's prompt, so the sentence that gets said was written
                    # while the other half already existed. Tools stay on the mouth.
                    drafted = ""
                    if can_prepare:
                        prestart = self._arm_draft(prefix, other, "", out, index + 1)
                        # Only the first sentence waits. A later wait would be
                        # silence between two halves that should already overlap.
                        if not spoken:
                            await self._wait_draft(prestart)
                        drafted = self._draft_excerpt(prestart)
                    msgs = list(prefix)
                    msgs.append({"role": "user", "content": self._tail(
                        current, pending_note, pending_hold, drafted=drafted)})
                    speak_task = self._track(
                        self._speak(msgs, current, out, lead, visible_done=visible_done, live=live,
                                    speak_id=self._speaking_id, prep_ready=prep_ready,
                                    sentence_limit=self._sentence_limit(),
                                    plan_box=plan_box, plan_ready=prep_ready if can_prepare else None,
                                    hold_for=_other_ready if can_prepare else None,
                                    require_answer=owed_answer,
                                    bridge=self._bridge_for(current)
                                    if result.beats and not self._interrupted and not owed_answer else ""))

                # The draft stays gated until this stream finishes. Opening it
                # earlier would speak the other half before a tool call is known,
                # and the findings would never reach the next sentence.
                self._int_evt = asyncio.Event()
                if self._interrupted:
                    self._int_evt.set()
                int_wait = self._track(self._int_evt.wait())
                self._inner_evt = asyncio.Event()
                if self._inner_notes:
                    self._inner_evt.set()
                inner_wait = self._track(self._inner_evt.wait())
                prep_wait = self._track(prep_ready.wait()) if can_prepare else None
                # Readiness alone never transfers ownership. Only a finished
                # sentence followed by a plainly non-floor suffix (feel/hold) may
                # hand over early; floor intent is written before speech.
                suffix_wait = self._track(visible_done.wait()) if can_prepare else None
                waiting = {speak_task, int_wait, inner_wait} | ({prep_wait} if prep_wait else set()) | (
                    {suffix_wait} if suffix_wait else set())
                while not speak_task.done():
                    done_now, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                    if int_wait in done_now:
                        waiting.discard(int_wait)
                        if self.steer_on_interrupt and not (prestart and prestart.get("committed")):
                            words = self._take_interrupt()
                            if words is not None:
                                said = " ".join(words)[:300]
                                if prestart is not None:
                                    await self._end_prestart(prestart, dropped=True)  # aimed at the old direction
                                prestart = self._recourse_prestart(prefix, other, "".join(live).strip(), said,
                                                                   out, ("beat", index + 1))
                                budget += 1
                                self._terse = asks_for_terse(said)
                                if self._terse:
                                    terse_from = None
                                result.interrupts.append({"words": said, "kind": "steer",
                                                          "beats": self._beats_used, "mid_beat": True})
                                self._event({"type": "interrupt", "kind": "steer", "words": said})
                        self._int_evt = asyncio.Event()
                        int_wait = self._track(self._int_evt.wait())
                        waiting.add(int_wait)
                        continue
                    if inner_wait in done_now:
                        waiting.discard(inner_wait)
                        # Finish this sentence, then the memory joins. A user
                        # interrupt already owns the cut. A beat that has not
                        # said anything yet is not cut: the note waits for the
                        # next seam instead of leaving the reply empty.
                        if (not self._interrupted and self._speaking_id is not None
                                and "".join(live).strip()):
                            self._cut_target = self._speaking_id
                        self._inner_evt = asyncio.Event()
                        inner_wait = self._track(self._inner_evt.wait())
                        waiting.add(inner_wait)
                        continue
                    if prep_wait in done_now:
                        waiting.discard(prep_wait)
                        prep_wait = None
                        # The fact has started. Restart the quiet draft against
                        # what is actually being said, still before this sentence
                        # ends, still gated. One pair of hands: the draft cannot
                        # press a tool.
                        heard = "".join(live).strip()
                        if plan := plan_box.get("text"):
                            heard = (heard + " " if heard else "") + (
                                f"(Its whole point, shared privately before saying it: {plan})")
                        if heard and prestart is not None and not prestart.get("recourse") and not prestart.get("refreshed"):
                            prestart = await self._refresh_draft(
                                prestart, prefix, other, heard, thought, out, index + 1)
                        elif prestart is None and heard:
                            prestart = self._arm_draft(prefix, other, heard, out, index + 1)
                    if suffix_wait in done_now:
                        waiting.discard(suffix_wait)
                        suffix_wait = None
                        said_now = "".join(live).strip()
                        if (prestart is not None and not prestart.get("recourse")
                                and not prestart.get("committed") and not self._interrupted
                                and ends_on_sentence(said_now)):
                            prestart = await self._choose_draft(prestart)
                            excerpt = self._draft_excerpt(prestart)
                            if (prestart is not None and prestart["task"].done() and excerpt
                                    and restates(said_now, excerpt)):
                                await self._end_prestart(prestart, dropped=True)
                                prestart = None
                                declined_quiet = True
                            else:
                                await self._open_handoff(prestart, lead, said_now)
                for waiter in (int_wait, inner_wait, prep_wait, suffix_wait):
                    if waiter is not None:
                        await self._cancel(waiter)
                res, ttft, hold = await speak_task
                speaker_intent = (res.raw or {}).get("floor", "")
                handoff_note = (res.raw or {}).get("handoff", "")
                self._event({"type": "floor", "writing": current.key, "intent": speaker_intent or "none",
                             "plan": bool((res.raw or {}).get("plan")), "handoff": bool(handoff_note)})
                listed = (res.raw or {}).get("todo", "")
                # A checkbox plan copied from the kernel's own list is not a new
                # list: live, it ticked every part after one page of three.
                if (not listed and self._task_live and not self._todo_parts
                        and re.search(r"\[\s*[xX ]?\s*\]", (res.raw or {}).get("plan", ""))):
                    listed = (res.raw or {}).get("plan", "")
                todo = parse_todo(listed)
                self._todo_by_model = self._todo_by_model or bool(todo)
                if todo and todo != self._todo:
                    self._todo = todo
                    self._event({"type": "todo", "writing": current.key, "items": todo})
                self._last_plan = (res.raw or {}).get("plan", "") or self._last_plan
                self._kernel_todo(user_text, result.tool_calls)
                shown_bridge, self._shown_bridge = getattr(self, "_shown_bridge", ""), ""
                if shown_bridge and not (res.text or "").strip():
                    spoken = (lead + shown_bridge).strip()
                if (self.whole_handoff and speaker_intent not in ("stay", "end")
                        and prestart is not None and (res.text or "").strip()
                        and not res.tool_calls and not prestart.get("recourse") and not prestart.get("committed")):
                    prestart = await self._hand_whole(prestart, prefix, other, (lead + res.text).strip(),
                                                      res.text, thought, out, index + 1)
                prestart = await self._choose_draft(prestart)
                other_declined = False
                excerpt = "".join(prestart.get("live") or []).strip() if prestart else ""
                said_this = (lead + (res.text or "")).strip()
                if declined_quiet or (
                        prestart is not None and not prestart.get("recourse") and not prestart.get("committed")
                        and ((prestart["task"].done() and not excerpt)
                             or (excerpt and restates(said_this, excerpt)))):
                    # The other half looked, and nothing was left out.
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    other_declined = True
                if not (res.text or "").strip() and not res.tool_calls:
                    # A page came back and nobody said what was on it. One more
                    # pass, aimed at the result, before the turn is allowed to die.
                    if owed_answer and not nudged_page:
                        nudged_page = True
                        if prestart is not None and not prestart.get("committed"):
                            await self._end_prestart(prestart, dropped=True)
                            prestart = None
                        prefix.append({"role": "user", "content": self._in_front(
                            "[not shown to them] The tool returned. Answer the original request "
                            "using its result. If it failed, use an available alternative or explain "
                            "what could not be verified. An opener or private done marker is not an answer.",
                            doing="reading what came back")})
                        if index + 1 >= budget and budget < self.page_cap:
                            budget += 1
                        continue
                    # One silent character must not suppress a useful peer.
                    if (prestart is not None and self._draft_excerpt(prestart)
                            and not result.beats and index + 1 < budget and not self._interrupted):
                        pending_note, pending_hold = "", ""
                        current, other = (other, current) if self.public_lane is None else (current, other)
                        continue
                    if prestart is not None and not prestart.get("committed"):
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    break
                beat.raw = (res.text or "").strip()
                beat.ttft = ttft
                beat.ms = (time.perf_counter() - started) * 1000
                beat.hold = hold
                beat.private = (res.raw or {}).get("private", "")
                if hold:
                    self._event({"type": "hold", "from": current.key, "text": hold[:300]})

                # Hands. If this pass asked for a tool, run it now - the voice is
                # already mid-sentence elsewhere, so the wait is not dead air.
                if (res.tool_calls and self.caps.tools is not None and tool_rounds < self.max_tool_rounds
                        and prestart is not None and prestart.get("committed") and bg_hands is None):
                    # The other writing already has the mic. The hands go to work
                    # behind it; their findings join at the next seam.
                    tool_rounds += 1
                    beat.text = beat.raw
                    result.beats.append(beat)
                    spoken = (lead + beat.text) if beat.text else spoken
                    prefix.append({"role": "assistant", "content": beat.text})
                    bg_hands = asyncio.create_task(self._hands([], res))
                    self._event({"type": "silent", "writing": current.key, "kind": "tool",
                                 "name": ",".join(c.name for c in res.tool_calls)[:120], "phase": "start"})
                    carry_note = "\n".join(x for x in (carry_note, self._done_text(thought) if thought.done() else "") if x)
                    if not thought.done():
                        late_thought = late_thought or thought
                    carry_hold = beat.hold or carry_hold
                    pending_note, pending_hold = "", ""
                    current, other = (other, current) if self.public_lane is None else (current, other)
                    continue
                if res.tool_calls and self.host_tools and any(c.name in self.host_tools for c in res.tool_calls):
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    tool_msgs: list[dict] = []
                    if self.caps.tools is not None:
                        calls, tool_msgs = await self._hands(msgs, res)
                        beat.tool_calls = calls
                        result.tool_calls.extend(calls)
                    beat.text = beat.raw
                    result.beats.append(beat)
                    spoken = (lead + beat.raw) if beat.raw else spoken
                    await out(spoken)
                    if self.host_sent is None or self.host_sent() or tool_rounds >= self.max_tool_rounds:
                        break  # the agent runs it; its result is the next request
                    # The director stopped it: the other writing decides again.
                    tool_rounds += 1
                    prefix.append(msgs[-1 - len(tool_msgs)])
                    prefix.extend(tool_msgs)
                    pending_note = "; ".join(str(m["content"]) for m in tool_msgs)[:600]
                    pending_hold = ""
                    current, other = (other, current) if self.public_lane is None else (current, other)
                    continue
                if res.tool_calls and self.caps.tools is not None and tool_rounds < self.max_tool_rounds:
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    tool_rounds += 1
                    # The wait is where the design earns its keep, so all of it
                    # runs at once: the hand goes to the host, the thought that
                    # was already running while this beat was spoken is
                    # collected, a fresh silent pass prepares for the result,
                    # and the other writing keeps talking (K2).
                    spoken = (lead + beat.raw) if beat.raw else spoken
                    await out(spoken)
                    # Calls on different resources: each lobe drives one part,
                    # with its own next steps, while the voice keeps talking.
                    split = split_parts(res.tool_calls, self.caps.tools) if self._splits() else None
                    if split:
                        hands = asyncio.create_task(run_parts(
                            self, msgs, res, split, (current, other),
                            rounds=(self.part_rounds if self.max_tool_rounds == float('inf') else min(self.part_rounds, max(0, (self.max_tool_rounds - tool_rounds) // 2)))))
                    else:
                        hands = asyncio.create_task(self._hands(msgs, res))
                    # The silent work runs beside the hands. Kept separate from
                    # them so an interrupt can let the hands keep going (a
                    # browser must not be abandoned mid-navigation) while the
                    # rest of the beat moves on. Split parts prepare their own
                    # next steps, so there is no prefetch to make for them.
                    prefetch = asyncio.ensure_future(
                        asyncio.sleep(0, "") if split else
                        self._prefetch(other, spoken, user_text, res, found, want_plan=screen_plan is None))
                    # A first screen action's prefetch also writes the task's plan,
                    # which the proof gate needs, so that one is not cut short.
                    plans_screen = (not split and screen_plan is None
                                    and any(_screen_tool(c.name) for c in res.tool_calls))
                    narrating = asyncio.ensure_future(self._narrate(
                        prefix, other, res, out, spoken,
                        busy=lambda: not self._interrupted and (
                            not hands.done() or (plans_screen and not prefetch.done()))))
                    side = asyncio.gather(self._collect(thought), prefetch, narrating,
                                          return_exceptions=True)
                    # The one thing that cuts the wait short is the person
                    # speaking. A single pass cannot offer that either: its only
                    # channel is the one the tool is using.
                    while not (hands.done() and side.done()) and not self._interrupted:
                        if hands.done() and narrating.done() and not prefetch.done() and not plans_screen:
                            # The result is in. The guess at it is advisory and
                            # must not hold the floor: live, a 24 s prefetch sat
                            # on an instant tool result.
                            await asyncio.wait({prefetch}, timeout=self.prefetch_grace)
                            if not prefetch.done():
                                prefetch.cancel()
                            continue
                        tick = asyncio.ensure_future(asyncio.sleep(self.interrupt_grace))
                        await asyncio.wait({asyncio.shield(hands), asyncio.shield(side), tick},
                                           return_when=asyncio.FIRST_COMPLETED)
                        tick.cancel()
                    interrupted_wait = self._interrupted and not hands.done()
                    carried, ready, narration = (x if isinstance(x, str) else "" for x in await side)
                    if interrupted_wait:
                        # The host's call stays in flight; its findings join the
                        # shared mind at a later seam (bg_hands -> _land_hands),
                        # tool_calls and results together.
                        bg_hands = hands
                        result.interrupts.append({"words": " ".join(self._pending_words)[:300],
                                                  "kind": "tool_recourse", "beats": self._beats_used})
                        calls, tool_msgs = [], []
                    else:
                        try:
                            calls, tool_msgs = await hands
                        except Exception:  # noqa: BLE001 - a failed hand is a report, not a crash
                            calls, tool_msgs = [], []
                        if split:
                            tool_rounds += self._parts_rounds
                    beat.tool_calls = calls
                    beat.note = "\n".join(x for x in (carried, ready) if x)
                    result.tool_calls.extend(beat.tool_calls)
                    result.beats.append(beat)
                    if narration:
                        spoken += " " + narration
                    # The hands' findings join the shared mind, so both writings
                    # can use them from here on. The assistant message that ASKED
                    # for the tools must precede the tool results - strict
                    # OpenAI-compatible providers reject a bare `tool` message.
                    if tool_msgs:
                        prefix.append(msgs[-1 - len(tool_msgs)])
                        prefix.extend(tool_msgs)
                    elif beat.raw:
                        # Hands still out (interrupted): only what was SAID joins
                        # now; the call and its result join together when they land.
                        prefix.append({"role": "assistant", "content": beat.raw})
                    if narration:
                        prefix.append({"role": "assistant", "content": narration})
                    if tool_msgs:
                        # Every tool result owes an answer, not just a screen.
                        owed_answer = True
                        if index + 1 >= budget and budget < self.page_cap:
                            budget += 1
                    await out(spoken)
                    pending_note = "The hands reported: " + "; ".join(
                        f"{c['name']} -> {str(c['result'])[:300]}" for c in beat.tool_calls)
                    if split and tool_msgs:
                        pending_note = self._parts_note
                    expected = _expectation(ready)
                    if expected and beat.tool_calls and not interrupted_wait:
                        got = " ".join(str(c.get("result", "")) for c in beat.tool_calls)
                        if not screen_proves(expected, got):
                            # D9: the silent side predicted one thing, the hands
                            # brought back another. That gap is worth a word.
                            pending_note += (f"\nThat is not what you expected (you expected: {expected}). "
                                             "If it changes anything, say so in a few words and adjust.")
                            self._event({"type": "silent", "writing": other.key, "kind": "surprise",
                                         "phase": "end", "name": expected[:120], "took_ms": 0})
                    if ready:
                        # Composed before the result existed. The pass reading it
                        # has the result in front of it and may disagree.
                        pending_note += f"\nDecided while waiting for that: {ready}"
                        if screen_plan is None:
                            if tasks := parse_plan(ready):
                                screen_plan = ScreenPlan(tasks)
                                self._event({"type": "silent", "writing": other.key, "kind": "plan", "subtasks": tasks})
                        if screen_plan is not None and not screen_plan.finished:
                            if f"subtask: {screen_plan.current}" in ready.lower() or marked_done(ready):
                                # The claim is cheap; the screen is not. The closing
                                # marker only counts when the newest observation of a
                                # screen hand carries the subtask's evidence - the
                                # prompt asked for this, the gate enforces it.
                                proved, gate_note = prove_or_wait(
                                    screen_plan.current, latest_screen_text(beat.tool_calls))
                                self._event({"type": "silent", "writing": other.key,
                                             "kind": "subtask_gate", "proved": proved,
                                             "subtask": screen_plan.current[:80]})
                                if proved:
                                    screen_plan.advance()
                                    # Explicit progress, NeuralAgent's rule: the
                                    # pass that presses next is told which goal
                                    # it is on and how many are left, so a long
                                    # plan cannot silently repeat or skip a step.
                                    if screen_plan.finished:
                                        pending_note += ("\nEvery subtask is done and the screen "
                                                         "showed each one. Say the outcome.")
                                    else:
                                        pending_note += (
                                            f"\nSubtask {screen_plan.index + 1} of "
                                            f"{len(screen_plan.subtasks)} now: {screen_plan.current}. "
                                            "Work only on this one.")
                                else:
                                    pending_note += "\n" + gate_note
                        failed = [c["name"] for c in beat.tool_calls if _failed(c["result"])]
                        if fast_fires < 3 and tool_rounds < self.max_tool_rounds and not failed:
                            if nxt := parse_next_action(ready):
                                name = nxt.get("name", "")
                                if name and any(t["function"]["name"] == name for t in self.caps.schemas_for("L")):
                                    fast_fires += 1
                                    scratch_calls = [ToolCall(id=f"fast-{fast_fires}", name=name, arguments=json.dumps(nxt["arguments"]))]
                                    scratch_res = ChatResult(text="", tool_calls=scratch_calls)
                                    tool_rounds += 1
                                    calls, t_msgs = await self._hands(msgs, scratch_res)
                                    beat.tool_calls.extend(calls)
                                    result.tool_calls.extend(calls)
                                    tool_msgs.extend(t_msgs)
                                    # The fast hand is a real hand: its call and result
                                    # must join the shared prefix before the next lobe
                                    # speaks, just like the original action above.
                                    prefix.append(msgs[-1 - len(t_msgs)])
                                    prefix.extend(t_msgs)
                                    self._event({"type": "fast", "name": name, "args": nxt["arguments"]})
                                    owed_answer = True
                                    if budget < self.page_cap:
                                        budget += 1
                    failed = [c["name"] for c in beat.tool_calls if _failed(c["result"])]
                    if failed:
                        # BR4 - quiet recovery. The failure is a fact for the
                        # mind, not news for the person: she takes the other path.
                        pending_note += QUIET_RECOVERY.format(names=", ".join(failed)[:200])
                    pending_hold = ""
                    if speaker_intent != "stay":
                        current, other = (other, current) if self.public_lane is None else (current, other)
                    continue

                beat.text = beat.raw
                result.beats.append(beat)
                spoken = (lead + beat.text) if beat.text else spoken
                if beat.text and not res.tool_calls and not _status_only(beat.text):
                    owed_answer = False
                await out(spoken)

                # The shared mind grows by what was said, not by the scaffolding.
                prefix.append({"role": "assistant", "content": beat.text})
                folded_inner = self._drain_inner()
                if folded_inner:
                    from ..brains.voice_a import inner_voice
                    prefix.append({"role": "user", "content": inner_voice(
                        [{"text": n} for n in folded_inner])})
                    self._event({"type": "inner", "text": folded_inner[0][:300], "n": len(folded_inner)})
                    if prestart is not None and not prestart.get("committed"):
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                # A thought that is not ready at the seam does not hold the mic:
                # a short grace, then it travels to the beat after.
                if thought.done() or prestart is None:
                    beat.note = await self._collect(thought, None if prestart is None else 0.0)
                else:
                    beat.note = ""
                    late_thought = thought
                if restates(spoken, beat.note) or _squash(spoken) == _squash(beat.note):
                    beat.note = ""

                # C verifies in the background (K6): the next beat is not held
                # back by it; its verdict joins the shared mind when it lands.
                if self.caps.verify is not None and verify_task is None:
                    verify_task = asyncio.create_task(self._check(spoken, user_text, result.tool_calls))
                    verify_task.beat_index = len(result.beats) - 1  # type: ignore[attr-defined]
                    self._event({"type": "verify", "state": "PENDING"})

                # The other half speaks in this same answer when it found something
                # the direct answer left out. A draft that already declined does
                # not get a second chance to pad the reply.
                other_spoken = any(b.writing == other.key for b in result.beats)
                need_shift = (self.early_handoff and not other_spoken and not other_declined
                              and ends_on_sentence(beat.text)
                              and index + 1 < budget and not self.host_tools)
                prepared = bool(prestart is not None and (
                    prestart.get("committed") or "".join(prestart.get("live") or []).strip()))
                # "end" closes the reply: nothing the other pass prepared is added.
                ending = speaker_intent == "end"
                more = bool((not ending and (beat.hold or beat.note or late_thought is not None or need_shift
                                             or prepared or speaker_intent in ("stay", "pass")))
                            or bg_hands is not None or self._interrupted or folded_inner
                            or (prestart is not None and (prestart.get("recourse") or prestart.get("committed"))))
                task_state = (res.raw or {}).get("task", "") if self._task_live else ""
                if (self._task_live and not task_state and not res.tool_calls
                        and (beat.text or "").rstrip().endswith("?")):
                    task_state = "ask"  # a question to them is waiting on them, marker or not
                peer_on_air = prestart is not None and prestart.get("committed")
                if task_state == "ask" or (task_state == "done" and reviewed):
                    more, task_finished, task_outcome = False, True, task_state
                elif task_state == "done" and not peer_on_air and index + 1 < self.page_cap:
                    # The other lobe checks the work before the turn may end.
                    reviewed = True
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    budget = max(budget, index + 2)
                    pending_note, pending_hold = PEER_REVIEW, ""
                    self._event({"type": "silent", "writing": other.key, "kind": "review", "phase": "start"})
                    current, other = (other, current) if self.public_lane is None else (current, other)
                    continue
                if bg_hands is not None and not beat.hold and not beat.note and late_thought is None:
                    # The reply would end, but the hands are still out: wait for
                    # them (that wait is a tool wait) and give one more beat.
                    if prestart is not None and not prestart.get("committed"):
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    if prestart is None:
                        try:
                            await bg_hands
                        except Exception:  # noqa: BLE001
                            pass
                        pending_note = self._land_hands(bg_hands, prefix, result)
                        bg_hands = None
                        pending_hold = ""
                        current, other = (other, current) if self.public_lane is None else (current, other)
                        continue
                if not more:
                    if (owed_answer and _status_only(beat.text) and status_nudges < _STATUS_NUDGES
                            and tool_rounds < self.max_tool_rounds):
                        # "Opening the docs." / "Reading." is the hands talking,
                        # not the answer. The screen changed; stay on it.
                        status_nudges += 1
                        self._event({"type": "silent", "writing": current.key, "kind": "continue",
                                     "phase": "status", "text": (beat.text or "")[:160]})
                        if index + 1 >= budget and budget < self.page_cap:
                            budget += 1
                        if index + 1 < budget:
                            if prestart is not None and not prestart.get("committed"):
                                await self._end_prestart(prestart, dropped=True)
                                prestart = None
                            pending_note = "\n".join(x for x in (
                                pending_note,
                                "The tool returned. Answer from its result, take the next step, or "
                                "explain what failed. A sentence that only says you are checking "
                                "is not the answer. Do not invent missing facts.",
                            ) if x)
                            pending_hold = ""
                            continue
                    if screen_plan is not None and not screen_plan.finished:
                        if screen_plan.holds < 4:
                            screen_plan.holds += 1
                            if index + 1 >= budget and budget < self.page_cap:
                                budget += 1
                            if index + 1 < budget:
                                if prestart is not None and not prestart.get("committed"):
                                    await self._end_prestart(prestart, dropped=True)
                                    prestart = None
                                pending_note = "\n".join(x for x in (
                                    pending_note,
                                    f"Subtask {screen_plan.index + 1} of {len(screen_plan.subtasks)} not finished yet: {screen_plan.current}. Work only on this one. Not done until the screen shows it. Emit ⟦subtask done⟧ privately when complete.",
                                ) if x)
                                pending_hold = ""
                                current, other = (other, current) if self.public_lane is None else (current, other)
                                continue
                    # A beat that only said the reply again is not progress to nudge on.
                    repeated = bool(beat.text) and not res.tool_calls and restates(lead, beat.text)
                    if (self._task_live and not task_finished and not peer_on_air and not self._interrupted
                            and task_nudges < _TASK_NUDGES and tool_rounds < self.max_tool_rounds
                            and index + 1 < self.page_cap and not repeated):
                        # Work started and nobody said it is done: the same lobe keeps going.
                        task_nudges += 1
                        if prestart is not None:
                            await self._end_prestart(prestart, dropped=True)
                            prestart = None
                        budget = max(budget, index + 2)
                        pending_note, pending_hold = KEEP_GOING, ""
                        self._event({"type": "silent", "writing": current.key, "kind": "continue", "phase": "task"})
                        continue
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    # Last chance for C: a wrong fact gets one correcting beat.
                    if verify_task is not None and index + 1 < budget:
                        try:
                            await asyncio.wait_for(asyncio.shield(verify_task), timeout=self.final_verify_wait)
                        except Exception:  # noqa: BLE001
                            pass
                        corrected = verify_task.done() and self._apply_verdict(verify_task, prefix, result)
                        if verify_task.done():
                            verify_task = None
                        if corrected:
                            pending_note, pending_hold = "The check found something — fold the correction into what you say next naturally. Do not announce it as a correction; just say the right thing.", ""
                            current, other = (other, current) if self.public_lane is None else (current, other)
                            continue
                    break  # nothing left to trade: the mind has finished
                # Only trade the microphone on a finished thought. A beat that
                # ran past its length without closing is spoken whole and the
                # same writing keeps going - the seam has to be a sentence.
                if not ends_on_sentence(beat.text):
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    pending_note = "\n".join(x for x in (carry_note, beat.note) if x)
                    pending_hold = beat.hold or carry_hold
                    carry_note, carry_hold = "", ""
                    continue

                if speaker_intent == "stay" and not (prestart and prestart.get("recourse")):
                    peer_ready = self._draft_excerpt(prestart)
                    if prestart is not None:
                        await self._end_prestart(prestart, dropped=True)
                        prestart = None
                    pending_note = "\n".join(x for x in (carry_note, beat.note,
                        ("More is already drafted to follow. Finish this point, then pass at a "
                         "completed thought; do not repeat yourself."
                         if peer_ready else "")) if x)
                    pending_hold = beat.hold or carry_hold
                    carry_note, carry_hold = "", ""
                    self._event({"type": "speaker", "writing": current.key,
                                 "kind": "continue", "index": index + 1})
                    continue

                if handoff_note and prestart is not None and not prestart.get("recourse"):
                    # A note written after preparation belongs in a fresh tail.
                    await self._end_prestart(prestart, dropped=True)
                    prestart = None
                    beat.note = "\n".join(x for x in (beat.note, handoff_note) if x)
                if prestart is not None:
                    # The other writing already started (without this beat's
                    # hold - it was still being written). The hold and note go
                    # to the beat after; the hold event already showed it.
                    carry_note, carry_hold = beat.note, beat.hold
                    pending_note, pending_hold = "", ""
                else:
                    pending_note = "\n".join(x for x in (carry_note, beat.note) if x)
                    pending_hold = beat.hold or carry_hold
                    carry_note, carry_hold = "", ""
                current, other = (other, current) if self.public_lane is None else (current, other)
            self._kernel_todo(user_text, result.tool_calls, finished="done" if task_outcome == "done" else (
                "clean" if not owed_answer and not result.interrupts and not self._interrupted else ""))
            if owed_answer and not result.interrupts and not self._interrupted:
                spoken = drop_dangling_bridge(spoken)
                spoken += (" " if spoken else "") + (
                    "I couldn't verify the result, so I can't call this finished.")
                await out(spoken)
            if bg_hands is not None:
                # The reply is over but a tool is still out (e.g. the browser,
                # after an interrupt re-aimed her). Let it finish - abandoning it
                # would leave the page mid-action - and record what it found.
                try:
                    await asyncio.wait_for(asyncio.shield(bg_hands), timeout=self.final_hands_wait)
                except Exception:  # noqa: BLE001
                    pass
                if bg_hands.done():
                    self._land_hands(bg_hands, prefix, result)
                    bg_hands = None
            if verify_task is not None and verify_task.done():
                self._apply_verdict(verify_task, prefix, result)
        finally:
            if prestart is not None:
                await self._end_prestart(prestart, dropped=True)
            if verify_task is not None and not verify_task.done():
                self._pending.append(verify_task)
                verify_task.add_done_callback(self._publish_late_verification)
            if late_thought is not None and not late_thought.done():
                late_thought.cancel()
            if bg_hands is not None and not bg_hands.done():
                stopped = any(i.get("kind") == "stop" for i in result.interrupts)
                if self.on_late_hands is not None and not stopped:
                    # C12: the job keeps going after the reply; she says when it is done.
                    self._pending.append(bg_hands)
                    bg_hands.add_done_callback(self._report_late_hands)
                    self._event({"type": "hands", "kind": "background", "phase": "start"})
                else:
                    bg_hands.cancel()
            # Silent thoughts still running when the reply ends are dropped -
            # and their "finished" event must go out BEFORE the stream closes,
            # or the app keeps showing "thinking..." for a thought that is gone.
            open_thoughts = [t for t in getattr(self, "_thoughts", []) if not t.done()]
            for t in open_thoughts:
                t.cancel()
            if open_thoughts:
                await asyncio.gather(*open_thoughts, return_exceptions=True)
            self._thoughts = []
            await out(None)

        result.text = drop_dangling_bridge(spoken)
        result.ms = (time.perf_counter() - t0) * 1000
        if spoken and self.caps.memory is not None and (user_text or not self._user_content):
            # Fire and forget. Keeping a memory is not allowed to delay a reply,
            # and a host that cannot store one right now is a host with no past
            # tomorrow, not a host with a broken answer now.
            task = asyncio.create_task(self._keep(user_text, spoken))
            self._pending.append(task)
        if (self.on_anticipate is not None and spoken and self.silent_thinks
                and not self.host_tools and not result.interrupts and user_text):
            self._pending.append(asyncio.create_task(self._anticipate(user_text, spoken)))
        return result

    def _report_late_hands(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None or self.on_late_hands is None:
            return
        calls = task.result()[0]
        self._event({"type": "hands", "kind": "background", "phase": "end",
                     "name": ",".join(c["name"] for c in calls)[:120]})
        try:
            self.on_late_hands(calls)
        except Exception:  # noqa: BLE001
            pass

    async def _anticipate(self, question: str, said: str) -> None:
        """A20. After the reply, the silent side guesses the next question and
        answers it, so a likely follow-up is already thought through."""
        prompt = (f"They said: {question[:600]}\n\nYou answered: {said[-900:]}\n\n"
                  "You are not speaking. Guess the one thing they are most likely to say next, and "
                  "answer it in at most two sentences. Reply exactly in two lines:\nQ: ...\nA: ...\n"
                  "If nothing is likely, reply: none")
        try:
            res = await self.llm.chat([{"role": "system", "content": self.right.tail},
                                       {"role": "user", "content": prompt}],
                                      temperature=0.3, max_tokens=140, model=self._instance(self.right))
        except Exception:  # noqa: BLE001
            return
        text = (res.text or "").strip()
        q = re.search(r"^\s*Q:\s*(.+)$", text, re.M)
        a = re.search(r"^\s*A:\s*(.+)$", text, re.M)
        if q and a and self.on_anticipate is not None:
            try:
                self.on_anticipate({"q": q.group(1).strip()[:300], "a": a.group(1).strip()[:500]})
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def _instance(self, w: Writing) -> str | None:
        """The served copy this writing speaks from. L is AL, R is AR."""
        return (self.models or {}).get(w.key) or None

    def _tail(self, w: Writing, note: str, hold: str, *, heard: str = "", drafted: str = "") -> str:
        """The only part of the request that differs between writings.

        ``drafted`` is the other lobe's live text, so this sentence is written
        in its presence. ``heard`` is what the mouth has already said, so the
        quiet lobe adds only what that left out. Both stay in the tail. The
        shared prefix above it does not change.
        """
        if self.public_lane == "R":
            return self._in_front(
                "[host agent inference] You are R, the host's working lane. "
                "Follow the host system, preserve its required output schema, and return "
                "one complete decision or tool call. Do not add conversation, narration, "
                "handoff markers or a second answer.\n" + note)
        head = ""
        if drafted:
            head += ("\n\n[what you will say right after this, already drafted]\n"
                     f"{drafted[:500]}\n"
                     "Do not say it here.")
        if heard:
            head += ("\n\n[your reply so far, in your own words]\n"
                     f"{heard[-800:]}\n"
                     "Pick up right after it. Do not restate it, re-list it or ask its question again.")
        if note:
            head += f"\n\n[private notes, not shown to them]\n{note}"
        if hold:
            head += "\n\n[the thought you are picking up - continue it, do not restart]\n" + hold
        if self._task_live and self._todo:
            head += "\n\n[your to-do list so far]\n" + "\n".join(
                ("[x] " if t["done"] else "[ ] ") + t["text"] for t in self._todo)
        elif self._task_live:
            head += ("\n\n[no to-do list yet: write your plan line as the steps of this task, "
                     "⟦plan: [x] what is done; [ ] what is next; [ ] what remains⟧]")
        return self._in_front(head + "\n\n" + w.tail + HANDOFF_PROTOCOL +
                              (TASK_PROTOCOL if self._task_live else "") +
                              "\nTheir request, including length, applies to the whole reply. "
                              "If it is already answered, write only ⟦done⟧.")

    def _orient(self, doing: str = "") -> str:
        fn = self.orient
        if fn is None:
            return ""
        try:
            return str(fn(doing) or "").strip()
        except Exception:  # noqa: BLE001 - a clock that fails is not a failed reply
            return ""

    def _in_front(self, text: str, doing: str = "") -> str:
        """The stopwatch is the last thing A reads before she writes."""
        line = self._orient(doing)
        if not line:
            return text
        return text + "\n\n" + line + "\n[not shown to them]"

    def _bridge_for(self, w: Writing) -> str:
        if not self.bridges or not self.early_handoff or getattr(self, "_terse", False):
            return ""
        options = BRIDGES.get(w.key, BRIDGES["R"])
        self._bridges = getattr(self, "_bridges", -1) + 1
        return options[self._bridges % len(options)]

    def _sentence_limit(self) -> int | None:
        """The model ends its thought; peer readiness is not a generation cap."""
        return None

    def _arm_draft(self, prefix, w, heard: str, out, next_index: int, note: str = "") -> dict:
        """Start the quiet lobe. No tools: the mouth is the only driver.

        ``heard`` is what the mouth has already said. It lives in the tail, so
        the shared prefix stays byte-identical.
        """
        msgs = list(prefix)
        msgs.append({"role": "user", "content": self._tail(w, note, "", heard=heard)})
        gate = _Gate(out)
        visible_done = asyncio.Event()
        prep_ready = asyncio.Event()
        live: list[str] = []
        lead_ref = {"v": ""}
        speak_id = ("draft", id(gate))
        plan_box: dict = {}
        hold_ref: dict = {"f": None}
        task = asyncio.create_task(self._speak(
            msgs, w, gate, lambda: lead_ref["v"], visible_done=visible_done, live=live,
            speak_id=speak_id, prep_ready=prep_ready, sentence_limit=self._sentence_limit(),
            with_tools=False, plan_box=plan_box,
            hold_for=lambda: hold_ref["f"] is None or hold_ref["f"]()))
        self._event({"type": "silent", "writing": w.key, "kind": "early_start", "phase": "start"})
        return {
            "writing": w.key, "msgs": msgs, "task": task, "gate": gate, "speak_id": speak_id,
            "kind": "early_start", "t0": time.perf_counter(),
            "visible_done": visible_done, "prep_ready": prep_ready, "live": live,
            "lead_ref": lead_ref, "lead_len": 0, "next_index": next_index,
            "refreshed": bool(heard), "plan_box": plan_box, "hold_ref": hold_ref,
        }

    async def _wait_draft(self, prestart: dict | None) -> None:
        """Give the quiet lobe a moment to produce a clause before the mouth starts."""
        if prestart is None or self.couple_timeout <= 0:
            return
        ready = asyncio.create_task(prestart["prep_ready"].wait())
        try:
            await asyncio.wait({ready, prestart["task"]}, timeout=self.couple_timeout,
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            await self._cancel(ready)

    @staticmethod
    def _draft_is_stale(prestart: dict, prefix: list[dict]) -> bool:
        """True when the shared mind gained something other than the sentence just said."""
        old = prestart["msgs"][:-1]
        if prefix[:len(old)] != old:
            return True
        return any(m.get("role") != "assistant" for m in prefix[len(old):])

    @staticmethod
    def _draft_excerpt(prestart: dict | None) -> str:
        if not prestart:
            return ""
        return "".join(prestart.get("live") or []).strip()[:500]

    async def _hand_whole(self, prestart, prefix, w, said, segment, thought, out, next_index) -> dict:
        """README.md rule 8. The segment is fully written before it is
        heard: the other lobe gets all of it and prepares while it is spoken.
        Spoken, a segment takes ~15 chars/s; that is how long its reply may take
        to find words before the earlier draft is used instead."""
        nxt = await self._refresh_draft(prestart, prefix, w, said, thought, out, next_index)
        try:
            await asyncio.wait_for(nxt["prep_ready"].wait(),
                                   timeout=min(len(segment) / 15.0, 8.0))
        except asyncio.TimeoutError:
            pass
        return nxt

    async def _refresh_draft(self, prestart, prefix, w, heard, thought, out, next_index) -> dict:
        """The fact has started. Draft the quiet half again against what was said.

        The earlier draft stays until this one has words. If the mouth reaches
        the seam first, the earlier words are what get said, so there is no gap.
        """
        note = ""
        if thought is not None and thought.done():
            note = self._done_text(thought)
            if note.lower().startswith("none"):
                note = ""
        nxt = self._arm_draft(prefix, w, heard, out, next_index, note=note)
        nxt["previous"] = prestart
        return nxt

    async def _choose_draft(self, prestart):
        """Keep the draft that can speak now. Prefer the one that saw the fact."""
        if prestart is None or not prestart.get("previous"):
            return prestart
        if not "".join(prestart.get("live") or []).strip():
            # The rewrite just started. An instant model finishes in this gap;
            # a slow one keeps the words it already had.
            try:
                await asyncio.wait_for(prestart["prep_ready"].wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass
        prev = prestart.pop("previous")
        if "".join(prestart.get("live") or []).strip():
            await self._end_prestart(prev, dropped=True)
            return prestart
        if "".join(prev.get("live") or []).strip():
            await self._end_prestart(prestart, dropped=True)
            prev["refreshed"] = True
            return prev
        await self._end_prestart(prev, dropped=True)
        return prestart

    def _arm_next(self, prefix, other, said, lead, thought, out, next_index) -> dict:
        started = self._seam_prestart(prefix, other, said, lead, thought, out)
        started["next_index"] = next_index
        return started

    async def _open_handoff(self, prestart: dict, lead: str, said_now: str) -> None:
        """Release the lobe that has been preparing, joined to the finished sentence."""
        if prestart.get("committed") or prestart.get("recourse"):
            return
        full = (lead + said_now).strip()
        full_lead = (full + " ") if full else (lead or "")
        ref = prestart.get("lead_ref")
        if ref is not None:
            ref["v"] = full_lead
        prestart["lead_len"] = len(full_lead)
        if not prestart.get("announced"):
            self._event({"type": "beat", "writing": prestart["writing"],
                         "index": prestart.get("next_index", 0), "chars": len(full_lead)})
            prestart["announced"] = True
        await self._end_prestart(prestart, dropped=False)
        # The mic is this lobe's now, so a cut-in lands on it rather than on
        # the one that just finished its sentence.
        self._speaking_id = prestart.get("speak_id")
        if self._cut_target is not None:
            self._cut_target = prestart.get("speak_id")
        live = "".join(prestart.get("live") or [])
        await prestart["gate"].release(full_lead + live if live else None)
        prestart["committed"] = True

    async def _speak(self, msgs: list[dict], w: Writing, out: Emitter, already, *,
                     visible_done: asyncio.Event | None = None,
                     live: list[str] | None = None,
                     speak_id: object = None,
                     prep_ready: asyncio.Event | None = None,
                     sentence_limit: int | None = None,
                     with_tools: bool = True,
                     bridge: str = "", plan_box: dict | None = None,
                     plan_ready: asyncio.Event | None = None,
                     hold_for=None, retry_empty: bool = True,
                     require_answer: bool = False) -> tuple[ChatResult, float, str]:
        """Streams so the person watches the beat arrive.

        Returns the visible result, the time-to-first-token, and whatever the
        pass wrote privately. Only visible text ever reaches ``out``.
        """
        started = time.perf_counter()
        ttft = 0.0
        acc: list[str] = live if live is not None else []
        calls: list[ToolCall] = []
        invented: list[str] = []
        finish = "stop"
        filt = HoldFilter()
        # The belt is per writing: the pass holding the microphone may press
        # things, the quiet one may only look. A draft of the other half is
        # called with no belt at all, so it cannot fire the browser while the
        # mouth is the one driving.
        tools = (self.caps.schemas_for(w.key) or None) if with_tools else None
        base = (lambda: already()) if callable(already) else (lambda: already)
        # A beat that continues a reply shows its first sentence only once it is
        # clear that sentence is not the one already said.
        # A beat reading a fresh tool result reports it even when it matches.
        after_tool = next((m.get("role") == "tool" for m in reversed(msgs)
                           if m.get("role") in ("assistant", "tool")), False)
        echo_hold = "" if (base() or "").strip() and not bridge and not after_tool else None
        meta = MetaFilter()
        echoed = False
        echo_text = ""
        cut = False
        capped = False
        worker = self.kv_generation_worker(w.key) if self.kv_generation_worker else self.llm
        stream = worker.stream(msgs, tools=tools, temperature=w.temperature,
                                 max_tokens=self.max_tokens, model=self._instance(w))
        bridged = False
        wrote = False
        opener: str | None = ""
        held: list[str] = []
        gate_tools = bool(tools) and self.act_gate and (self._acting or self._task_live)

        async def _bridge() -> None:
            # The floor is here but the thought is not: a short "hmm -" keeps
            # the stream alive instead of a silence between the two lobes.
            nonlocal bridged
            await asyncio.sleep(BRIDGE_AFTER_S)
            if not acc:
                acc.append(bridge)
                bridged = True
                await out(base() + bridge)

        bridge_task = asyncio.create_task(_bridge()) if bridge else None
        shown = 0
        drip_task: asyncio.Task | None = None

        def _ready() -> bool:
            return hold_for is None or hold_for()

        def _benign_suffix() -> bool:
            # A private feel/hold suffix cannot carry floor intent or a tool.
            return (filt.hidden and filt.floor not in ("stay", "end")
                    and bool(re.match(r"\s*(?:\u27e6\s*(?:feel|hold)\s*:|\[\[?feel\s*:)",
                                      filt.captured, re.I)))

        async def _emit(final: bool = False) -> None:
            # Shows what is written, minus a stretch kept back while the other
            # lobe has nothing to say yet. Only ever extends what was shown.
            nonlocal shown
            full = "".join(acc)
            limit = len(full)
            if not final and not _ready():
                limit = max(min(len(full), FIRST_SHOW), len(full) - STRETCH_CHARS)
                if limit < len(full):
                    sp = full.rfind(" ", shown, limit + 1)
                    limit = sp if sp > shown else shown
            if limit > shown:
                shown = limit
                await out(base() + full[:limit])

        async def _drip() -> None:
            nonlocal shown
            while True:
                full = "".join(acc)
                if shown >= len(full):
                    break
                if _ready():
                    await _emit(final=True)
                    break
                nxt = full.find(" ", shown + 1)
                shown = len(full) if nxt == -1 else nxt
                await out(base() + full[:shown])
                await asyncio.sleep(DRIP_S)
            # Only a private suffix means "the visible part is over" early; a
            # beat ending on a tool call must not hand the floor away.
            if visible_done is not None and _benign_suffix():
                visible_done.set()

        try:
            async for ev in stream:
                et = ev.get("type")
                if et == "text":
                    piece = ev.get("text") or ""
                    wrote = wrote or bool(piece.strip())
                    if piece and not ttft:
                        ttft = (time.perf_counter() - started) * 1000
                    visible = meta.feed(filt.feed(piece))
                    if filt.plan and plan_box is not None and "text" not in plan_box:
                        plan_box["text"] = filt.plan
                        if plan_ready is not None:
                            plan_ready.set()
                    if visible and bridge_task is not None:
                        await self._cancel(bridge_task)
                        bridge_task = None
                    if bridged and len(acc) == 1 and opener is not None:
                        # Held back until it is clear the lobe is not opening
                        # with a hesitation of its own.
                        opener += visible
                        rest = strip_interjection(opener)
                        if len(rest) < _OPENER_CHARS and not filt.hidden:
                            visible = ""
                        else:
                            visible, opener = _after_bridge(rest), None
                    elif visible and not acc and not echo_hold:
                        visible = visible.lstrip()
                    if visible and echo_hold is not None:
                        echo_hold, visible = echo_hold + visible, ""
                        if _sentence_end(echo_hold) != -1 or len(echo_hold) > 240:
                            if says_again(base(), echo_hold) or recaps(base(), echo_hold) or paraphrases(base(), echo_hold):
                                echoed, echo_text = True, echo_hold
                                break
                            visible, echo_hold = echo_hold, None
                    if gate_tools and visible:
                        shown_text = "".join(acc)
                        if held:
                            held.append(visible)
                            visible = ""
                        elif m := _FIRST_CLAUSE.search(shown_text + visible):
                            joined = shown_text + visible
                            stop = max(m.end(), len(shown_text))
                            visible = joined[len(shown_text):stop]
                            if joined[stop:]:
                                held.append(joined[stop:])
                    cutting = speak_id is not None and getattr(self, "_cut_target", None) == speak_id
                    if cutting:
                        # Cut in by the person: finish THIS sentence, then stop.
                        # A sentence that already ended does not get a successor.
                        if acc and ends_on_sentence("".join(acc)):
                            cut = True
                            if visible_done is not None:
                                visible_done.set()
                            break
                        # The sentence end is usually mid-chunk, so the chunk is
                        # trimmed at it - nothing past it is ever shown.
                        if not acc and not visible:
                            cut = True
                            break
                        end = _sentence_end(visible) if visible else -1
                        if end != -1 or (acc and ends_on_sentence("".join(acc)) and not visible):
                            if end != -1:
                                acc.append(visible[:end])
                                await out(base() + "".join(acc))
                            cut = True
                            if visible_done is not None:
                                visible_done.set()
                            break
                    if visible and not capped:
                        if sentence_limit:
                            merged = "".join(acc) + visible
                            kept, reached = _take_sentences(merged, sentence_limit, allow_final=False)
                            add = kept[len("".join(acc)):]
                            if add:
                                acc.append(add)
                            if reached:
                                capped = True
                            if add:
                                await _emit()
                        else:
                            acc.append(visible)
                            await _emit()
                    if prep_ready is not None and not prep_ready.is_set() and _enough_to_react("".join(acc)):
                        prep_ready.set()
                    if _benign_suffix() and hold_for is not None and drip_task is None:
                        drip_task = asyncio.create_task(_drip())
                    elif (_benign_suffix() and hold_for is None and visible_done is not None
                          and not visible_done.is_set()):
                        visible_done.set()
                elif et == "tool_call":
                    # A quiet draft is called with no belt. If the model emits a
                    # call anyway, it must not become a hand the host presses.
                    # A name nobody offered (an invented "calc") is not a hand
                    # either: running it only returns "unknown tool" and a
                    # beat that pretends work is happening.
                    offered = {t["function"]["name"] for t in tools or ()} | self.host_tools
                    if with_tools and (not tools or ev["call"].name in offered):
                        calls.append(ev["call"])
                    elif with_tools:
                        invented.append(ev["call"].name)
                elif et == "done":
                    finish = ev.get("finish_reason") or "stop"
            if not cut and bridged and len(acc) == 1 and opener:
                if rest := _after_bridge(strip_interjection(opener + meta.feed(filt.flush()) + meta.flush())):
                    acc.append(rest)
                    await _emit()
            if held and not cut:
                if calls:
                    filt.flush()  # said ahead of a result that does not exist yet
                    meta.flush()
                else:
                    acc.append("".join(held))
                    await _emit()
                held.clear()
        except BaseException:
            if drip_task is not None:
                await self._cancel(drip_task)
            raise
        finally:
            if bridge_task is not None:
                await self._cancel(bridge_task)
            if cut and drip_task is not None:
                await self._cancel(drip_task)
            if bridged and "".join(acc).strip() == bridge.strip():
                # The lobe had nothing after all. The bridge is already on the
                # screen and shown words are never taken back, so it stays part
                # of the reply and the next beat continues after it.
                acc.clear()
                if not cut:
                    self._shown_bridge = bridge.strip()
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass
        if cut:
            finish = "interrupted"
            if visible_done is not None:
                visible_done.set()
            self._cut_target = None
            self._last_private = ""
            return ChatResult(text="".join(acc), tool_calls=[], finish_reason=finish), ttft, ""
        # Drain the private/metadata filters once. A continuation can contain
        # a discarded filler sentence followed by a real answer: never lose it.
        tail = meta.feed(filt.flush()) + meta.flush()
        if echo_hold is not None and not echoed:
            pending, echo_hold = echo_hold + tail, None
            tail = ""
            if (meta.dropped and not pending.strip()) or (pending.strip() and
                    (says_again(base(), pending) or recaps(base(), pending)
                     or paraphrases(base(), pending))):
                echoed, echo_text = True, pending
            elif pending:
                acc.append(pending.lstrip() if not acc else pending)
                await _emit()
        if echoed:
            logging.getLogger(__name__).info("echo beat %s dropped: %r", w.key, echo_text[:200])
            acc.clear()
            self._event({"type": "silent", "writing": w.key, "kind": "echo", "phase": "beat"})
        if not capped and not echoed and tail:
            acc.append(tail.lstrip() if not acc else tail)
            await _emit()
        if invented:
            logging.getLogger(__name__).warning("dropped unoffered tool calls %s", invented)
        if (retry_empty and not acc and not calls and not echoed and (filt.plan.strip() or wrote or invented)
                and (require_answer or not re.search(r"⟦\s*(?:task\s*:\s*)?done", filt.captured)
                     or not base().strip())):
            retry_msgs = [*msgs, {"role": "user", "content": (
                "[not shown to them] No public answer was produced. Answer the original request "
                "from the conversation and tool results in ordinary visible sentences now. "
                "Do not write a private plan for this retry. Keep coordination notes private."
                + (f" There is no tool named {', '.join(sorted(set(invented)))}; use only the tools "
                   "you were given, or answer in words." if invented else ""))}]
            retry_started = time.perf_counter()
            res, retry_ttft, hold = await self._speak(
                retry_msgs, w, out, already, visible_done=visible_done, live=live,
                speak_id=speak_id, prep_ready=prep_ready, sentence_limit=sentence_limit,
                with_tools=with_tools, plan_box=plan_box, plan_ready=plan_ready,
                hold_for=hold_for, retry_empty=False)
            return res, (retry_started - started) * 1000 + retry_ttft, hold
        if hold_for is not None:
            if drip_task is None:
                drip_task = asyncio.create_task(_drip())
            await drip_task
        self._last_private = filt.captured
        if not acc:
            logging.getLogger(__name__).debug(
                "empty beat %s: finish=%s calls=%s plan=%r private=%r pending=%r", w.key, finish, [c.name for c in calls],
                filt.plan[:120], filt.captured[:300], filt.buf[:300])
        return ChatResult(text="".join(acc), tool_calls=calls, finish_reason=finish,
                          raw={"floor": filt.intent(), "handoff": filt.handoff(), "task": filt.task(),
                               "todo": filt.todo,
                               "private": filt.captured, "plan": filt.plan}), ttft, filt.hold()

    def _seam_prestart(self, prefix: list[dict], w: Writing, said_now: str, lead: str,
                       thought: asyncio.Task, out: Emitter) -> dict:
        """Start ``w``'s speaking call now, gated: its words are held until the
        current beat is confirmed to hand over, then released. Uses whatever
        the silent pass has ready this instant; the rest goes to the beat after."""
        note = self._done_text(thought) if thought.done() else ""
        if note.lower().startswith("none"):
            note = ""
        msgs = list(prefix)
        if said_now:
            msgs.append({"role": "assistant", "content": said_now})
        msgs.append({"role": "user", "content": self._tail(w, note, "")})
        gate = _Gate(out)
        visible_done = asyncio.Event()
        prep_ready = asyncio.Event()
        live: list[str] = []
        lead_ref = {"v": (lead + said_now + " ") if said_now else lead}
        speak_id = ("early", id(gate))
        task = asyncio.create_task(self._speak(
            msgs, w, gate, lambda: lead_ref["v"], visible_done=visible_done, live=live,
            speak_id=speak_id, prep_ready=prep_ready, sentence_limit=self._sentence_limit()))
        self._event({"type": "silent", "writing": w.key, "kind": "early_start", "phase": "start"})
        return {"writing": w.key, "msgs": msgs, "task": task, "gate": gate, "speak_id": speak_id,
                "kind": "early_start", "t0": time.perf_counter(),
                "visible_done": visible_done, "prep_ready": prep_ready, "live": live,
                "lead_ref": lead_ref, "lead_len": len(lead_ref["v"])}

    def _recourse_prestart(self, prefix: list[dict], w: Writing, said_so_far: str, words: str,
                           out: Emitter, speak_id: object) -> dict:
        """The person gave a new command mid-beat. The silent writing starts on
        it NOW, while the speaking writing finishes its current sentence. Gated:
        its words wait for that sentence to end, then flow with no gap."""
        msgs = list(prefix)
        if said_so_far:
            msgs.append({"role": "assistant", "content": said_so_far})
        msgs.append({"role": "user", "content": self._interruption_message(words)})
        msgs.append({"role": "user", "content": self._tail(w, (
            f"They just cut in with a new instruction: "
            f"{words or 'new original audio (attached above; no transcript)'}\n"
            "Switch to it immediately and smoothly - do not apologise, do not "
            "restart, do not repeat what was said. Keep anything from before "
            "that still helps; drop what no longer applies."
            + ("" if asks_for_terse(words) else " If the sentence they cut still matters after "
               "you answer them, come back to it in one short clause.")), "")})
        lead_ref = {"v": ""}
        gate = _Gate(out)
        visible_done = asyncio.Event()
        live: list[str] = []
        task = asyncio.create_task(self._speak(msgs, w, gate, lambda: lead_ref["v"],
                                               visible_done=visible_done, live=live, speak_id=speak_id))
        self._event({"type": "silent", "writing": w.key, "kind": "recourse", "phase": "start", "name": words[:120]})
        return {"writing": w.key, "msgs": msgs, "task": task, "gate": gate, "visible_done": visible_done,
                "live": live, "recourse": True, "lead_ref": lead_ref, "speak_id": speak_id, "words": words,
                "user_content": self._interruption_message(words),
                "t0": time.perf_counter(),
                "committed": True, "announced": True}

    async def _end_prestart(self, prestart: dict, *, dropped: bool) -> None:
        """Close a pre-started writing's "silent" label: it took the mic, or it
        was dropped (and its call cancelled). Never leaves "...ing" on screen."""
        if not prestart.get("ended"):
            prestart["ended"] = True
            kind = prestart.get("kind", "recourse" if prestart.get("recourse") else "early_start")
            if kind == "early_start" or dropped:
                self._event({"type": "silent", "writing": prestart["writing"], "kind": kind, "phase": "end",
                             "dropped": dropped,
                             "took_ms": round((time.perf_counter() - prestart.get("t0", time.perf_counter())) * 1000)})
        if dropped:
            await self._cancel(prestart["task"])

    def _on_public(self, chunk: str) -> None:
        after = getattr(self, "_respond_after", None)
        if after is not None and chunk[after:].strip():
            self._respond_after = None
            self._mark_responded()

    def _mark_responded(self) -> None:
        if getattr(self, "_interrupt_at", 0.0):
            self._event({"type": "interrupt", "kind": "responded",
                         "respond_ms": round((time.perf_counter() - self._interrupt_at) * 1000)})
            self._interrupt_at = 0.0

    def _track(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        if not hasattr(self, "_thoughts"):
            self._thoughts = []
        self._thoughts.append(task)
        return task

    @staticmethod
    async def _cancel(task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001
            pass

    @staticmethod
    def _done_text(task: asyncio.Task) -> str:
        try:
            return str(task.result() or "")
        except BaseException:  # noqa: BLE001
            return ""

    def _land_hands(self, task: asyncio.Task, prefix: list[dict], result: "DualResult") -> str:
        """Fold background tool results into the shared mind. Returns the note."""
        try:
            calls, tool_msgs = task.result()
        except BaseException:  # noqa: BLE001
            return ""
        if not calls:
            return ""
        tool_msgs = [m for m in tool_msgs if m.get("role") == "tool"] + \
            [m for m in tool_msgs if m.get("role") != "tool"]
        prefix.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": m["tool_call_id"], "type": "function",
             "function": {"name": m["name"], "arguments": json.dumps(c["args"])}}
            for c, m in zip(calls, tool_msgs)]})
        prefix.extend(tool_msgs)
        result.tool_calls.extend(calls)
        self._event({"type": "silent", "writing": "", "kind": "tool", "phase": "end",
                     "name": ",".join(c["name"] for c in calls)[:120]})
        return "The hands reported: " + "; ".join(f"{c['name']} -> {str(c['result'])[:300]}" for c in calls)

    def _kernel_todo(self, request: str, calls: list[dict], *, finished: str = "") -> None:
        """The checklist for a model that never writes ``⟦todo:⟧`` itself.

        ``finished`` is "done" (the task was marked done), "clean" (the turn
        ended with an answer: the parts that are said count once every fetched
        part is in) or "".
        """
        if self._todo_by_model or not self._task_live:
            return
        if not self._todo_parts:
            parts = task_parts(request)
            self._todo_parts = parts if len(parts) >= 2 else task_parts(self._last_plan)
            if len(self._todo_parts) < 2:
                self._todo_parts = []
                return
        items = tool_checklist(self._todo_parts, calls, request=request)
        if finished:
            closed = tool_checklist(self._todo_parts, calls, request=request, finished=True)
            fetched = all(a["done"] for a, b in zip(items, closed) if a["done"] == b["done"])
            if finished == "done" or fetched:
                items = closed
        if items != self._todo:
            self._todo = items
            self._event({"type": "todo", "writing": "", "source": "kernel", "items": items})

    def _check(self, said: str, question: str, calls: list[dict]):
        """C's check, with the turn's tool results when the verifier takes them."""
        verify = self.caps.verify.verify
        evidence = tool_evidence(calls)
        if evidence and _takes(verify, "evidence"):
            return verify(said=said, question=question, evidence=evidence)
        return verify(said=said, question=question)

    def _apply_verdict(self, task: asyncio.Task, prefix: list[dict], result: "DualResult") -> bool:
        try:
            value = task.result()
        except BaseException:
            self._event({"type": "verify", "state": "UNRATED", "level": "", "risk": None})
            return False
        level, risk = "", None
        if isinstance(value, VerificationResult):
            verdict = value.note.strip()
            level, risk = value.level, value.risk
            self._event({"type": "verify", "state": value.state, "level": value.level,
                         "risk": value.risk, "basis": value.basis,
                         "finding_count": value.finding_count, "correction": verdict[:300]})
        else:
            verdict = str(value or "").strip()
            if verdict:
                self._event({"type": "verify", "state": "READY", "level": "FLAGGED",
                             "correction": verdict[:300]})
        # The level belongs to the meter. Only a concrete fix, or a RED on work
        # done with tools, reaches the next beat - and never as an announcement.
        tool_red = level == "RED" and bool(result.tool_calls)
        if not verdict and not tool_red:
            return False
        i = getattr(task, "beat_index", len(result.beats) - 1)
        if verdict and 0 <= i < len(result.beats):
            result.beats[i].verdict = verdict
        self._steer_next = True  # a failed check is a critical moment
        note = "[not shown to them]"
        if verdict:
            note += (" A check found a wrong fact in what was just said:\n" + verdict +
                     "\nUse the right fact in what you say next, plainly, as if it were the first "
                     "time. Do not announce a correction, apologise, or open with a hesitation. "
                     "Never correct what the person said.")
        if tool_red and self._task_live:
            # Mid-task the check often sees the conversation without the pages
            # the hands fetched; stopping there abandoned real work half done.
            note += ("\nIn this task: check what you said against the tool results above. Anything "
                     "no result shows, get it with a tool now or leave it out. Keep working; do not "
                     "announce a correction.")
        elif tool_red:
            # A bare note reads as "try harder", so she re-attempts the same
            # impossible thing: the meter climbs (fabrication -> task
            # substitution) while the tool rounds drain and the person watches
            # "pulling it up..." forever. A RED has to name the way out.
            note += ("\nThis was rated RED" + (f" (risk {risk}/100)" if risk is not None else "")
                     + ". Stop this line of attempt. Do not call another tool for it and do not "
                       "try it again a different way. This is your LAST beat for this turn - say "
                       "plainly, in a sentence or two, what you actually did, what you cannot do "
                       "from where you are and why, and what they would have to do themselves. "
                       "Do not describe progress. Do not open with a hesitation marker like "
                       "'wait' or 'actually'. One clean beat and the turn ends.")
            # Mechanical teeth: _red_stop makes the beat loop exit after this
            # beat lands. Without it, the recourse is advisory and the model
            # keeps generating "wait - actually - I was wrong" cycles.
            self._red_stop = True
        prefix.append({"role": "user", "content": note})
        return True

    def _publish_late_verification(self, task: asyncio.Task) -> None:
        if task in self._pending:
            self._pending.remove(task)
        try:
            value = task.result()
        except BaseException:
            self._event({"type": "verify", "state": "UNRATED", "level": "", "risk": None})
            return
        if isinstance(value, VerificationResult):
            self._event({"type": "verify", "state": value.state, "level": value.level,
                         "risk": value.risk, "basis": value.basis,
                         "finding_count": value.finding_count, "correction": value.note[:300]})
        else:
            verdict = str(value or "").strip()
            self._event({"type": "verify", "state": "READY" if verdict else "UNRATED",
                         "level": "FLAGGED" if verdict else "", "risk": None,
                         "correction": verdict[:300]})

    async def _narrate(self, prefix: list[dict], w: Writing, res: ChatResult, out: Emitter,
                       spoken: str, busy: "Callable[[], bool] | None" = None) -> str:
        """K2 / BR3: talk while the hands are gone.

        The writing that is NOT waiting on the tool takes the mic for one short
        clause ("pulling it up..."). It may describe what it is doing or what it
        already knows; it may not invent the result. Streams through ``out`` so
        the person hears it *during* the wait. While ``busy()`` stays true (a
        slow page), it adds a fresh clause about what it can see each time the
        last one has been heard, at most ``narrate_repeats`` more.
        """
        if not self.narrate_tools:
            return ""
        wanted = ", ".join(f"{c.name}({c.arguments})" for c in res.tool_calls)[:300]
        # One status clause per action: a beat that already said what it is
        # doing is not narrated a second time. A long wait still gets what is
        # new - what she can see - once that line has been heard.
        if before := (res.text or "").strip():
            if busy is None:
                return ""
            return await self._narrate_more(prefix, w, wanted, out, spoken + " ", "", busy,
                                            took=0.0, before=before)
        t = time.perf_counter()
        self._event({"type": "silent", "writing": w.key, "kind": "narrate", "name": wanted[:120],
                     "phase": "start"})
        msgs = list(prefix)
        msgs.append({"role": "user", "content": self._in_front(
            f"[not shown to them] You just started: {wanted}. It is still running. Say one short "
            "clause about what you are doing, as a person mid-task would, using only what you have "
            "seen so far. Continue right after what you already said.",
            doing=f"while {wanted[:80]} runs")})
        acc: list[str] = []
        filt = HoldFilter()
        lead = " "
        try:
            async for ev in self.llm.stream(msgs, tools=None, temperature=w.temperature,
                                            max_tokens=self.narrate_max_tokens, model=self._instance(w)):
                if ev.get("type") == "text":
                    visible = filt.feed(ev.get("text") or "")
                    if visible and not acc:
                        visible = visible.lstrip()
                    if visible:
                        acc.append(visible)
                        await out(spoken + lead + "".join(acc))
        except Exception:  # noqa: BLE001 - a narration that fails is just silence
            # Whatever already streamed stays the caller's to extend.
            return "".join(acc).rstrip()
        finally:
            self._event({"type": "silent", "writing": w.key, "kind": "narrate", "phase": "end",
                         "took_ms": round((time.perf_counter() - t) * 1000)})
        if tail := filt.flush():
            acc.append(tail.lstrip() if not acc else tail)
        # Exactly what was streamed: the emitter is monotonic, so the caller's
        # next buffer must extend this byte for byte.
        text = "".join(acc).rstrip()
        if text:
            await out(spoken + lead + text)
            if busy is not None:
                text = await self._narrate_more(prefix, w, wanted, out, spoken + lead, text, busy,
                                                took=time.perf_counter() - t)
        return text

    async def _narrate_more(self, prefix: list[dict], w: Writing, wanted: str, out: Emitter,
                            head: str, text: str, busy: "Callable[[], bool]", *, took: float,
                            before: str = "") -> str:
        """The rest of a long wait: one more honest clause each time the last
        has been heard. ``before`` is what the beat that pressed the tool
        already said. Returns everything streamed after ``head``."""
        started = time.perf_counter() - took
        last = text
        for n in range(1, self.narrate_repeats + 1):
            # Ask early enough that the next clause is ready as the last one ends.
            hold = (max(self.narrate_gap, spoken_seconds(last) - took) if last
                    else self.narrate_gap)
            if not await self._while_busy(busy, hold):
                break
            seen = await self._fresh_look(w, wanted)
            asked = time.perf_counter()
            self._event({"type": "silent", "writing": w.key, "kind": "narrate", "name": wanted[:120],
                         "phase": "start", "again": n})
            said = " ".join(x for x in (before, text) if x)
            msgs = list(prefix)
            msgs.append({"role": "user", "content": self._in_front(
                f"[not shown to them] {wanted} is still running ({asked - started:.0f} s so far). "
                f"While it ran you already said: \"{said[-400:]}\"\n"
                + (f"What the page or screen showed when you last looked (read-only):\n{seen[:1200]}\n"
                   if seen else "")
                + "Say ONE more short clause to them now, something new: what you can already see, "
                "or what you are about to check. Never state the answer or any result you have not "
                "seen, and nothing you already said.",
                doing=f"while {wanted[:80]} runs")})
            clause_task = asyncio.ensure_future(self._clause(msgs, w))
            while not clause_task.done():
                await asyncio.wait({clause_task}, timeout=0.1)
                if not clause_task.done() and not busy():
                    # The result is in: the next beat speaks from it now.
                    clause_task.cancel()
                    await asyncio.gather(clause_task, return_exceptions=True)
                    break
            clause = ""
            if clause_task.done() and not clause_task.cancelled() and clause_task.exception() is None:
                clause = clause_task.result()
            self._event({"type": "silent", "writing": w.key, "kind": "narrate", "phase": "end",
                         "again": n, "took_ms": round((time.perf_counter() - asked) * 1000)})
            if not clause or _squash(clause) in _squash(said):
                break
            text = f"{text} {clause}" if text else clause
            await out(head + text)
            last, took = clause, time.perf_counter() - asked
        return text

    async def _while_busy(self, busy: "Callable[[], bool]", seconds: float) -> bool:
        """Waits up to ``seconds``; False as soon as the wait is over."""
        end = time.perf_counter() + seconds
        while busy():
            left = end - time.perf_counter()
            if left <= 0:
                return True
            await asyncio.sleep(min(0.1, left))
        return False

    async def _fresh_look(self, w: Writing, wanted: str) -> str:
        """What the page shows now; one look covers a few seconds of narration."""
        at, seen = getattr(self, "_last_seen", (0.0, ""))
        if time.perf_counter() - at > 4.0:
            seen = await self._observe(w, wanted)
        return seen

    async def _clause(self, msgs: list[dict], w: Writing) -> str:
        filt = HoldFilter()
        parts: list[str] = []
        async for ev in self.llm.stream(msgs, tools=None, temperature=w.temperature,
                                        max_tokens=self.narrate_max_tokens, model=self._instance(w)):
            if ev.get("type") == "text":
                parts.append(filt.feed(ev.get("text") or ""))
        parts.append(filt.flush())
        return " ".join("".join(parts).split())

    async def _hands(self, msgs: list[dict], res: ChatResult) -> tuple[list[dict], list[dict]]:
        """Runs what the speaking pass asked for. A failed hand is a report, not
        a crash - the voice keeps going either way.

        Returns the calls plus the ``tool`` messages, so the caller can fold the
        findings into the shared prefix.
        """
        assert self.caps.tools is not None
        calls: list[dict] = []
        tool_msgs: list[dict] = []
        images: list[str] = []
        msgs.append({"role": "assistant", "content": res.text or None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in res.tool_calls]})
        async def one(tc) -> tuple:
            args = tc.args()
            try:
                result = await self.caps.tools.run(tc.name, args)
            except Exception as exc:  # noqa: BLE001
                result = f"the {tc.name} attempt failed: {exc}"
            return tc, args, str(result)

        finished: list[tuple] = []
        for group in _hand_groups(list(res.tool_calls)):
            if len(group) == 1:
                finished.append(await one(group[0]))
                continue
            # Several look-ups at once: one wait instead of a queue of them.
            t = time.perf_counter()
            names = ",".join(tc.name for tc in group)[:120]
            self._event({"type": "hands", "kind": "parallel", "phase": "start", "name": names, "n": len(group)})
            finished.extend(await asyncio.gather(*(one(tc) for tc in group)))
            self._event({"type": "hands", "kind": "parallel", "phase": "end", "name": names, "n": len(group),
                         "took_ms": round((time.perf_counter() - t) * 1000)})
        for tc, args, result in finished:
            result, image = _split_image(result, see=self.vision)
            calls.append({"name": tc.name, "args": args, "result": result})
            tool_msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})
            if image:
                images.append(image)
        # C10: an action's receipt is not the screen. Read it again inside the
        # same wait, so the next beat - and the proof gate - see the real result.
        reread = _reground_name(list(res.tool_calls), self._tool_names()) if self.reground else ""
        if reread:
            t = time.perf_counter()
            self._event({"type": "hands", "kind": "reground", "phase": "start", "name": reread})
            try:
                seen = str(await self.caps.tools.run(reread, {}))
            except Exception:  # noqa: BLE001 - a missed reading is not a failed action
                seen = ""
            seen, image = _split_image(seen, see=self.vision)
            ok = bool(seen.strip()) and not _failed(seen)
            self._event({"type": "hands", "kind": "reground", "phase": "end", "name": reread, "ok": ok,
                         "took_ms": round((time.perf_counter() - t) * 1000)})
            if ok:
                calls.append({"name": reread, "args": {}, "result": seen, "auto": True})
                tool_msgs.append({"role": "user", "content": (
                    "[the screen right after that, read again automatically - not shown to them]\n"
                    + seen[:4000])})
            if image:
                images.append(image)
        # Tool messages are text-only in the chat API, so a screenshot rides in
        # a user message right after the results - only for a model that can see.
        for img in images:
            tool_msgs.append({"role": "user", "content": [
                {"type": "text", "text": "[the screenshot you just took - not shown to them]"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]})
        msgs.extend(tool_msgs)
        return calls, tool_msgs

    def _splits(self) -> bool:
        """Whether a round may run as two parts: task mode, own hands, no visiting agent."""
        return bool(self.parts and self.persist and self.caps.tools is not None and not self.host_tools)

    def _tool_names(self) -> set[str]:
        try:
            return {t["function"]["name"] for t in self.caps.schemas_for("L")}
        except Exception:  # noqa: BLE001
            return set()

    async def _prefetch(self, w: Writing, said: str, question: str, res: ChatResult, remembered: str,
                        *, want_plan: bool = False) -> str:
        t = time.perf_counter()
        names = ",".join(c.name for c in res.tool_calls)[:120]
        self._event({"type": "silent", "writing": w.key, "kind": "tool", "name": names, "phase": "start"})
        try:
            return await self._prefetch_inner(w, said, question, res, remembered, want_plan=want_plan)
        finally:
            self._event({"type": "silent", "writing": w.key, "kind": "tool", "name": names, "phase": "end",
                         "took_ms": round((time.perf_counter() - t) * 1000)})

    async def _prefetch_inner(self, w: Writing, said: str, question: str, res: ChatResult,
                              remembered: str, *, want_plan: bool = False) -> str:
        """What the silent pass does with the time the hands are gone.

        The ordinary silent think asks "what should be said next". During a tool
        wait there is a sharper question, and it is the one the next beat needs
        first: what does this call bring back, and what does the mind do about it
        the moment it lands. Answering it *before* the result arrives is the
        whole saving - the alternative is the model sitting still for a second
        and a half and then starting to think.

        The answer is advisory. It is written without the result, so the pass
        that reads it has the result and is told to prefer what it actually sees.
        """
        if not self.tool_prefetch:
            return ""
        wanted = "; ".join(f"{c.name}({c.arguments})" for c in res.tool_calls)[:400]
        # BR1 - waiting = looking. While the hands are on the page/screen, the
        # silent writing LOOKS at it (read-only): did the last action do what was
        # intended? A modal, an error, a login wall, a spinner? A single-brain
        # agent is frozen here; this writing is not the one holding the tool.
        seen = await self._observe(w, wanted)
        note = ""
        if self.caps.insight is not None:
            try:
                note = str(await self.caps.insight.insight(mode=w.key, said=said, question=question) or "")[:600]
            except Exception:  # noqa: BLE001
                note = ""
        # Parallel memory recall — pull a second angle while the tool runs.
        # The initial recall used the raw question; this one uses what was said
        # so far + what the tool is doing, surfacing different memories for R.
        if self.caps.memory is not None and said:
            try:
                alt_query = f"{said[-200:]} {wanted[:100]}"
                alt_lines = await asyncio.wait_for(
                    self.caps.memory.recall(alt_query, limit=3),
                    timeout=self.memory_timeout,
                )
                alt_text = "\n".join(str(x) for x in (alt_lines or []) if str(x).strip())[:600]
                if alt_text and alt_text not in remembered:
                    remembered = f"{remembered}\n{alt_text}" if remembered else alt_text
            except Exception:  # noqa: BLE001 - a second recall is a bonus
                pass
        if not self.silent_thinks:
            return note[:400]

        prompt = f"They said: {question}\n\nWhat has been said so far:\n{(said or '(nothing yet)')}"
        if remembered:
            prompt += f"\n\nWhat you remember about them:\n{remembered}"
        if note:
            prompt += f"\n\nWhat you have been turning over while they talked:\n{note}"
        if seen:
            prompt += ("\n\nWhat is on the screen RIGHT NOW, while that action runs (you looked; it is "
                       f"read-only):\n{seen}\nIf it shows the action is going wrong (error, modal, wrong "
                       "page, login wall), say what to do instead.")
        prompt += (
            f"\n\nYou are not speaking. The pass that spoke just asked for: {wanted}\n"
            "It has not come back yet. In at most two sentences, say what it will "
            "probably bring back and what should be done the instant it does - "
            "including what to do if it fails or says something unexpected. "
            "If you can predict what comes back, add a line EXPECT: <the few words the "
            "result will contain>. "
            "You cannot press the buttons yourself; you are telling the next pass "
            "what you would press. If the next screen action is obvious, end with "
            'NEXT {"name":"browser_click","arguments":{"index":N,"text":"label"}} - '
            "or the matching desktop hand (desktop_click, desktop_type_text, desktop_key, "
            "desktop_scroll) when the task is on her desktop. Always include the visible "
            "text so the press re-grounds against the live screen. "
            "If there is nothing to prepare, reply: none")
        # The plan, asked for at the one moment it is free. NeuralAgent's planner
        # is a blocking stage before the first press; here the same question goes
        # to the pass that is already awake beside the hands, so the plan costs
        # no wall clock. Without this ask the model never volunteers the JSON,
        # ScreenPlan is never built, and the proof gate never runs.
        if want_plan and any(_screen_tool(c.name) for c in res.tool_calls):
            prompt += (
                "\n\nThere is no plan for this screen task yet. Write one first, on its own "
                'line: {"subtasks": ["...", "..."]} - the goals in order, at most six.\n'
                'Each subtask is a logical user action with a visible outcome - "post the '
                'update on LinkedIn", not "click the share box". One goal per subtask, each '
                "self-contained, no granular clicks. A subtask closes only when the screen "
                "shows it, so write what the screen will show when it is done, not which "
                "buttons get pressed on the way. Then continue with the two sentences above.")
            asked_plan = True
        else:
            asked_plan = False
        try:
            # A tool wait is always a critical moment: the steer model, if any.
            text = await self._silent_chat(w, prompt, steer=True, plan_room=asked_plan)
        except Exception:  # noqa: BLE001 - a prefetch that fails costs nothing
            return self._keep_alerts("", note)
        if not text or text.lower().startswith("none"):
            return self._keep_alerts("", note)
        return self._keep_alerts(text, note)

    async def _observe(self, w: Writing, wanted: str) -> str:
        """Read-only look at the page/screen during a tool wait. "" when the host
        has nothing to look at."""
        tools = self.caps.tools
        look = getattr(tools, "observe", None) if tools is not None else None
        if look is None:
            return ""
        t = time.perf_counter()
        self._event({"type": "silent", "writing": w.key, "kind": "observe", "name": wanted[:120], "phase": "start"})
        try:
            seen = str(await asyncio.wait_for(look(), timeout=self.thought_timeout) or "")[:2500]
        except Exception:  # noqa: BLE001 - looking is a bonus, never a blocker
            seen = ""
        if seen:
            self._last_seen = (time.perf_counter(), seen)
        self._event({"type": "silent", "writing": w.key, "kind": "observe", "phase": "end",
                     "took_ms": round((time.perf_counter() - t) * 1000), "saw": seen[:200]})
        return seen

    async def _think(self, w: Writing, said: str, question: str, *, remembered: str = "",
                     steer: bool = False) -> str:
        t = time.perf_counter()
        self._event({"type": "silent", "writing": w.key, "kind": "think", "phase": "start",
                     "steer": bool(steer and self.steer_llm is not None)})
        try:
            return await self._think_inner(w, said, question, remembered=remembered, steer=steer)
        finally:
            me = asyncio.current_task()
            dropped = bool(me is not None and me.cancelling())
            self._event({"type": "silent", "writing": w.key, "kind": "think", "phase": "end",
                         "took_ms": round((time.perf_counter() - t) * 1000), "dropped": dropped})

    async def _think_inner(self, w: Writing, said: str, question: str, *, remembered: str = "",
                           steer: bool = False) -> str:
        """What the silent pass does with the time it has. Never blocks speech.

        The host's note is *material for* the thought, not a replacement for it -
        the emotional pass contemplates about what the psychological read found,
        the factual one about what memory just surfaced. That is the difference
        between a mind that receives context and one that thinks with it.

        ``silent_thinks=False`` skips the model call and passes the host's note
        through untouched, for hosts on a rate limit they cannot afford to spend.
        """
        note = ""
        if self.caps.insight is not None:
            try:
                note = str(await self.caps.insight.insight(mode=w.key, said=said, question=question) or "")[:1200]
            except Exception:  # noqa: BLE001
                note = ""
        if not self.silent_thinks:
            return note[:400]

        said_or_nothing = said or "(nothing yet)"
        heard = question or ("(their original audio is attached; there is no transcript)"
                             if self._has_media() else "")
        prompt = f"They said: {heard}\n\nWhat has been said so far:\n{said_or_nothing}"
        if remembered:
            # The quiet side thinks *with* the past rather than receiving a
            # summary of it: the emotional pass reads it for what the person
            # has not gotten over, the factual pass for what it makes relevant.
            prompt += f"\n\nWhat you remember about them:\n{remembered}"
        if note:
            prompt += f"\n\nWhat you have been turning over while they talked:\n{note}"
        prompt += ("\n\nYou are not speaking. In at most two sentences, note privately the one "
                   "thing worth adding to this reply that has not been said yet: a memory, "
                   "missing context, or the next step. Do not repeat what is above. "
                   "If there is nothing, reply: none")
        try:
            text = await self._silent_chat(w, prompt, steer=steer)
        except Exception:  # noqa: BLE001
            return self._keep_alerts("", note)
        if not text or text.lower().startswith("none"):
            return self._keep_alerts("", note)
        return self._keep_alerts(text, note)

    async def _silent_chat(self, w: Writing, prompt: str, *, steer: bool, plan_room: bool = False) -> str:
        """One silent-pass call. In steer mode, critical moments go to the
        expensive model with a tight budget and a directive-only instruction.

        ``plan_room`` is for the one call that must also return a structure the
        kernel parses. The director's 25-word muzzle and its 90-token budget are
        right for a directive and wrong for a six-line plan, so both are lifted
        for that call only - the directive itself stays just as short.
        """
        prompt = self._in_front(prompt, "thinking")
        # A director that cannot hear the original would steer from a label;
        # the lobe that heard it keeps the critical moment instead.
        can_hear = not self._has_media() or getattr(self.steer_llm, "supports_native_audio", False)
        if steer and self.steer_llm is not None and can_hear:
            protocol = STEER_PROTOCOL + (PLAN_EXEMPTION if plan_room else "")
            res = await self.steer_llm.chat(
                [{"role": "system", "content": w.tail + protocol},
                 {"role": "user", "content": self._with_content(prompt)}],
                temperature=0.3,
                max_tokens=(self.steer_max_tokens + _PLAN_TOKENS) if plan_room else self.steer_max_tokens)
            text = (res.text or "").strip()
            self.steer_calls += 1
            self.steer_chars += len(text)
            if text and not text.lower().startswith("none"):
                self._event({"type": "steer", "writing": w.key, "text": text[:300]})
                return "DIRECTION (follow it): " + text
            return text
        res = await self.llm.chat(
            [{"role": "system", "content": w.tail},
             {"role": "user", "content": self._with_content(prompt)}],
            temperature=w.temperature, max_tokens=160 + (_PLAN_TOKENS if plan_room else 0),
            model=self._instance(w))
        return (res.text or "").strip()

    def _event(self, ev: dict) -> None:
        if self.on_event is None:
            return
        ev.setdefault("ms", round((time.perf_counter() - self._t0) * 1000))
        try:
            self.on_event(ev)
        except Exception:  # noqa: BLE001 - showing the lobes never breaks the turn
            pass

    async def _recall(self, question: str) -> str:
        """What the past has to say about this question. "" when there is no past."""
        if self.caps.memory is None:
            return ""
        try:
            lines = await self.caps.memory.recall(question)
        except Exception:  # noqa: BLE001 - a dead database is a mind with no past
            return ""
        return "\n".join(str(x) for x in (lines or []) if str(x).strip())[:1200]

    async def _keep(self, question: str, said: str) -> None:
        try:
            await self.caps.memory.remember(f"They said: {question[:300]}\nI said: {said[:700]}")
        except Exception:  # noqa: BLE001
            pass

    async def _collect(self, task: asyncio.Task, timeout: float | None = None) -> str:
        limit = self.thought_timeout if timeout is None else timeout
        if task.done():
            try:
                return task.result()
            except Exception:  # noqa: BLE001
                return ""
        # The speech finished first. Give the thought a moment, then let it go -
        # a note that arrives late is worth less than a reply that arrives now.
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=limit)
        except Exception:  # noqa: BLE001
            task.cancel()
            return ""

    def as_dict(self) -> dict:
        return {
            "type": "sawii.dual_loop",
            "writings": [self.left.key, self.right.key],
            "model": getattr(self.llm, "model", None) or getattr(self.llm, "name", "?"),
            "capabilities": self.caps.describe(),
            "steer_model": (getattr(self.steer_llm, "model", None) or getattr(self.steer_llm, "name", None))
                           if self.steer_llm is not None else None,
        }


class _Gate:
    """Holds a pre-started beat's output until the seam is confirmed."""

    def __init__(self, out: Emitter):
        self.out = out
        self.open = False
        self.held = None
        self.on_first = None   # called once, when the first words pass after release

    async def __call__(self, chunk) -> None:
        if self.open:
            if self.on_first is not None:
                cb, self.on_first = self.on_first, None
                cb()
            await self.out(chunk)
        else:
            self.held = chunk

    async def release(self, override: str | None = None) -> None:
        if self.open:
            return
        self.open = True
        held, self.held = self.held, None
        if override is not None:
            if held is not None or override:
                await self.out(override)
        elif held is not None:
            await self.out(held)


async def _silent(_chunk) -> None:
    return None


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _raw_index(text: str, n: int) -> int:
    """Index in text just past its first n characters once whitespace is squashed."""
    i = count = 0
    while i < len(text) and count < n:
        if text[i].isspace():
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            if count:
                count += 1
            i = j
        else:
            count += 1
            i += 1
    return i


def _monotonic(emit: Emitter, on_emit=None) -> Emitter:
    """Wraps a host emitter so it can never be shown a shorter reply than last.

    A beat is stripped and re-joined as it becomes part of the whole reply, so
    without this a handoff could flash a character or two backwards. The person
    watching a growing sentence must never watch it shrink.
    """
    state = {"last": ""}

    async def send(chunk) -> None:
        if chunk is None:
            await emit(None)
            return
        chunk = re.sub(r"[\r\n]\s*$", "", chunk).lstrip()
        last = state["last"]
        if not chunk.startswith(last):
            # The same words re-joined with different spacing (a stripped
            # beat, a paragraph break) still continue what is on screen.
            said = _squash(last)
            if not _squash(chunk).startswith(said):
                return
            rest = chunk[_raw_index(chunk, len(said)):]
            chunk = last + (rest.lstrip() if last[-1:].isspace() else rest)
        if len(chunk) > len(last):
            state["last"] = chunk
            await emit(chunk)
            if on_emit is not None:
                on_emit(chunk)

    return send


__all__ = ["Beat", "DualLoop", "DualResult", "ends_on_sentence"]
