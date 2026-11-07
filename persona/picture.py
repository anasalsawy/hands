"""The picture: everything, at three depths, small enough to be read every step.

Not "the big picture". It is the whole thing, drawn at once - what is directly in
front of us, what is around it, and what is true whatever happens next. The
reason it has depth is that these three change at wildly different speeds, and
one blob updated at one speed would either be stale or expensive:

    NOW      this minute. The line of work running, what they just said, how
             they are right now. Written by CODE, from evidence, every step.
    HERE     this stretch. The project's parts, the affairs still open in their
             life. Rewritten when something finishes or opens.
    ALWAYS   who they are, who is around them, what they asked to be held to.
             Rewritten rarely. Lives in the CACHED prefix of the prompt, so
             knowing all of it costs nothing per message.

The problem this solves is not forgetting. A model working on one file does not
forget the other files - it was never shown them. Attention follows the last few
tool results, so everything else stops existing. Same on the personal side: a
model deep in a task stops being a companion mid-sentence. The picture is what
stops existing from happening.

Two rules, and they are the whole design:

1. FACTS COME FROM EVIDENCE, NOT FROM ASSURANCE. Rows under DONE, BROKEN and
   WORKING are written by code, from line status and real tool results. A model
   cannot type "done" here; it has to finish the work and let the result say so.
   The same instinct as ClaimGate in inference_lanes.py. A model may only write
   the parts that are genuinely its judgment: the goal, the open questions, and
   which areas nobody has looked at.

2. IT IS A PICTURE, NOT A PROSE SUMMARY. Fixed sections, same order every time,
   one line each. A summary drifts; a form does not. An edit is a diff on one
   line, not a rewrite of an essay. Anything the UI wants to draw as a real
   diagram draws from these rows, so no model ever spends a call generating an
   image of it - and an image could not be diffed, searched, or cached anyway.
"""
from __future__ import annotations

import json
import re

# The map is read on every step by every lobe, so its whole point is being small.
# These caps are the budget that keeps it worth reading.
GOAL_MAX = 240
ROW_MAX = 160
DONE_MAX = 12
OPEN_MAX = 6
AREA_MAX = 24

# A path that has been read or written is an area of the project the mind has
# actually touched. Everything in the tree that is not here is what it has not.
_PATH_KEYS = ("path", "file", "filepath", "file_path", "filename", "target", "dir", "directory")
_CODE_KEYS = ("code", "command", "script", "shell", "bash", "python")
_IN_CODE = re.compile(r"""['"]([\w./\\-]*[/\\][\w./\\-]+)['"]""")
_FAILED = re.compile(r"\b(error|failed|failure|exception|traceback|timeout|denied|not found|"
                     r"no such|cannot|could not|blocked|refused)\b", re.I)


def _one_line(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit].rstrip() + "…" if len(text) > limit else text


def touched(name: str, arguments) -> list[str]:
    """Which parts of the project this call looked at or changed.

    Feeds the UNSEEN section. Anything no call has named is a place the mind has
    not been, which is exactly the blind spot the map exists to point at.
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except (ValueError, TypeError):
        return []
    if not isinstance(args, dict):
        return []
    out: list[str] = []
    for key in _PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and ("/" in value or "\\" in value or "." in value):
            out.append(value.strip("/\\ "))
    if any(word in str(name or "").lower() for word in ("run", "exec", "shell", "bash", "python", "test")):
        for key in _CODE_KEYS:
            value = args.get(key)
            if isinstance(value, str):
                out += _IN_CODE.findall(value)
    return [o for o in out if o]


class TaskPicture:
    """The working half: one job, all of it, at once. Rebuilt fast, cheap, and
    partly by code - the part a model is not allowed to assure its way through."""

    def __init__(self) -> None:
        self.goal = ""
        self.constraints: list[str] = []
        self.done: list[str] = []          # verified by a finished line or a tool result
        self.broken: list[str] = []        # failed and not yet repaired
        self.working: list[str] = []       # what is running right now (rebuilt each round)
        self.open: list[str] = []          # noticed, nobody working it
        self.known: list[str] = []         # areas a call has touched
        self.whole: list[str] = []         # the project, when something has listed it

    # ------------------------------------------------------- model may write these
    def set_goal(self, text: str, constraints: list[str] | None = None) -> None:
        """What the person actually asked for, and the lines it must not cross."""
        if text:
            self.goal = _one_line(text, GOAL_MAX)
        for note in constraints or []:
            note = _one_line(note, 120)
            if note and note not in self.constraints:
                self.constraints.append(note)
        self.constraints = self.constraints[-6:]

    def raise_open(self, question: str) -> None:
        """Something noticed that no line is doing. The section that outlives
        whoever noticed it, so it cannot fall off the end of a context window."""
        question = _one_line(question, ROW_MAX)
        if question and question not in self.open:
            self.open.append(question)
        self.open = self.open[-OPEN_MAX:]

    def close_open(self, question: str) -> None:
        words = set(re.findall(r"[a-z0-9]+", question.lower()))
        self.open = [q for q in self.open if not words & set(re.findall(r"[a-z0-9]+", q.lower()))]

    def set_areas(self, paths: list[str]) -> None:
        """What the project contains, from a listing. UNSEEN is derived, never claimed."""
        self.whole = [_one_line(p, 90) for p in list(paths)[:AREA_MAX]]

    # -------------------------------------------------- evidence writes these only
    def note_result(self, name: str, arguments, content) -> None:
        """What a real tool result says. Facts enter here or not at all."""
        for path in touched(name, arguments):
            top = path.strip("/\\ ").lower()
            if top and top not in self.known:
                self.known.append(top)
            if path.strip("/\\ ").lower().endswith("/") and path not in self.whole:
                self.set_areas([*self.whole, path])
        text = str(content or "")
        if _FAILED.search(text[:400]):
            self.fail(f"{name}: {_one_line(text, 110)}")

    def finish(self, label: str, topic: str, result: str) -> None:
        """A line ended and its result says what it found. Only now may it be DONE."""
        self.working = [w for w in self.working if not w.startswith(f"{label} ")]
        row = _one_line(f"{label} {topic}: {result}", ROW_MAX)
        if _FAILED.search(result[:200]):
            self.broken = [row, *[b for b in self.broken if b != row]][:DONE_MAX]
            return
        self.done = [*self.done, row][-DONE_MAX:]
        self.close_open(topic)

    def fail(self, what: str) -> None:
        row = _one_line(what, ROW_MAX)
        if row not in self.broken:
            self.broken = [row, *[b for b in self.broken if b != row]][:DONE_MAX]

    def repair(self, what: str) -> None:
        """Something broke and is now known to be handled: it stops being an alarm."""
        self.broken = [b for b in self.broken if _one_line(what, 40).lower() not in b.lower()]

    def start(self, label: str, topic: str) -> None:
        row = _one_line(f"{label} {topic}", 90)
        if row not in self.working:
            self.working.append(row)
        self.close_open(topic)

    def stop(self, label: str) -> None:
        self.working = [w for w in self.working if not w.startswith(f"{label} ")]

    # ------------------------------------------------------------------ the picture
    def unseen(self) -> list[str]:
        """Areas of the project no line has read or changed yet.

        This is the section the map exists for. Tunnel vision is not the model
        forgetting what it saw; it is the model never being told what it did not.
        """
        seen = {k.lower().rstrip("/") for k in self.known}
        out: list[str] = []
        for path in self.whole:
            low = path.lower().rstrip("/")
            if not low:
                continue
            # A directory counts as seen once something inside it has been read.
            if any(low == s or s.startswith(low + "/") or low.startswith(s + "/") for s in seen):
                continue
            out.append(path)
        return out[:10]

    def render(self) -> str:
        """The diagram a lobe reads. Stable order, one line each, no prose."""
        out: list[str] = []
        if self.goal:
            out.append(f"GOAL      {self.goal}")
        for note in self.constraints:
            out.append(f"MUST NOT  {note}")
        for row in self.working:
            out.append(f"WORKING   {row}")
        for row in self.done:
            out.append(f"DONE      {row}")
        for row in self.broken:
            out.append(f"BROKEN    {row}")
        for row in self.open:
            out.append(f"OPEN      {row}")
        for row in self.unseen():
            out.append(f"UNSEEN    {row}")
        return "\n".join(out)

    def diagram(self) -> str:
        """The same rows as a graph, for a client that wants to draw the picture.

        Rendered from the rows, never generated by a model: it is free, it is
        instant, it is always in step with the map, and it cannot show work
        that did not happen.
        """
        lines = ["flowchart LR"]
        lines.append(f'  goal["{self.goal or "the job"}"]')
        for i, row in enumerate(self.working):
            lines.append(f'  w{i}["{row}"]')
            lines.append("  goal --> w%d" % i)
        for i, row in enumerate(self.done):
            lines.append(f'  d{i}["{row}"]{{"shape": "rect"}}')
            lines.append("  d%d -.-> goal" % i)
        for i, row in enumerate(self.broken):
            lines.append(f'  b{i}["{row}"]')
            lines.append("  goal -.-o b%d" % i)
        for i, row in enumerate(self.open):
            lines.append(f'  o{i}["{row}"]')
            lines.append("  goal -.-> o%d" % i)
        for i, row in enumerate(self.unseen()):
            lines.append(f'  u{i}["{row}"]')
            lines.append("  goal -.-> u%d" % i)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience for logs
        return self.render() or "(no map yet)"


# ---------------------------------------------------------------- the person
#
# The project map is rewritten whenever the job changes. This one is read on
# EVERY turn and rewritten rarely, which is the whole reason it can sit in the
# cached prefix of the prompt: typing "hi" costs the same as ever and still
# arrives with the entire person in front of the voice.
#
# It is deliberately a form, not a profile. Sections in fixed order, one line
# each, so a rewrite is a small diff instead of a fresh essay, and so nothing
# the person once said can drift out of the picture because B re-summarised it
# slightly differently this week.
#
# WHO/WITH/WEIGHT lines are things B may only write from what the person said.
# NOTHING HERE IS INFERRED FROM A MOOD. The emotional read stays where it is
# (psych profile, chords) because that is a per-turn guess; this is memory.

PERSON_SECTIONS = ("WHO", "WITH", "WEIGHT", "DOING", "SINCE", "TEND", "ASKED")
SECTION_MAX = 40
PERSON_ROW_MAX = 180

# The model-editable keys and where they land. One list, so a typo in a section
# name cannot silently create a place facts go and nobody reads.
PERSON_FIELDS = (("who", "who"), ("with", "with_"), ("weight", "weight"),
                 ("doing", "doing"), ("since", "since"), ("tend", "tend"),
                 ("asked", "asked"))


class PersonPicture:
    """Everything known about one person, in one picture, read on every turn."""

    def __init__(self) -> None:
        # who: durable facts about them. with: the people in their life.
        # weight: what is heavy right now and when it landed - grief, illness,
        # a court date, a sick parent. This is the section that makes "hi"
        # land correctly, so it carries a date rather than a vibe.
        self.who: list[str] = []
        self.with_: list[str] = []
        self.weight: list[str] = []
        self.doing: list[str] = []    # affairs still running in their life
        self.since: list[str] = []    # dated: when something began/changed
        self.tend: list[str] = []     # how to be with them, learned not guessed
        self.asked: list[str] = []    # things they asked for that are still owed

    def merge(self, patch: dict) -> None:
        """Apply B's edit. Additive, deduplicated, capped.

        B writes lines; it never deletes the picture. Dropping a weight line is
        its own decision - "their father died" does not expire because it is no
        longer the topic - so removal only happens through `retire`, which is
        not reachable from the model's output.
        """
        for key, dest in PERSON_FIELDS:
            rows = patch.get(key) or []
            if isinstance(rows, str):
                rows = [rows]
            for raw in rows:
                row = _one_line(raw, PERSON_ROW_MAX)
                if not row:
                    continue
                current: list[str] = getattr(self, dest)
                # Same line, new wording: replace in place so the picture does
                # not carry two versions of one fact.
                words = set(re.findall(r"[a-z0-9]{4,}", row.lower()))
                for i, old in enumerate(current):
                    if words and words <= set(re.findall(r"[a-z0-9]{4,}", old.lower())):
                        current[i] = row
                        break
                else:
                    current.append(row)
            setattr(self, dest, getattr(self, dest)[:SECTION_MAX])

    def retire(self, section: str, needle: str) -> None:
        """Code-side only: an affair genuinely closed, said so in words."""
        rows: list[str] = getattr(self, section if section != "with" else "with_", [])
        low = needle.lower()
        setattr(self, section if section != "with" else "with_",
                [r for r in rows if low not in r.lower()])

    def render(self) -> str:
        out: list[str] = []
        for name, rows in zip(PERSON_SECTIONS, (self.who, self.with_, self.weight,
                                                self.doing, self.since, self.tend, self.asked)):
            for row in rows:
                out.append(f"{name:<7}  {row}")
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps({"who": self.who, "with": self.with_, "weight": self.weight,
                           "doing": self.doing, "since": self.since, "tend": self.tend,
                           "asked": self.asked}, ensure_ascii=False)

    @classmethod
    def parse(cls, raw) -> "PersonPicture":
        map_ = cls()
        data = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        if isinstance(data, dict):
            for key, dest in PERSON_FIELDS:
                value = data.get(key)
                if isinstance(value, list):
                    setattr(map_, dest, [_one_line(v, PERSON_ROW_MAX) for v in value if v][:SECTION_MAX])
        return map_

    def __str__(self) -> str:  # pragma: no cover
        return self.render() or "(no picture of them yet)"