"""The earliest NeuralAgent loop, inside this engine.

Source: https://github.com/mosdehcom/neuralagent (desktop/aiagent/main.py and
backend/routers/aiagent/generic.py, the public version from 2025-07-23).

That core is three rules:

1. A planner writes the subtasks once. They are the user's goals, not every click.
2. The step that follows reads the live screen and may fire the next press
   immediately. It does not sit still and re-plan a click it already prepared.
3. A subtask is finished only when the screen shows it. "Opening the page" is
   not finished. The task is not finished while a subtask is still open.

The dual lobe keeps the part NeuralAgent does not have: while the press is in
flight, the other writing is already looking and talking.
"""
from __future__ import annotations

import json
import re


class ScreenPlan:
    """One plan for the screen task on this turn. Written once, walked in order."""

    def __init__(self, subtasks: list[str]):
        self.subtasks = [s for s in subtasks if s][:6]
        self.index = 0
        self.holds = 0

    @property
    def current(self) -> str:
        if self.finished:
            return ""
        return self.subtasks[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.subtasks)

    def advance(self) -> None:
        if not self.finished:
            self.index += 1


def parse_plan(text: str) -> list[str]:
    """Subtasks from a planner line. Empty when the text is not a plan."""
    raw = text or ""
    marker = raw.find('"subtasks"')
    if marker < 0:
        return []
    start = raw.rfind("{", 0, marker)
    if start < 0:
        return []
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return []
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return []
    found = []
    for item in obj.get("subtasks") or []:
        if isinstance(item, str) and item.strip():
            found.append(item.strip())
        elif isinstance(item, dict):
            line = str(item.get("subtask") or item.get("text") or "").strip()
            if line:
                found.append(line)
    return found[:6]


_NEXT = re.compile(r"NEXT\s*(\{)", re.IGNORECASE)


def parse_next_action(text: str) -> dict | None:
    """The one press prepared during the wait. None when the screen said wait."""
    raw = text or ""
    match = _NEXT.search(raw)
    if not match:
        return None
    start = match.end(1) - 1
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    name = str(obj.get("name") or "").strip()
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {}
    if not name:
        return None
    return {"name": name, "arguments": dict(args)}


def marked_done(text: str) -> bool:
    """The private mark that this subtask is visible on the screen."""
    low = (text or "").lower()
    return "subtask done" in low or "subtask_completed" in low


def ground_index(elements, text: str, fallback: int) -> int:
    """Find the live element by the words on it. A saved index is the fallback.

    NeuralAgent's later replays refuse stored coordinates. The earliest public
    version already re-reads the interactive elements on every step. Matching
    the label is that rule for a page whose indexes moved.
    """
    want = " ".join((text or "").lower().split())
    if not want:
        return fallback
    for el in elements or []:
        if not isinstance(el, dict):
            continue
        label = " ".join(str(el.get("text") or "").lower().split())
        if label and (want in label or label in want):
            try:
                return int(el["index"])
            except (KeyError, TypeError, ValueError):
                continue
    return fallback
