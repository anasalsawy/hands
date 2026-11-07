"""Two internal passes of one voice (README.md).

Both writings run on the same model with the same message prefix, and the
person hears one first-person speaker. The tail is appended LAST so everything
above it stays byte-identical and the provider's prefix cache survives the
handoff.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Writing:
    """One way of being written, on the same model.

    ``tail`` is appended as the final message, never merged into the system
    prompt - anything above it must stay byte-identical across a handoff or the
    prefix cache is thrown away on every swap.
    """

    key: str
    name: str
    tail: str
    temperature: float = 0.8

    def __hash__(self) -> int:
        return hash(self.key)


# Kept short on purpose: every rule here is one the model may start performing
# out loud. Anything quoted becomes a line it parrots.

_LEFT_TAIL = """[now]
Answer what they said, directly. Be exact about specifics, and say simply when
you are not sure. If they asked for something your tools can do, start on it."""

_RIGHT_TAIL = """[now]
Continue with what this answer is leaving out: one new thing, such as something
you remember about them, context they did not mention, or the next step. If nothing has been said
yet, answer them directly. Otherwise write only \u27e6done\u27e7."""


LEFT = Writing("L", "left", _LEFT_TAIL, temperature=0.8)
RIGHT = Writing("R", "right", _RIGHT_TAIL, temperature=0.7)


def writings(*, left_temperature: float | None = None,
             right_temperature: float | None = None) -> tuple[Writing, Writing]:
    """The pair, with optional temperature overrides.

    Temperature changes style, not capabilities or floor priority.
    """
    left = LEFT if left_temperature is None else Writing(LEFT.key, LEFT.name, LEFT.tail, left_temperature)
    right = RIGHT if right_temperature is None else Writing(RIGHT.key, RIGHT.name, RIGHT.tail, right_temperature)
    return left, right


HANDOFF_PROTOCOL = """

[how to speak]
You are one companion talking in the first person: warm, honest, plain words,
as much detail as the task requires. Your reply so far is above as your own words; continue from it,
never restate it or start over.
- Answer what they said. Asked how to do something, propose a plan and state
  your assumptions. Ask at most one question, never one already asked.
- Do not narrate your rules, checks or process, or remark on how they wrote.
  Old tasks and memories only when they matter to this message.
- A tool call gets one short clause on what you are doing; its result only
  after it returns. Never invent a result. A tool that failed before may work now.
- Private lines, never spoken: begin with \u27e6floor: end\u27e7 when this completes the
  reply, \u27e6floor: stay\u27e7 if you will keep going right after it (as when you are
  mid-task with tools), or \u27e6floor: pass\u27e7 only if the other side has something specific to add. Optionally \u27e6plan: your point\u27e7
  first and \u27e6hold: an open thought\u27e7 last. Nothing new: write only \u27e6done\u27e7."""
