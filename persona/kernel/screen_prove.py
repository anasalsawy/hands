"""Not finished until the screen shows it — made mechanical.

NeuralAgent's earliest public contract (see ``screen_plan.py``) has one rule
this kernel could only ask for in a prompt until now: a subtask is closed when
the live screen demonstrates it, not when the speaker says so. A prompt can be
ignored; this gate cannot. When a writing emits the private ``⟦subtask done⟧``
marker, the kernel checks the last live observation of a screen hand. If the
observation does not carry the subtask's evidence, the marker does not advance
the plan — a note re-aims the next beat instead.

Two honesty limits, stated so nobody over-trusts the gate:

* Proof is **necessary, not sufficient**. The words being on the screen does
  not mean the right thing happened (a stale page can contain them). C's
  background check of each action stays the sufficient judge; this gate only
  removes the cheapest lie — declaring the screen done while it shows nothing.
* A step with **no screen channel at all** (a shell command in the same plan)
  cannot be proved by the screen, so the marker is honored for it. The screen
  proves the things the screen can see. Nothing here is a dice or a timer.
"""
from __future__ import annotations

import re

# Words that carry no evidence: a subtask made only of these has no visible
# expectation to look for.
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "then", "them", "they",
    "their", "there", "here", "what", "when", "which", "while", "about", "after",
    "before", "into", "onto", "over", "under", "your", "you", "are", "was",
    "were", "been", "does", "done", "make", "made", "take", "took", "give",
    # screen-hand verbs: describing the hand is not describing the outcome
    "open", "opens", "opening", "click", "clicks", "clicking", "type", "types",
    "typing", "press", "presses", "pressing", "scroll", "scroll down",
    "find", "finds", "search", "searches", "show", "shows", "showing", "look",
    "read", "reads", "go", "gets", "select", "selects", "choose", "navigates",
    "page", "screen", "window", "app", "file", "tab", "button", "field", "menu",
}

# A quoted span in a subtask is the exact text the person expects to see.
_QUOTED = re.compile(r"[\"'“‘]([^\"'”’]{2,60})[\"'”’]")
_WORD = re.compile(r"[a-z0-9][a-z0-9.'%-]*")


def _normal(text: str) -> str:
    return " ".join((text or "").lower().split())


def evidence_terms(subtask: str) -> list[str]:
    """The content words a live screen must carry for this subtask to be shown."""
    words = [w for w in _WORD.findall(_normal(subtask)) if len(w) >= 4 and w not in _STOP]
    # "returns" and "return", "booked" and "book": stems overlap loosely, but
    # dropping a plural 's' catches most of it without fuzzy libraries.
    seen: list[str] = []
    for w in words:
        stem = w[:-2] if w.endswith("ing") else (w[:-1] if w.endswith("s") else w)
        if stem not in seen:
            seen.append(stem)
    return seen


def screen_proves(subtask: str, observation: str) -> bool:
    """True when the live observation carries the subtask's evidence.

    Exact quoted text is required verbatim. Otherwise at least 60% of the
    subtask's content-word stems must appear (as a prefix of an observation
    word, so ``return`` matches ``returns`` and ``documented`` matches
    ``documentation``). Below two content words the subtask states nothing
    visible and proof falls through to the caller.
    """
    seen = _normal(observation)
    if not seen:
        return False
    quotes = [q.lower() for q in _QUOTED.findall(subtask or "") if q.strip()]
    if quotes:
        return all(_normal(q) in seen for q in quotes)
    terms = evidence_terms(subtask)
    if len(terms) < 2:
        return False
    hits = sum(1 for t in terms if any(w.startswith(t) for w in _WORD.findall(seen)))
    return hits / len(terms) >= 0.6


def latest_screen_text(calls) -> str:
    """The newest OBSERVATION of the screen this beat produced, or "".

    ``calls`` is the kernel's list of ``{"name", "result"}`` dicts. A beat may
    end with a fast (prepared) action after the spoken one, so the newest
    reading is the one the closing marker is judged against.

    An action's result is usually a receipt rather than an observation: the
    real desktop hand answers a click with ``clicked button 'Bluetooth'``,
    which says what was pressed and nothing about what the screen now shows.
    But a reading only proves a state it was taken *after*. So position
    decides, not kind:

    * a reading newer than every action in the beat is the observation;
    * an action after the newest reading makes that reading stale - it cannot
      show what the action just did - so the action's own result stands
      instead. When that is a bare receipt the gate withholds the marker,
      which is the right answer: read the screen, then close the subtask.

    Returning "" is reserved for a beat with no screen hand at all, because
    the gate reads that as "nothing for the screen to prove" and advances.
    """
    from .dual import _reads_screen, _screen_tool  # local: dual imports this too

    reading, reading_at = "", -1
    acted, acted_at = "", -1
    for i, c in enumerate(calls or []):
        try:
            name = c["name"] if isinstance(c, dict) else c.name
            result = c["result"] if isinstance(c, dict) else c.result
        except (KeyError, AttributeError):
            continue
        name = str(name)
        if _reads_screen(name):
            reading, reading_at = str(result or ""), i
        elif _screen_tool(name):
            acted, acted_at = str(result or ""), i
    if reading and reading_at > acted_at:
        return reading
    return acted or reading


def prove_or_wait(subtask: str, seen: str) -> tuple[bool, str]:
    """The gate the kernel applies to a closing marker. (advance?, note).

    No observation this round means no screen channel was used — nothing for
    the screen to prove (see the module docstring). Otherwise the marker only
    counts when the screen actually shows the subtask.
    """
    if not seen:
        return True, ""
    if screen_proves(subtask, seen):
        return True, ""
    return False, (
        f"The live screen does not show \"{subtask}\" yet. ⟦subtask done⟧ only counts "
        "when the observation in front of you carries it. Take the next step or say "
        "what the screen actually shows."
    )