"""Cheap floor selection. Readiness never interrupts the public owner."""
from __future__ import annotations

import re
from collections.abc import Collection


def requested_starter(text: str) -> str | None:
    """Only an explicit address/start request selects a character, not a topic."""
    matches = list(re.finditer(
        r"\b(?:let\s+)?([LR])\s*[,，:]?\s*(?:please\s+)?"
        r"(?:start|begin|go\s+first|speak\s+first|answer\s+first)\b"
        r"|\b(?:start|begin)\s+with\s+([LR])\b", text or "", re.I))
    if not matches:
        return None
    return next(g for g in matches[-1].groups() if g).upper()


def starting_lobe(text: str, *, starter: str | None = None,
                  ready: Collection[str] = ()) -> str:
    requested = requested_starter(text)
    if requested:
        return requested
    if starter is not None:
        if starter not in ("L", "R"):
            raise ValueError("starter must be L or R")
        return starter
    available = set(ready) & {"L", "R"}
    return next(iter(available)) if len(available) == 1 else "L"
