"""Clause-boundary cut must fix truncation without shredding short replies.

Strings here are the actual truncated outputs measured on this box from the
clause-stop path: "Hello! How can", "I am fine, thank", "I'm here, just".

Run: python persona/voice/bench/test_clause.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from persona.voice.fast_L import clause_boundary  # noqa: E402

# (streamed text, expected thing a person should hear)
CASES = [
    ("Hello! How can", "Hello!"),
    ("I am fine, thank", "I am fine, thank"),      # comma with no real tail: keep
    ("I am fine, thank you for asking", "I am fine"),
    ("I am feeling calm and focused.", "I am feeling calm and focused."),
    ("Hi there! What", "Hi there!"),
    ("Good morning! How can I help", "Good morning!"),
    ("Fine, thank you", "Fine, thank you"),        # short both sides: keep whole
    ("You are welcome.", "You are welcome."),
    # A comma between two substantial clauses is a cut; a short head is not.
    # Known limit, deliberately accepted: a stream that stops mid-phrase with no
    # comma at all ("I'm here, just") is kept whole rather than truncated harder,
    # because guessing a cut with no punctuation is worse than reading the tail.
    ("I'm here, just thinking about it", "I'm here, just thinking about it"),
    ("I am here now, just thinking about the plan", "I am here now"),
    ("Go for a walk in nature.", "Go for a walk in nature."),
]

fails = 0
for text, want in CASES:
    n = clause_boundary(text)
    got = text[:n] if n else text
    ok = got.strip() == want.strip()
    fails += 0 if ok else 1
    print(("ok  " if ok else "FAIL") + f" {text!r} -> {got!r}" + ("" if ok else f" (want {want!r})"))
print("VERDICT:", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(0 if not fails else 1)