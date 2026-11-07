"""Emotional memory as a *chord*: many feelings at once, each with an intensity.

Emotions are a spectrum, never one label. A chord is a vector over named axes
(0..1 each). Bittersweet is {joy: .6, sadness: .7}, not a coin flip between
the two. Recall works on the geometry (resonance = cosine of chords), and is
vague by design: a loose band, not an exact match. Words and colours are just
renderings of the numbers.
"""
from __future__ import annotations

import math
from typing import Mapping

AXES: tuple[str, ...] = (
    "joy", "tenderness", "pride", "curiosity",
    "sadness", "fear", "anger", "shame",
)

# Rendering palette (the visible "line on a spectrum").
AXIS_COLOR = {
    "joy": (255, 214, 10), "tenderness": (255, 143, 171), "pride": (255, 159, 28),
    "curiosity": (72, 202, 228), "sadness": (58, 90, 200), "fear": (123, 44, 191),
    "anger": (214, 40, 40), "shame": (110, 110, 110),
}

_WORDS = {
    ("joy",): "happy", ("tenderness",): "warm", ("pride",): "proud", ("curiosity",): "curious",
    ("sadness",): "sad", ("fear",): "anxious", ("anger",): "angry", ("shame",): "embarrassed",
    ("joy", "sadness"): "bittersweet", ("sadness", "tenderness"): "wistful",
    ("anger", "fear"): "defensive", ("fear", "curiosity"): "uneasy-curious",
    ("joy", "pride"): "elated", ("anger", "sadness"): "hurt", ("fear", "sadness"): "dread",
    ("joy", "tenderness"): "affectionate", ("joy", "curiosity"): "playful",
    ("anger", "shame"): "humiliated", ("fear", "shame"): "insecure",
}

Chord = dict[str, float]
MAX_VOICES = 4


def clamp(v: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


def normalize(raw: Mapping[str, object] | None) -> Chord:
    """Coerce any model output into a clean chord over the known axes."""
    if not raw or not isinstance(raw, Mapping):
        return {}
    aliases = {"happiness": "joy", "love": "tenderness", "warmth": "tenderness", "affection": "tenderness",
               "interest": "curiosity", "grief": "sadness", "sorrow": "sadness", "anxiety": "fear",
               "worry": "fear", "frustration": "anger", "annoyance": "anger", "embarrassment": "shame",
               "guilt": "shame", "excitement": "joy", "amusement": "joy"}
    out: Chord = {}
    for k, v in raw.items():
        axis = aliases.get(str(k).lower().strip(), str(k).lower().strip())
        if axis in AXES:
            val = clamp(v)
            if val > 0.0:
                out[axis] = max(out.get(axis, 0.0), round(val, 3))
    if len(out) > MAX_VOICES:  # a chord keeps its loudest voices
        out = dict(sorted(out.items(), key=lambda kv: -kv[1])[:MAX_VOICES])
    return out


def vector(chord: Mapping[str, float] | None) -> list[float]:
    chord = chord or {}
    return [float(chord.get(a, 0.0)) for a in AXES]


def intensity(chord: Mapping[str, float] | None) -> float:
    v = vector(chord)
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0


def resonance(a: Mapping[str, float] | None, b: Mapping[str, float] | None) -> float:
    """How much two chords sound alike (cosine, 0..1)."""
    va, vb = vector(a), vector(b)
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def blend(current: Mapping[str, float] | None, new: Mapping[str, float] | None, weight: float = 0.35) -> Chord:
    """Mood drifts; it does not snap. Exponential moving blend of chords."""
    out: Chord = {}
    for a in AXES:
        v = (1 - weight) * float((current or {}).get(a, 0.0)) + weight * float((new or {}).get(a, 0.0))
        if v >= 0.02:
            out[a] = round(v, 3)
    return out


def describe(chord: Mapping[str, float] | None) -> str:
    """Human words for a chord (rendering only - the numbers are the truth)."""
    chord = normalize(chord)
    if not chord:
        return "neutral"
    top = sorted(chord.items(), key=lambda kv: -kv[1])
    strong = [a for a, v in top if v >= 0.35][:2]
    if len(strong) == 2:
        key = tuple(sorted(strong, key=AXES.index))
        if key in _WORDS:
            return _WORDS[key]
        return f"{_WORDS[(strong[0],)]} and {_WORDS[(strong[1],)]}"
    a, v = top[0]
    word = _WORDS[(a,)]
    if v < 0.2:
        return f"faintly {word}"
    if v > 0.75:
        return f"very {word}"
    return word


def color(chord: Mapping[str, float] | None) -> str:
    """Hex colour mixing the axes by intensity (white = calm)."""
    chord = normalize(chord)
    total = sum(chord.values())
    if not total:
        return "#f5f5f5"
    r = g = b = 0.0
    for a, v in chord.items():
        cr, cg, cb = AXIS_COLOR[a]
        r += cr * v / total
        g += cg * v / total
        b += cb * v / total
    k = min(1.0, intensity(chord) * 2.2)  # faint feelings fade toward white
    r = 245 * (1 - k) + r * k
    g = 245 * (1 - k) + g * k
    b = 245 * (1 - k) + b * k
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def render(chord: Mapping[str, float] | None) -> str:
    chord = normalize(chord)
    if not chord:
        return "neutral"
    parts = ", ".join(f"{a} {v:.1f}" for a, v in sorted(chord.items(), key=lambda kv: -kv[1]))
    return f"{describe(chord)} ({parts})"


def parse_declaration(text: str) -> Chord:
    """Parse ``joy=.6 tenderness=.4`` style declarations."""
    out: dict[str, float] = {}
    for part in (text or "").replace(",", " ").split():
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = clamp(v)
    return normalize(out)
