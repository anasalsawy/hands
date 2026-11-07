"""The dual loop: one voice, two writings, one model.

This package is the part of the mind that needs nothing but an OpenAI-compatible
endpoint. Give it a base URL, a key and a model id and it thinks, alternates and
verifies on its own. Everything else - the workshop, voice, sleep, the UI - is a
capability a host supplies through :mod:`sawii_persona.kernel.caps`.

The loop is two writings of the *same* model. One holds the microphone; the other
is silent and doing the homework (widening context, or reading the person). They
trade at sentence boundaries, carrying state forward, so the voice never stops
while the thinking keeps going.
"""
from .caps import (
    CapabilityBundle,
    MemoryCap,
    NullCapabilities,
    ToolsCap,
    VerifyCap,
)
from .dual import Beat, DualLoop, DualResult
from .writings import LEFT, RIGHT, Writing, writings

__all__ = [
    "Beat",
    "CapabilityBundle",
    "DualLoop",
    "DualResult",
    "LEFT",
    "MemoryCap",
    "NullCapabilities",
    "RIGHT",
    "ToolsCap",
    "VerifyCap",
    "Writing",
    "writings",
]