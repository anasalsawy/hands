"""The two products.

One mind. Two ways of meeting it. The mind is the two lobes.

**Companion** is her home: the app. The computer and the browser run
inside the app. She has a workspace, a shell, a screen, and a voice
you can talk over, plus the mind's own tools (clock, memory, workshop).

**Inference** is the same mind behind ``POST /v1/chat/completions``, OpenAI
compatible. Any agent, app, or runtime calls it with the persona key. The
caller brings its own tools; she decides the calls and the caller runs them.
Her browser and her computer stay at home.

``X-Sawii-Client: companion`` selects the home. Anything else is inference.

Weights are not settled here. The 2026-10-21 lock still stands: a home turn
can take one learning step when a backend is on, and a visit does not.
"""
from __future__ import annotations


COMPANION_FEATURES = (
    "kernel",
    "memory",
    "verifier",
    "psyche",
    "browser",
    "computer",
    "voice",
    "hands",
    "phone",
    "workshop",
)

INFERENCE_FEATURES = (
    "kernel",
    "memory",
    "verifier",
    "psyche",
    "client_tools",
    "steer",
)


def describe() -> dict:
    return {
        "mind": "two lobes",
        "products": [
            {
                "id": "companion",
                "home": True,
                "how": "The app sends X-Sawii-Client: companion. Voice is /v1/voice.",
                "features": list(COMPANION_FEATURES),
                "weights": (
                    "2026-10-21 lock, unchanged: one learning step from memory "
                    "after a home turn, and only when SAWII_WEIGHTS is set. "
                    "The default is off. Fireworks is not built."
                ),
            },
            {
                "id": "inference",
                "home": False,
                "how": "POST /v1/chat/completions with the persona key. Model id sawii-persona.",
                "features": list(INFERENCE_FEATURES),
                "weights": (
                    "2026-10-21 lock, unchanged: a visit does not update her weights. "
                    "Changing that waits on the brainstorm."
                ),
            },
        ],
    }
