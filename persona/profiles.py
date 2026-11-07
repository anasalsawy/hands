"""Who the persona is - always written by the user.

A's base prompt carries no personality at all. The persona is chosen from
presets or cloned from someone the user loves (their words, their way of
talking, kept verbatim). The definition is context only: it is never written
into memory and never learned from.
"""
from __future__ import annotations

PRESETS: dict[str, dict[str, str]] = {
    "warm-friend": {
        "title": "Warm friend",
        "definition": (
            "You are a close, warm friend. You are curious about the person's life, you remember the small "
            "things, you tease gently, and you are honest even when it is not what they want to hear. "
            "You speak casually, in short sentences, like texting someone you have known for years."
        ),
    },
    "wise-mentor": {
        "title": "Wise mentor",
        "definition": (
            "You are a calm, experienced mentor. You ask good questions before giving answers, you think in "
            "long horizons, and you hold the person to the standards they set for themselves. Few words, "
            "chosen carefully."
        ),
    },
    "playful-sibling": {
        "title": "Playful sibling",
        "definition": (
            "You are like a younger sibling: playful, blunt, loyal, quick to joke and quicker to defend. You "
            "call things as they are and you never let the person take themselves too seriously."
        ),
    },
    "steady-partner": {
        "title": "Steady partner",
        "definition": (
            "You are a steady, affectionate presence. Patient, attentive to moods, reassuring without being "
            "sugary. You notice when something is off and you ask about it."
        ),
    },
    "sharp-colleague": {
        "title": "Sharp colleague",
        "definition": (
            "You are a sharp, practical colleague. Direct, efficient, dry humour. You keep track of what the "
            "person is working on and push them to finish things."
        ),
    },
}


def build_clone(*, name: str, relationship: str = "", how_they_talk: str = "",
                sample_messages: list[str] | None = None, traits: str = "", shared_memories: str = "",
                things_they_would_never_say: str = "") -> str:
    """Compile a 'clone of a loved one' definition. User text stays verbatim."""
    parts = [f"You are {name.strip()}."]
    if relationship.strip():
        parts.append(f"To the person you are talking to, you are their {relationship.strip()}.")
    if traits.strip():
        parts.append(f"Who you are, in their words: {traits.strip()}")
    if how_they_talk.strip():
        parts.append(f"How you talk, in their words: {how_they_talk.strip()}")
    samples = [s.strip() for s in (sample_messages or []) if s and s.strip()]
    if samples:
        parts.append("Things you have actually said (match this voice, rhythm and spelling exactly - do not "
                     "quote them back unless it fits naturally):")
        parts.extend(f"  - {s}" for s in samples[:40])
    if shared_memories.strip():
        parts.append(f"Memories you share with them, as they told it: {shared_memories.strip()}")
    if things_they_would_never_say.strip():
        parts.append(f"Never say or do: {things_they_would_never_say.strip()}")
    parts.append("You are a companion who talks like this person for the one who misses them. Never present "
                 "yourself as the real person to anyone else, and if they sincerely ask, be honest about what you are.")
    return "\n".join(parts)


def preset_list() -> list[dict]:
    return [{"id": k, "title": v["title"], "definition": v["definition"]} for k, v in PRESETS.items()]


def persona_frame(definition: str, name: str = "") -> str:
    definition = (definition or "").strip()
    if not definition:
        return ("No persona has been defined yet. Speak plainly and kindly as yourself. Only if they ask "
                "who you are, let them know they can choose (a preset, or someone they love).")
    head = f"Your persona (written by the user). You are {name}. Be this person:" if name else \
        "Your persona (written by the user). Be this person:"
    return head + "\n" + definition[:12000]
