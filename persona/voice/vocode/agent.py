"""Vocode Agent backed by Persona's shared DualLoop mind."""
from __future__ import annotations

from typing import Any, AsyncIterator

from ..native_input import NATIVE_REQUEST
from .base import BaseAgent, Transcription


class DualLoopAgent(BaseAgent):
    """One mind only: both dual-listen ears feed ``mind.respond`` / DualLoop.

    Never spawns a second agent or a parallel DualLoop. Progressive dual-listen
    context arrives as ``host_system``; L and R lobes of the same loop hear the
    original audio attachment together.
    """

    def __init__(self, mind: Any, persona: dict, *, session_id: str = "",
                 lane: str | None = None, dual: bool | None = True,
                 steer: bool | None = None, tz: str = "",
                 realtime_voice: bool = True):
        self.mind = mind
        self.persona = persona
        self.session_id = session_id
        self.lane = lane
        self.dual = dual
        self.steer = steer
        self.tz = tz
        self.realtime_voice = realtime_voice

    async def respond(self, transcription: Transcription) -> AsyncIterator[dict]:
        recording = transcription.audio
        text = (transcription.message or "").strip() or (
            NATIVE_REQUEST if recording is not None else "")
        if text or recording is not None:
            kwargs: dict[str, Any] = {
                "channel": "voice",
                "dual": self.dual,
                "steer": self.steer,
                "surface": "on a voice call",
                "tz": self.tz,
                "session_id": self.session_id,
                "host_system": transcription.prepared_context or "",
            }
            if self.lane:
                kwargs["lane"] = self.lane
            if recording is not None:
                kwargs["input_audio"] = recording
                if self.realtime_voice:
                    kwargs["realtime_voice"] = True
                if transcription.voice_role in ("L", "R"):
                    kwargs["voice_responder"] = transcription.voice_role
            async for event in self.mind.respond(self.persona["id"], text, **kwargs):
                yield event
