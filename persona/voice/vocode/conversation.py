"""Vocode StreamingConversation analogue wired to Persona's DualLoop pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent import DualLoopAgent
from .synthesizer import DeepInfraMiMoSynthesizer
from .transcriber import DualListenTranscriber, status_dict


@dataclass
class VoicePipeline:
    """Bundle of Vocode-shaped components for one companion voice session."""

    transcriber: DualListenTranscriber
    agent: DualLoopAgent
    synthesizer: DeepInfraMiMoSynthesizer

    async def close(self) -> None:
        await self.transcriber.close()


class StreamingConversation:
    """Orchestration handle — VoiceSession remains the WS protocol owner.

    Vocode's StreamingConversation owns mic/speaker I/O. Here the browser
    owns capture/playback over ``/v1/voice``; this object holds the three
    pipeline stages so transcription → DualLoop → TTS stay clearly named.
    """

    def __init__(self, pipeline: VoicePipeline):
        self.pipeline = pipeline

    @property
    def transcriber(self) -> DualListenTranscriber:
        return self.pipeline.transcriber

    @property
    def agent(self) -> DualLoopAgent:
        return self.pipeline.agent

    @property
    def synthesizer(self) -> DeepInfraMiMoSynthesizer:
        return self.pipeline.synthesizer

    async def close(self) -> None:
        await self.pipeline.close()


def build_pipeline(*, mind: Any, persona: dict, speech: Any, duplex: Any,
                   send_ready_meta: bool = True,  # noqa: ARG001 - reserved
                   session_id: str = "", lane: str | None = None,
                   dual: bool | None = True, steer: bool | None = None,
                   tz: str = "", realtime_voice: bool = True,
                   http=None) -> VoicePipeline:
    """Construct the Vocode-faithful DeepInfra companion voice pipeline."""
    settings = getattr(mind, "s", None)
    voice = getattr(speech, "voice", "") or getattr(settings, "tts_voice", "") or "Mia"
    transcriber = DualListenTranscriber(duplex)
    agent = DualLoopAgent(
        mind, persona, session_id=session_id, lane=lane, dual=dual,
        steer=steer, tz=tz, realtime_voice=realtime_voice)
    synthesizer = DeepInfraMiMoSynthesizer(speech.tts, voice=voice, http=http)
    return VoicePipeline(transcriber=transcriber, agent=agent, synthesizer=synthesizer)


def pipeline_status(mind: Any, speech: Any) -> dict:
    return status_dict(mind, speech, getattr(mind, "s", None))
