"""Minimal Vocode-shaped abstract components (transcriber / agent / synthesizer)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Transcription:
    """Vocode Transcription analogue — may carry native audio instead of text."""

    message: str = ""
    confidence: float = 1.0
    is_final: bool = True
    audio: Any | None = None
    mime: str = "audio/wav"
    prepared_context: str = ""
    voice_role: str = "L"
    speculative_opener: str = ""


@dataclass
class AgentResponse:
    text: str
    is_interruptible: bool = True
    events: list[dict] = field(default_factory=list)


@dataclass
class SynthesisResult:
    data: bytes
    content_type: str = "audio/mpeg"


class BaseTranscriber(ABC):
    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None: ...

    @abstractmethod
    async def end_utterance(self, mime: str = "audio/webm") -> Transcription: ...

    async def close(self) -> None:
        return None


class BaseAgent(ABC):
    @abstractmethod
    def respond(self, transcription: Transcription) -> AsyncIterator[dict]: ...


class BaseSynthesizer(ABC):
    @abstractmethod
    async def create_speech(self, text: str, *, clone: dict | None = None) -> SynthesisResult: ...
