"""Vocode Transcriber: DeepInfra MiMo native hear + dual-listen into one mind.

This is NOT classic Whisper STT. MiMo-V2.6-Flash hears original audio; dual
listen (L+R ears) prepares a shared DualLoop turn without splitting agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..duplex import DuplexVoice
from ..native_input import NativeAudio, decode_native_audio
from .base import BaseTranscriber, Transcription


@dataclass
class HeardUtterance:
    recording: NativeAudio
    prepared_context: str = ""
    voice_role: str = "L"
    source: dict | None = None


class DeepInfraNativeHear(BaseTranscriber):
    """Buffers mic frames and decodes a final native-audio utterance."""

    def __init__(self, *, max_bytes: int = 25_000_000):
        self.max_bytes = max_bytes
        self._buf = bytearray()
        self._overflow = False

    async def send_audio(self, chunk: bytes) -> None:
        if self._overflow:
            return
        if len(self._buf) + len(chunk) > self.max_bytes:
            self._buf.clear()
            self._overflow = True
            return
        self._buf.extend(chunk)

    def peek(self) -> bytes:
        return bytes(self._buf)

    def clear(self) -> None:
        self._buf.clear()
        self._overflow = False

    @property
    def overflowed(self) -> bool:
        return self._overflow

    async def end_utterance(self, mime: str = "audio/webm") -> Transcription:
        audio, self._buf = bytes(self._buf), bytearray()
        overflow, self._overflow = self._overflow, False
        if overflow or not audio:
            return Transcription(message="", is_final=True)
        recording = await decode_native_audio(audio, mime)
        return Transcription(
            message="",
            is_final=True,
            audio=recording,
            mime="audio/wav",
        )


class DualListenTranscriber(BaseTranscriber):
    """Two ears on the same original audio, feeding one DualLoop mind.

    Progressive prefixes run through ``DuplexVoice`` (L listener + R draft)
    while the mic is still open. The final utterance attaches the same
    ``NativeAudio`` to DualLoop — never a second mind or split agent.
    """

    def __init__(self, duplex: DuplexVoice, *, max_bytes: int = 25_000_000):
        self.duplex = duplex
        self._hear = DeepInfraNativeHear(max_bytes=max_bytes)

    async def send_audio(self, chunk: bytes) -> None:
        await self._hear.send_audio(chunk)

    def offer_progress(self, mime: str) -> None:
        audio = self._hear.peek()
        if audio and not self._hear.overflowed:
            self.duplex.offer(audio, mime)

    async def end_utterance(self, mime: str = "audio/webm") -> Transcription:
        voice_role, prepared, opener = await self.duplex.finish_prefix()
        base = await self._hear.end_utterance(mime)
        if base.audio is None:
            return base
        return Transcription(
            message=base.message,
            is_final=True,
            audio=base.audio,
            mime=base.mime,
            prepared_context=prepared,
            voice_role=voice_role or "L",
            speculative_opener=opener or "",
        )

    async def close(self) -> None:
        await self.duplex.close()


def status_dict(mind: Any, speech: Any, settings: Any) -> dict:
    """Compact voice capability snapshot for /v1/voice-status and ready frames."""
    native = getattr(mind, "voice_native", None)
    tts = getattr(speech, "tts", None)
    voice_ep = getattr(settings, "voice_native", None)
    native_model = getattr(getattr(native, "endpoint", None), "model", None) or (
        getattr(voice_ep, "model", None) if voice_ep is not None else None)
    native_on = bool(getattr(settings, "native_audio_default", False) and native is not None)
    return {
        "asr": "native" if native_on else (
            "server" if getattr(speech, "has_asr", False) else "client"),
        "native": {
            "enabled": native_on,
            "model": native_model,
            "configured": bool(native is not None),
            "supports_native_audio": bool(getattr(native, "supports_native_audio", False)),
        },
        "tts": {
            "enabled": bool(getattr(speech, "has_tts", False)),
            "model": getattr(tts, "model", None),
            "base_url": getattr(tts, "base_url", None),
        },
        "dual_listen": True,
        "shared_mind": "DualLoop",
        "architecture": "vocode-faithful",
    }
