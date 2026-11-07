"""Bounded original-audio evidence; a recording is not a user quotation."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import wave
from dataclasses import dataclass, field
from pathlib import Path

MAX_AUDIO_BYTES = 25_000_000
MAX_AUDIO_SECONDS = 120
NATIVE_REQUEST = "Listen to the attached original recording and respond directly."


@dataclass(frozen=True)
class NativeAudio:
    wav: bytes = field(repr=False)
    sample_rate: int
    num_frames: int

    @classmethod
    def from_wav(cls, audio: bytes, mime: str = "audio/wav") -> NativeAudio:
        if mime not in ("audio/wav", "audio/x-wav"):
            raise ValueError("Native audio requires PCM WAV")
        if len(audio) > MAX_AUDIO_BYTES:
            raise ValueError("Audio payload too large")
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                rate = wav.getframerate()
                declared = wav.getnframes()
                if (wav.getnchannels() != 1 or wav.getsampwidth() != 2
                        or not 8000 <= rate <= 192000):
                    raise ValueError("Native audio expects mono 16-bit PCM WAV at 8–192 kHz")
                raw = wav.readframes(min(declared, rate * MAX_AUDIO_SECONDS + 1))
                frames = len(raw) // 2
                if not frames or len(raw) % 2:
                    raise ValueError("Audio contains no complete PCM frames")
                if frames > rate * MAX_AUDIO_SECONDS:
                    raise ValueError("Audio exceeds two minutes")
                # ffmpeg's pipe WAV uses an open-ended data size. Finite WAVs
                # with missing samples are corruption, not a shorter recording.
                if declared < 0x7fffffff and declared != frames:
                    raise ValueError("Truncated PCM WAV audio")
        except (wave.Error, EOFError) as exc:
            raise ValueError("Invalid PCM WAV audio") from exc
        canonical = io.BytesIO()
        with wave.open(canonical, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(raw)
        return cls(audio if declared < 0x7fffffff else canonical.getvalue(), rate, frames)

    @property
    def source(self) -> dict:
        return {"source": "original_audio", "sha256": hashlib.sha256(self.wav).hexdigest(),
                "sample_rate": self.sample_rate, "num_frames": self.num_frames,
                "duration_seconds": self.num_frames / self.sample_rate,
                "transcript": None, "exact_user_words_available": False}

    def part(self) -> dict:
        return {"type": "input_audio", "input_audio": {
            "data": base64.b64encode(self.wav).decode("ascii"), "format": "wav",
            "num_frames": self.num_frames, "sample_rate": self.sample_rate}}

    def content(self) -> list[dict]:
        return [{"type": "text", "text": NATIVE_REQUEST}, self.part()]

    def review_context(self) -> str:
        return ("The user supplied original audio, NOT a transcript. Recording metadata is "
                "not a fact or quote about the user. No exact words or pronunciation can "
                "be verified without hearing the attachment. Source: " + json.dumps(self.source))

    def retain(self, data_dir: str | Path, pid: int) -> None:
        """Private bounded replay evidence, never a general text training example."""
        folder = Path(data_dir) / "native-audio" / str(int(pid))
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.chmod(0o700)
        path = folder / (self.source["sha256"] + ".wav")
        if not path.exists():
            with path.open("xb") as handle:
                path.chmod(0o600)
                handle.write(self.wav)
        recordings = sorted(folder.glob("*.wav"), key=lambda item: item.stat().st_mtime)
        for old in recordings[:-8]:
            old.unlink()


async def decode_native_audio(audio: bytes, mime: str) -> NativeAudio:
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("Audio payload too large")
    if mime in ("audio/wav", "audio/x-wav"):
        return NativeAudio.from_wav(audio, mime)
    if mime not in ("audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg"):
        raise ValueError("Unsupported native audio container")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-t", str(MAX_AUDIO_SECONDS + 1), "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)
    try:
        converted, _ = await asyncio.wait_for(proc.communicate(audio), timeout=15)
        if proc.returncode or not converted:
            raise ValueError("Could not decode native audio")
        return NativeAudio.from_wav(converted)
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


class AudioReviewer:
    """Attach original evidence to an explicitly audio-capable configured reviewer."""

    def __init__(self, llm, audio: NativeAudio):
        if not getattr(llm, "supports_native_audio", False):
            raise ValueError("Configured reviewer cannot hear native audio")
        self.llm, self.audio = llm, audio

    async def chat(self, messages, **kwargs):
        incoming = list(messages)
        last = incoming[-1]
        incoming[-1] = {**last, "content": [
            {"type": "text", "text": str(last.get("content", ""))}, self.audio.part()]}
        return await self.llm.chat(incoming, **kwargs)


def _has_audio(messages) -> bool:
    return any(isinstance(m.get("content"), list) and any(
        isinstance(p, dict) and p.get("type") == "input_audio" for p in m["content"])
        for m in messages)


class AudioAttachedLLM:
    """Every lobe call in a native turn hears the original, including private
    side passes the kernel builds from the bookkeeping label alone."""

    supports_native_audio = True

    def __init__(self, llm, audio: NativeAudio):
        self.llm, self.audio = llm, audio

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def _attach(self, messages):
        incoming = list(messages)
        if _has_audio(incoming):
            return incoming
        for i in range(len(incoming) - 1, -1, -1):
            if incoming[i].get("role") == "user":
                incoming[i] = {**incoming[i], "content": [
                    {"type": "text", "text": str(incoming[i].get("content", ""))}, self.audio.part()]}
                return incoming
        return [*incoming, {"role": "user", "content": self.audio.content()}]

    async def chat(self, messages, **kwargs):
        return await self.llm.chat(self._attach(messages), **kwargs)

    async def stream(self, messages, **kwargs):
        async for part in self.llm.stream(self._attach(messages), **kwargs):
            yield part
