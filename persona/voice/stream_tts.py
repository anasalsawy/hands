"""Streaming TTS: the first audio BYTE must arrive in a few hundred ms.

Measured on this box (see persona/voice/bench/tts_probe2.py):

    gemini native NON-streaming  :generateContent   p50 2215 ms  (whole clip only)
    gemini native STREAMING      :streamGenerateContent?alt=sse
                                                    p50  565 ms  first audio byte
    featherless ResembleAI/chatterbox                p50 5434 ms
    deepinfra inworld realtime-tts                   402 (key suspended)

So the speak path uses the Gemini SSE endpoint whenever the TTS endpoint is a
Gemini one, and returns PCM as it arrives instead of waiting for the clip.

Only the transport is added here: the caller keeps its own chunking, barge-in
and playback gating, and feeds each decoded slice straight to the socket.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    import httpx2 as httpx  # type: ignore

log = logging.getLogger(__name__)

# OpenAI-compat base -> native Google base (the compat /audio/speech is a 404).
GOOGLE_COMPAT = "https://generativelanguage.googleapis.com/v1beta/openai"
GOOGLE_NATIVE = "https://generativelanguage.googleapis.com/v1beta/models"

# Fastest intelligible Gemini voice measured here; Puck 508-1009 ms first byte.
DEFAULT_VOICE = "Puck"
DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"

_SAMPLE_RATE_RE = re.compile(r'"sampleRateHertz"\s*:\s*(\d+)')
# Google returns the rate on the inline part, e.g. "audio/L16;codec=pcm;rate=24000".
_MIME_RATE_RE = re.compile(r"rate=(\d+)")


def _rate_of(inline: dict, fallback: int) -> int:
    m = _MIME_RATE_RE.search((inline or {}).get("mimeType") or "")
    return int(m.group(1)) if m else fallback


def gemini_stream_enabled(base_url: str) -> bool:
    """True when the configured TTS base is Google's and streaming is not off."""
    off = (os.environ.get("SAWII_TTS_STREAM") or "").strip().lower()
    if off in ("0", "false", "no", "off"):
        return False
    b = (base_url or "").lower()
    return "generativelanguage.googleapis.com" in b


ELEVENLABS_BASE = "https://api.elevenlabs.io"


def elevenlabs_stream_enabled(base_url: str) -> bool:
    """True when the TTS base is ElevenLabs and streaming is not off.

    Measured on this box (2026-11-08, persona/voice/bench numbers in
    C:\\Users\\anasa\\localtts\\el2.log): first audio byte p50 ~176 ms
    (eleven_flash_v2_5) / ~191 ms (eleven_turbo_v2_5) on a 55-char clause.
    That beats Gemini streaming (~565-997 ms) and the local PocketTTS
    checkpoint (~3.8-13.5 s including load), so ElevenLabs is the TTS winner
    whenever a key is present.
    """
    off = (os.environ.get("SAWII_TTS_STREAM") or "").strip().lower()
    if off in ("0", "false", "no", "off"):
        return False
    return "elevenlabs.io" in (base_url or "").lower()


def native_base(base_url: str) -> str:
    b = (base_url or "").rstrip("/")
    if b.endswith("/openai"):
        b = b[: -len("/openai")]
    return b + "/models"


@dataclass
class AudioSlice:
    """One playable piece of speech. ``final`` marks the last slice."""

    data: bytes
    mime: str
    sample_rate: int
    ms_elapsed: int
    final: bool = False


class GeminiStreamTTS:
    """SSE speech client: yields audio as the provider produces it.

    The wire format is Google native ``streamGenerateContent?alt=sse`` with
    ``responseModalities: [AUDIO]``; every event carries an inlineData PCM chunk.
    """

    def __init__(self, base_url: str, api_key: str, *, model: str = DEFAULT_MODEL,
                 voice: str = DEFAULT_VOICE, http: "httpx.AsyncClient | None" = None):
        self.base = native_base(base_url)
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.voice = voice or DEFAULT_VOICE
        self._http = http
        self.last_first_byte_ms: int | None = None

    def http(self) -> "httpx.AsyncClient":
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5))
        return self._http

    def _body(self, text: str) -> dict:
        return {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {
                    "voiceName": self.voice}}},
            },
        }

    async def stream(self, text: str,
                     stop: "Callable[[], bool] | None" = None) -> AsyncIterator[AudioSlice]:
        """Yield PCM slices. First slice is the one that gates audible speech.

        ``stop`` is polled between slices so a barge-in tears the SSE response
        down instead of finishing a clip nobody will hear.
        """
        url = f"{self.base}/{self.model}:streamGenerateContent?alt=sse"
        t0 = time.perf_counter()
        got_any = False
        rate = 24000
        async with self.http().stream(
                "POST", url, headers={"x-goog-api-key": self.api_key},
                json=self._body(text), timeout=httpx.Timeout(60, connect=5)) as r:
            if r.status_code >= 400:
                raw = (await r.aread())[:200]
                raise RuntimeError(f"gemini tts {r.status_code}: "
                                   f"{raw.decode('utf-8', 'replace')[:120]}")
            async for line in r.aiter_lines():
                if stop is not None and stop():
                    await r.aclose()
                    return
                if not line.startswith("data:"):
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except ValueError:
                    continue
                cfg = json.dumps(obj)
                m = _SAMPLE_RATE_RE.search(cfg)
                if m:
                    rate = int(m.group(1))
                for cand in obj.get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        inline = part.get("inlineData") or part.get("inline_data")
                        data = (inline or {}).get("data")
                        if not data:
                            continue
                        rate = _rate_of(inline, rate)
                        raw = base64.b64decode(data)
                        if not raw:
                            continue
                        if not got_any:
                            self.last_first_byte_ms = round((time.perf_counter() - t0) * 1000)
                        got_any = True
                        yield AudioSlice(raw, "audio/l16;rate=" + str(rate), rate,
                                         round((time.perf_counter() - t0) * 1000))
        if not got_any:
            raise RuntimeError("gemini tts returned no audio")
        yield AudioSlice(b"", "audio/l16;rate=" + str(rate), rate,
                         round((time.perf_counter() - t0) * 1000), final=True)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


class ElevenStreamTTS:
    """ElevenLabs streaming speech: raw PCM out of the wire, first byte ~180 ms.

    Same shape as :class:`GeminiStreamTTS` - yields :class:`AudioSlice` PCM and
    honours ``stop`` so a barge-in tears the response down. The model id is the
    ElevenLabs ``model_id``; the voice is a voice id, not a preset name.
    """

    def __init__(self, base_url: str, api_key: str, *, model: str = "eleven_flash_v2_5",
                 voice: str = "", http: "httpx.AsyncClient | None" = None,
                 sample_rate: int = 22050):
        self.base = (base_url or ELEVENLABS_BASE).rstrip("/")
        self.api_key = api_key
        self.model = model or "eleven_flash_v2_5"
        self.voice = voice
        self.rate = sample_rate
        self._http = http
        self.last_first_byte_ms: int | None = None

    def http(self) -> "httpx.AsyncClient":
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5))
        return self._http

    async def stream(self, text: str,
                     stop: "Callable[[], bool] | None" = None) -> AsyncIterator[AudioSlice]:
        if not self.voice:
            raise RuntimeError("elevenlabs tts needs a voice id")
        url = f"{self.base}/v1/text-to-speech/{self.voice}/stream?output_format=pcm_22050"
        t0 = time.perf_counter()
        got_any = False
        async with self.http().stream(
                "POST", url, headers={"xi-api-key": self.api_key},
                json={"text": text, "model_id": self.model},
                timeout=httpx.Timeout(60, connect=5)) as r:
            if r.status_code >= 400:
                raw = (await r.aread())[:200]
                raise RuntimeError(f"elevenlabs tts {r.status_code}: "
                                   f"{raw.decode('utf-8', 'replace')[:120]}")
            async for chunk in r.aiter_bytes():
                if stop is not None and stop():
                    await r.aclose()
                    return
                if not chunk:
                    continue
                if not got_any:
                    self.last_first_byte_ms = round((time.perf_counter() - t0) * 1000)
                got_any = True
                yield AudioSlice(chunk, f"audio/l16;rate={self.rate}", self.rate,
                                 round((time.perf_counter() - t0) * 1000))
        if not got_any:
            raise RuntimeError("elevenlabs tts returned no audio")

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


async def gemini_native_once(http: "httpx.AsyncClient", base_url: str, api_key: str,
                             model: str, voice: str, text: str) -> tuple[bytes, str]:
    """One non-streaming Gemini speech call. Returns (audio_bytes, mime).

    This exists because the OpenAI-compat ``/audio/speech`` on a Google base is
    a permanent 404: when SSE fails (quota, throttle) the fallback has to speak
    through the native endpoint or the turn is silent. Measured cost of this
    path: p50 2215 ms versus 565 ms streamed.
    """
    url = f"{native_base(base_url)}/{model or DEFAULT_MODEL}:generateContent"
    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {
                "voiceName": voice or DEFAULT_VOICE}}},
        },
    }
    r = await http.post(url, headers={"x-goog-api-key": api_key}, json=body)
    r.raise_for_status()
    rate = 24000
    for cand in (r.json().get("candidates") or []):
        for part in (cand.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                pcm = base64.b64decode(inline["data"])
                return wav_from_pcm(pcm, _rate_of(inline, rate)), "audio/wav"
    raise RuntimeError("gemini tts returned no audio")


def wav_from_pcm(pcm: bytes, sample_rate: int, channels: int = 1, width: int = 2) -> bytes:
    """Wrap streamed PCM in a RIFF header so any client can play a finished clip."""
    import struct

    n = len(pcm)
    byte_rate = sample_rate * channels * width
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + n, b"WAVE", b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, channels * width, width * 8, b"data", n)
    return header + pcm