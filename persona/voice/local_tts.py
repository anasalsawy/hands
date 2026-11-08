"""Local speech: a neural TTS that runs in-process, off the network entirely.

Measured on this box (Intel Core 5 120U, 12 logical CPUs, no GPU):

    Piper en_US-lessac-low, voice preloaded   first audio byte  p50 588 ms  min 121 ms
    Piper CLI, one process per sentence       +1.85 s model load every call  (unusable)
    Gemini native streamed                    first audio byte  p50 565 ms
    Gemini native non-streaming               whole clip only   p50 2215 ms
    Featherless ResembleAI/chatterbox         whole clip only   p50 5434 ms

Piper is within noise of the best remote path and has no quota, no key and no
round trip, so it is the only speech that still works when a provider returns
429. The catch is the same as for the local model: loading the voice takes
~8.7 s, so it must be loaded at startup and never on the critical path.

Piper's own Python API yields audio chunks as it synthesises, which is what
keeps first-audio low: we hand the caller the first playable WAV as soon as
the first clause of audio exists instead of waiting for the sentence.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_VOICE = os.environ.get(
    "SAWII_PIPER_VOICE", r"C:\Users\anasa\localtts\voices\en_US-lessac-low.onnx")

# Enough audio to cover the first sentence's opening words; longer than this and
# the person waits for synthesis they could already be hearing.
HEAD_MS = 500


def piper_enabled() -> bool:
    """True when piper is importable and a voice file exists."""
    raw = (os.environ.get("SAWII_TTS_LOCAL") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(piper_voice_path())


def piper_voice_path() -> str:
    voice = (os.environ.get("SAWII_PIPER_VOICE") or DEFAULT_VOICE).strip()
    return voice if voice and os.path.isfile(voice) else ""


def piper_binary() -> str:
    """The piper.exe used to synthesise in a subprocess (fallback path)."""
    cand = (os.environ.get("SAWII_PIPER_EXE")
            or r"C:\Users\anasa\localtts\piper\piper.exe")
    return cand if cand and os.path.isfile(cand) else (shutil.which("piper") or "")


class PiperTTS:
    """Resident Piper voice. Loads once, synthesises in a worker thread.

    ONNX synthesis is CPU-bound and blocks the event loop, which would stall
    every other live turn, so it always runs through ``asyncio.to_thread``.
    """

    def __init__(self, voice_path: str = "", *, head_ms: int = HEAD_MS):
        self.voice_path = voice_path or piper_voice_path()
        self.head_ms = head_ms
        self._voice: Any = None
        self._load_task: asyncio.Task | None = None
        self.load_ms: int | None = None
        self.last_first_byte_ms: int | None = None

    @property
    def loaded(self) -> bool:
        return self._voice is not None

    async def warm(self) -> bool:
        """Load the voice off the critical path. Safe to await; idempotent."""
        if self._voice is not None:
            return True
        if self._load_task is None or self._load_task.done():
            self._load_task = asyncio.create_task(self._load())
        try:
            return bool(await self._load_task)
        except Exception:  # noqa: BLE001
            return False

    async def _load(self) -> bool:
        t0 = time.perf_counter()
        def _open() -> Any:
            from piper import PiperVoice  # type: ignore
            return PiperVoice.load(self.voice_path)
        self._voice = await asyncio.to_thread(_open)
        self.load_ms = round((time.perf_counter() - t0) * 1000)
        log.info("piper voice loaded in %s ms (%s)", self.load_ms,
                 os.path.basename(self.voice_path))
        return True

    def _synth_blocking(self, text: str) -> tuple[bytes, int, list[bytes]]:
        """Returns (head_wav, sample_rate, tail_clips). Runs off the loop."""
        from .stream_tts import wav_from_pcm

        rate = getattr(getattr(self._voice, "config", None), "sample_rate", 22050) or 22050
        head_limit = int(rate * 2 * self.head_ms / 1000)
        head = bytearray()
        tail: list[bytes] = []
        for chunk in self._voice.synthesize(text):
            raw = chunk.audio_int16_bytes
            if not raw:
                continue
            if len(head) < head_limit:
                head += raw
            else:
                tail.append(bytes(raw))
        return wav_from_pcm(bytes(head), rate), rate, [wav_from_pcm(b, rate) for b in tail]

    async def synthesize_head(self, text: str) -> tuple[bytes, str, list[tuple[bytes, str]]]:
        """(head wav, mime, [(wav, mime), ...]) — same shape as the remote path."""
        if self._voice is None:
            raise RuntimeError("piper voice not loaded")
        t0 = time.perf_counter()
        head, _rate, tail = await asyncio.to_thread(self._synth_blocking, text)
        self.last_first_byte_ms = round((time.perf_counter() - t0) * 1000)
        if not head:
            raise RuntimeError("piper produced no audio")
        return head, "audio/wav", [(c, "audio/wav") for c in tail]


async def _smoke() -> None:
    import json
    import sys

    tts = PiperTTS()
    if not tts.voice_path:
        print(json.dumps({"ok": False, "error": "no piper voice"}))
        sys.exit(1)
    t0 = time.perf_counter()
    await tts.warm()
    print(json.dumps({"event": "warm", "ms": round((time.perf_counter() - t0) * 1000)}))
    for text in ("Okay, one second.", "Hey, that's a good question, let me think.",
                 "Good, thanks for asking."):
        head, mime, tail = await tts.synthesize_head(text)
        print(json.dumps({"in": text[:28], "head_bytes": len(head),
                          "tail_clips": len(tail),
                          "first_clip_ms": tts.last_first_byte_ms}))


if __name__ == "__main__":
    asyncio.run(_smoke())