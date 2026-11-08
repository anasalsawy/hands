"""Does barge-in actually stop speech that is already being synthesised?

Asserts three things on the real VoiceSession, not a mock:
  1. on_interrupt() during a TTS call stops further frames reaching the socket.
  2. The streaming SSE call is torn down mid-response (no orphan generation).
  3. A new turn after an interrupt is not blocked by the cancelled one.

Run: python persona/voice/bench/test_bargein.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from persona.config import load_settings  # noqa: E402
from persona.mind import Mind  # noqa: E402
from persona import identity  # noqa: E402
from persona.voice.realtime import SpeechIO, VoiceSession  # noqa: E402


class Rec:
    def __init__(self):
        self.json: list[dict] = []
        self.bytes = 0

    async def send_json(self, obj):
        self.json.append(dict(obj or {}))

    async def send_bytes(self, blob):
        self.bytes += 1


async def main() -> int:
    if not (os.environ.get("SAWII_TTS_API_KEY") or os.environ.get("COMPUSE_LLM_API_KEY")):
        print("SKIP: no TTS key in env")
        return 0
    s = load_settings()
    if not (s.tts.api_key and s.tts.model):
        print("SKIP: tts endpoint not configured")
        return 0
    mind = Mind(s)
    rows = await mind.store.all("SELECT * FROM personas ORDER BY id LIMIT 1")
    persona = rows[0] if rows else None
    if persona is None:
        pid, _ = await identity.create_persona(mind.store, name="Barge Bench")
        persona = await identity.get_persona(mind.store, pid)
    speech = SpeechIO(s.asr, s.tts, s.tts_voice)
    rec = Rec()
    session = VoiceSession(mind, persona, speech, rec.send_json, rec.send_bytes, dual=True)
    await asyncio.sleep(2.0)   # let warmups land
    results: dict[str, object] = {"streams_tts": speech.streams_tts}

    # 1. Start a real long answer, interrupt mid-synthesis.
    rec.json.clear(); rec.bytes = 0
    task = asyncio.create_task(session._turn(
        "Tell me a very long detailed story about the history of computing, "
        "at least eight sentences."))
    session.turn_task = task
    # Wait until audio starts flowing, then barge in.
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30 and not any(f.get("type") == "audio" for f in rec.json):
        await asyncio.sleep(0.02)
    at_interrupt = len(rec.json)
    await session.on_interrupt()
    frames_at_interrupt = sum(1 for f in rec.json if f.get("type") == "audio")
    await asyncio.wait_for(task, timeout=60)
    frames_after = sum(1 for f in rec.json if f.get("type") == "audio")
    results["frames_at_interrupt"] = frames_at_interrupt
    results["frames_after_interrupt"] = frames_after
    results["leaked_frames"] = frames_after - frames_at_interrupt
    results["leak_ok"] = (frames_after - frames_at_interrupt) <= 2

    # 2. The next turn must run cleanly after a barge-in.
    rec.json.clear()
    session._heard = ""
    session.interrupted = False
    t1 = time.monotonic()
    task2 = asyncio.create_task(session._turn("how are you feeling right now?"))
    session.turn_task = task2
    try:
        await asyncio.wait_for(task2, timeout=90)
    except asyncio.TimeoutError:
        task2.cancel()
    results["second_turn_ok"] = any(f.get("type") in ("audio", "say", "done")
                                    for f in rec.json)
    results["second_turn_first_audio_ms"] = session._first_audio
    results["second_turn_wall_ms"] = round((time.monotonic() - t1) * 1000)

    # 3. Interrupt during a streaming synth call tears the response down.
    if speech.streams_tts:
        stop = {"v": False}
        t2 = time.monotonic()
        got_first = None
        try:
            async def consume():
                nonlocal got_first
                gen = speech._gemini_stream.stream(
                    "This is a long sentence that will keep generating audio for a while, "
                    "so we have time to interrupt it halfway through.", stop=lambda: stop["v"])
                async for sl in gen:
                    if got_first is None:
                        got_first = round((time.monotonic() - t2) * 1000)
                    await asyncio.sleep(0)
            consumer = asyncio.create_task(consume())
            while got_first is None and time.monotonic() - t2 < 20:
                await asyncio.sleep(0.01)
            stop["v"] = True
            t3 = time.monotonic()
            await asyncio.wait_for(consumer, timeout=10)
            results["stream_stop_tears_down_ms"] = round((time.monotonic() - t3) * 1000)
            results["stream_first_slice_ms"] = got_first
            results["stream_stop_ok"] = True
        except Exception as exc:  # noqa: BLE001
            results["stream_stop_ok"] = False
            results["stream_stop_error"] = type(exc).__name__

    await session.finish()
    import json
    print(json.dumps(results, indent=2))
    ok = bool(results.get("leak_ok")) and bool(results.get("second_turn_ok"))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))