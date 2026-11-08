"""Local-first speech through the real SpeechIO: head clip latency, end to end.

Answers one question with a number: with the voice preloaded, how long from
"say this" to the first playable WAV frame, with no network at all.

Run: python persona/voice/bench/local_speech_bench.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from persona.config import load_settings  # noqa: E402
from persona.voice.local_tts import PiperTTS, piper_voice_path  # noqa: E402

TEXTS = [
    "Okay, one second.",
    "Good, thanks for asking.",
    "Hey, that's a good question, let me think for a second.",
    "Hey there, what's up?",
    "Morning. What are we doing today?",
]


async def main() -> None:
    if not piper_voice_path():
        print(json.dumps({"ok": False, "error": "no piper voice found"}))
        return
    tts = PiperTTS()
    t0 = asyncio.get_event_loop().time()
    ok = await asyncio.wait_for(tts.warm(), timeout=120)
    print(json.dumps({"event": "warm", "ok": ok,
                      "ms": round((asyncio.get_event_loop().time() - t0) * 1000)}), flush=True)
    heads = []
    for i in range(12):
        text = TEXTS[i % len(TEXTS)]
        t1 = asyncio.get_event_loop().time()
        head, mime, tail = await asyncio.wait_for(tts.synthesize_head(text), timeout=60)
        ms = round((asyncio.get_event_loop().time() - t1) * 1000)
        heads.append(ms)
        print(json.dumps({"text": text[:26], "head_ms": ms, "head_bytes": len(head),
                          "mime": mime, "tail_clips": len(tail)}), flush=True)
    print("\n=== LOCAL SPEECH HEAD (PiperTTS.synthesize_head) ===")
    s = sorted(heads)
    print(json.dumps({"p50": round(statistics.median(heads)), "min": min(s), "max": max(s),
                      "p95": s[int(0.95 * (len(s) - 1))]}))
    # The whole point: does SpeechIO route here and report it?
    from persona.voice.realtime import SpeechIO  # noqa: E402
    s2 = load_settings()
    io = SpeechIO(s2.asr, s2.tts, s2.tts_voice)
    await io.warm_local_tts()
    t2 = asyncio.get_event_loop().time()
    try:
        head, mime, tail = await asyncio.wait_for(io.synthesize_head("Okay, one second."),
                                                 timeout=60)
        print(json.dumps({"event": "speechio_head", "ms": round((asyncio.get_event_loop().time() - t2) * 1000),
                          "mime": mime, "bytes": len(head), "tail": len(tail),
                          "has_tts": io.has_tts}))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "speechio_head", "error": type(exc).__name__}))


if __name__ == "__main__":
    asyncio.run(main())