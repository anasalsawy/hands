"""Piper with the voice held in memory (the only shape that is fast enough).

The CLI pays ~1.85 s of model load per invocation, which alone breaks the
budget. Loaded once, synthesis is real-time-factor 0.17 on this CPU.

Run: python persona/voice/bench/piper_resident.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time

VOICE = r"C:\Users\anasa\localtts\voices\en_US-lessac-low.onnx"

TEXTS = [
    "Okay, one second.",
    "Good, thanks for asking.",
    "Hey, that's a good question, let me think for a second.",
    "Hey there, what's up?",
]


def main() -> None:
    try:
        from piper import PiperVoice
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"import {type(exc).__name__}"}))
        sys.exit(1)
    t0 = time.perf_counter()
    voice = PiperVoice.load(VOICE)
    load_ms = round((time.perf_counter() - t0) * 1000)
    print(json.dumps({"event": "load", "ms": load_ms}), flush=True)

    rows = []
    for i in range(12):
        text = TEXTS[i % len(TEXTS)]
        t0 = time.perf_counter()
        first = None
        nbytes = 0
        for chunk in voice.synthesize(text):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            nbytes += len(chunk.audio_int16_bytes)
        full = (time.perf_counter() - t0) * 1000
        r = {"text": text[:26], "first_byte_ms": round(first or full),
             "full_ms": round(full), "bytes": nbytes}
        rows.append(r)
        print(json.dumps(r), flush=True)

    fb = [r["first_byte_ms"] for r in rows]
    print("\n=== PIPER RESIDENT (voice preloaded) ===")
    print(f"load_ms={load_ms}")
    print(f"first_byte p50={round(statistics.median(fb))} p95={sorted(fb)[int(0.95*(len(fb)-1))]} "
          f"min={min(fb)} max={max(fb)}")
    print(f"full p50={round(statistics.median([r['full_ms'] for r in rows]))}")


if __name__ == "__main__":
    main()