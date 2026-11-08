"""Piper local TTS first-byte latency on this box (quota-free, offline).

Piper writes its wav as it synthesises when asked for stdout, so we read the
pipe and time the first bytes. That is exactly what gates audible speech.

Run: python persona/voice/bench/piper_bench.py
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

BINARY = os.environ.get("PIPER_EXE", r"C:\Users\anasa\localtts\piper\piper.exe")
VOICE = os.environ.get("PIPER_VOICE", r"C:\Users\anasa\localtts\voices\en_US-lessac-low.onnx")

TEXTS = [
    "Okay, one second.",
    "Hey, that's a good question, let me think for a second.",
    "Good, thanks for asking.",
]


def run(text: str) -> dict:
    """Pipe stdout: first byte time == the floor for audible speech."""
    t0 = time.perf_counter()
    first = None
    total = 0
    try:
        p = subprocess.Popen([BINARY, "-m", VOICE], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}
    p.stdin.write(text.encode())
    p.stdin.close()
    try:
        while True:
            chunk = p.stdout.read(2048)
            if not chunk:
                break
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            total += len(chunk)
        p.wait(timeout=30)
    except Exception as exc:  # noqa: BLE001
        p.kill()
        return {"ok": False, "error": type(exc).__name__}
    full = (time.perf_counter() - t0) * 1000
    return {"ok": total > 1000, "first_byte_ms": round(first or full),
            "full_ms": round(full), "bytes": total}


def main() -> None:
    if not os.path.exists(BINARY) or not os.path.exists(VOICE):
        print(json.dumps({"ok": False, "error": "piper or voice missing"}))
        sys.exit(1)
    rows = []
    # First call includes model load; that is the honest cold number.
    cold = run(TEXTS[0])
    print("COLD:", json.dumps(cold), flush=True)
    rows.append(cold)
    for _ in range(3):
        for text in TEXTS:
            r = run(text)
            r["text"] = text[:26]
            rows.append(r)
            print(json.dumps(r), flush=True)
    ok = [r for r in rows[1:] if r.get("ok")]
    if not ok:
        print("ALL FAILED")
        sys.exit(1)
    fb = [r["first_byte_ms"] for r in ok]
    print("\n=== PIPER LOCAL (warm) ===")
    print(f"first_byte p50={round(statistics.median(fb))} min={min(fb)} max={max(fb)}")
    print(f"cold (incl. model load): {cold.get('first_byte_ms')} ms")
    print(f"short-clause p50: {round(statistics.median([r['first_byte_ms'] for r in ok if 'one second' in r.get('text','')])) if any('one second' in r.get('text','') for r in ok) else '-'}")


if __name__ == "__main__":
    main()