"""Is Piper's crash/quiet-exit specific to worker threads, or the machine?

piper_resident.py synthesises fine in the main thread. local_tts.py puts the
same call in asyncio.to_thread and the process vanished silently. This separates
those two hypotheses in one run, with a heartbeat so a hang is distinguishable
from an exit.

Run: python persona/voice/bench/piper_thread_probe.py
"""
from __future__ import annotations

import faulthandler
import sys
import threading
import time

faulthandler.enable()
VOICE = r"C:\Users\anasa\localtts\voices\en_US-lessac-low.onnx"


def beat() -> None:
    time.sleep(10)
    print("[beat] still alive after 10s", flush=True)
    time.sleep(20)
    print("[beat] still alive after 30s", flush=True)


def main() -> None:
    threading.Thread(target=beat, daemon=True).start()
    print("importing piper", flush=True)
    from piper import PiperVoice
    print("loading", flush=True)
    t0 = time.perf_counter()
    v = PiperVoice.load(VOICE)
    print(f"loaded in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
    t0 = time.perf_counter()
    n = 0
    for c in v.synthesize("Okay, one second."):
        n += len(c.audio_int16_bytes)
    print(f"MAIN THREAD synth ok: {n} bytes in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    out: dict = {}

    def worker() -> None:
        try:
            t1 = time.perf_counter()
            k = 0
            for c in v.synthesize("Okay, one second."):
                k += len(c.audio_int16_bytes)
            out["bytes"] = k
            out["ms"] = round((time.perf_counter() - t1) * 1000)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"

    th = threading.Thread(target=worker)
    th.start()
    th.join(60)
    print("WORKER THREAD:", out or "TIMEOUT (still running)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stdout.flush()