"""List or kill leftover sawii_hands.py engine processes (orphans of a dead backend).

Usage: python scripts/_kill_engines.py [--dry]
Only ever touches processes whose command line runs sawii_hands.py. Never
stitch.py, never the mind, never anything else.
"""
import sys

import psutil

killed = 0
for p in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
        cmd = " ".join(p.info.get("cmdline") or [])
    except Exception:  # noqa: BLE001 - process vanished mid-scan
        continue
    if "sawii_hands.py" not in cmd:
        continue
    print(f"engine pid {p.pid}: {cmd[:200]}")
    if "--dry" in sys.argv:
        killed += 1
        continue
    try:
        p.terminate()
        try:
            p.wait(timeout=5)
        except psutil.TimeoutExpired:
            p.kill()
        killed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"  could not stop: {type(exc).__name__}: {exc}")

print(f"{killed} engine process(es) would be killed" if "--dry" in sys.argv
      else f"killed {killed} engine process(es)")