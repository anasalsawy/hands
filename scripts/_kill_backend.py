"""Stop ONLY the stitch.py backends this operator session started on temp roots.

Refuses to touch anything whose command line does not name the verification
root, so the other live agents' processes are never in scope.

Usage: python scripts/_kill_backend.py
"""
import sys

import psutil

NEEDLE = "--root C:\\Users\\anasa\\AppData\\Local\\Temp\\hands-root"
me = set()
for p in psutil.process_iter(["pid", "cmdline"]):
    try:
        cmd = " ".join(p.info.get("cmdline") or [])
    except Exception:  # noqa: BLE001
        continue
    if NEEDLE in cmd and ("stitch.py" in cmd or "sawii_hands.py" in cmd):
        me.add(p)

if "--dry" in sys.argv:
    for p in sorted(me, key=lambda x: x.pid):
        print(f"would stop {p.pid}")
    raise SystemExit(0)

for p in sorted(me, key=lambda x: x.pid, reverse=True):
    try:
        p.terminate()
        try:
            p.wait(timeout=6)
        except psutil.TimeoutExpired:
            p.kill()
        print(f"stopped {p.pid}")
    except Exception as exc:  # noqa: BLE001
        print(f"could not stop {p.pid}: {type(exc).__name__}")
print(f"done ({len(me)} process(es) were in scope)")