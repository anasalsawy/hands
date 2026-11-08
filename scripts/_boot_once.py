"""Boot stitch.py with a hard timeout and show exactly how far it got.

PowerShell pipes/redirects have repeatedly swallowed this output, so the child
is run through subprocess with real files and a wall-clock cap.

Usage: python scripts/_boot_once.py [port] [seconds]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
port = sys.argv[1] if len(sys.argv) > 1 else "8802"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 120

out = ROOT.parent / "tmp" / f"boot{port}.out"
out.parent.mkdir(exist_ok=True)
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
args = [sys.executable, "-u", "stitch.py", "local",
        "--root", r"C:\Users\anasa\AppData\Local\Temp\hands-root",
        "--port", port, "--approve", "auto", "--no-voice"]
with open(out, "w", encoding="utf-8", errors="replace") as fh:
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
print(f"pid {p.pid} -> {out}")
t0 = time.time()
while time.time() - t0 < limit:
    time.sleep(5)
    if p.poll() is not None:
        print(f"EXITED early with {p.returncode} after {time.time() - t0:.0f}s")
        break
    try:
        s = socket_open = None
    except Exception:  # noqa: BLE001
        pass
    import socket
    with socket.socket() as sk:
        sk.settimeout(0.5)
        if sk.connect_ex(("127.0.0.1", int(port))) == 0:
            print(f"LISTENING after {time.time() - t0:.0f}s")
            raise SystemExit(0)
print(f"--- output after {int(time.time() - t0)}s ---")
print(out.read_text(encoding="utf-8", errors="replace")[-4000:])
if p.poll() is None:
    print(f"(still running as pid {p.pid})")