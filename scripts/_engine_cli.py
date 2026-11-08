"""Run `python sawii_hands.py ...` exactly as Link does, output to a FILE, -u.

Usage: python scripts/_engine_cli.py <port> <key> <lane>
"""
import json
import subprocess
import sys
import time
import urllib.request

port, key, lane = sys.argv[1], sys.argv[2], sys.argv[3]
log = rf"C:\Users\anasa\sawii-persona\tmp\engine_cli_{lane}.log"
args = [sys.executable, "-u", "sawii_hands.py",
        "--url", f"http://127.0.0.1:{port}", "--key", key,
        "--approve", "auto", "--place", "computer", "--lane", lane,
        "--root", r"C:\Users\anasa\AppData\Local\Temp\hands-root"]
env = {k: v for k, v in __import__("os").environ.items()}
fh = open(log, "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(args, cwd=r"c:\Users\anasa\sawii-persona\hands",
                     stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
print("pid", p.pid, flush=True)


def hands():
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/hands")
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


t0 = time.time()
while time.time() - t0 < 24:
    time.sleep(3)
    h = hands()
    body = open(log, encoding="utf-8", errors="replace").read().strip()
    print(f"[t+{time.time() - t0:4.0f}s] alive={p.poll() is None} rc={p.returncode} "
          f"connected={h.get('connected')} places={list((h.get('places') or {}).keys())} "
          f"out={body[:120]!r}", flush=True)
    if h.get("connected") or p.poll() is not None:
        break
p.terminate()
try:
    p.wait(timeout=5)
except Exception:  # noqa: BLE001
    p.kill()
print("--- file ---")
print(open(log, encoding="utf-8", errors="replace").read()[:1500] or "(nothing)")