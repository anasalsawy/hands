"""Honesty test: kill a live engine and prove the court/link stop lying.

Usage: python scripts/_e2e_death.py <port> <key>
"""
import json
import sys
import time
import urllib.request

import psutil

PORT, KEY = sys.argv[1], sys.argv[2]
BASE = f"http://127.0.0.1:{PORT}"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("Authorization", f"Bearer {KEY}")
    r.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=45) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"{method} {path} -> {type(exc).__name__}: {exc}")
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw}


court = req("POST", "/v1/courts", {"title": "death test"})
cid = court["id"]
print("card", cid)
res = req("POST", f"/v1/courts/{cid}/goal",
          {"goal": "Open a shell and run exactly: Start-Sleep -Seconds 40; Write-Output SLEPT. "
                   "Then tell me what the command printed."})
print("goal ->", res.get("status"), res.get("lane"))

t0 = time.time()
lane = res.get("lane")
while time.time() - t0 < 40:
    time.sleep(2)
    h = req("GET", "/v1/hands") or {}
    if lane in {v.get("lane") for v in (h.get("places") or {}).values()}:
        print(f"[t+{time.time() - t0:.0f}s] runner live for lane {lane}")
        break
# Wait until R has actually dispatched a tool into that runner, so the kill
# lands mid-work rather than before the engine had anything to do.
for _ in range(10):
    time.sleep(2)
    mine = next((c for c in (req("GET", "/v1/courts") or {}).get("courts", [])
                 if c.get("id") == cid), {})
    if (mine.get("meta") or {}).get("last_tools"):
        print(f"[t+{time.time() - t0:.0f}s] tool in flight: "
              f"{mine['meta']['last_tools']}")
        break

me = psutil.Process()
victims = [p for p in psutil.process_iter(["pid", "cmdline"])
           if f"--lane {lane}" in " ".join(p.info.get("cmdline") or [])
           and "sawii_hands.py" in " ".join(p.info.get("cmdline") or [])]
print("engine for this lane:", [v.pid for v in victims])
if not victims:
    print("NOT HONEST: no engine was ever started for this lane")
    raise SystemExit(1)
for v in victims:
    v.kill()
print("KILLED the engine(s). Watching for an honest status change.")

last = ""
while time.time() - t0 < 90:
    time.sleep(3)
    cards = (req("GET", "/v1/courts") or {}).get("courts", [])
    mine = next((c for c in cards if c.get("id") == cid), {})
    h = req("GET", "/v1/hands") or {}
    lk = (req("GET", "/v1/link") or {}).get("links", {}).get("hands-engine", {})
    key = f"{mine.get('status')}|{mine.get('lane')}"
    if key != last:
        last = key
        print(f"[t+{time.time() - t0:4.0f}s] card={mine.get('status'):8s} "
              f"err={str(mine.get('error'))[:90]!r}")
        print(f"            /v1/hands runners={sorted((h.get('places') or {}).keys())} "
              f"link active={lk.get('active_lanes')}")
    if mine.get("status") in ("failed", "stopped"):
        print("\nHONEST: the card stopped claiming work the moment its motor died.")
        break
    if mine.get("status") == "done":
        # "done" after we murdered the engine mid-goal would be a lie unless the
        # work genuinely completed first, so show the evidence and judge it.
        print(f"            last_tools={mine.get('meta', {}).get('last_tools')}")
        print(f"            said={str(mine.get('meta', {}).get('result'))[:160]!r}")
        print("\nNOT HONEST: card reports done although its engine was killed mid-goal.")
        break
else:
    print("\nNOT HONEST: card still reports", mine.get("status"))