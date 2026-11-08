"""End-to-end proof: ONE goal, no spawn call, lanes + engines appear by themselves.

The only client calls here are POST /v1/courts (a card) and POST
/v1/courts/{id}/goal (the goal). There is deliberately NO call to
/v1/courts/{id}/spawn anywhere in this file: concurrency is the mind's job.

Usage: python scripts/_e2e_goal.py <port> <key> <goal> [seconds]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8802"
KEY = sys.argv[2] if len(sys.argv) > 2 else ""
GOAL = sys.argv[3] if len(sys.argv) > 3 else "Write a file called hello.txt containing hi"
SECS = int(sys.argv[4]) if len(sys.argv) > 4 else 90
BASE = f"http://127.0.0.1:{PORT}"
SPAWN_CALLS = 0


def req(method, path, body=None):
    global SPAWN_CALLS
    if path.rstrip("/").endswith("/spawn"):
        SPAWN_CALLS += 1  # must stay 0: this script proves the client never spawns
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("Authorization", f"Bearer {KEY}")
    r.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        print(f"{method} {path} -> {e.code} {raw[:300]}")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"{method} {path} -> {type(e).__name__}: {e}")
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw}


def hands_lanes(hk: dict) -> list[str]:
    """The runner keys /v1/hands currently shows (place:lane)."""
    return sorted((hk.get("places") or {}).keys())


print("=" * 78)
print("STEP 1  create a card (no spawn, no lane, no engine yet)")
court = req("POST", "/v1/courts", {"title": "e2e", "kind": "desktop"})
cid = (court or {}).get("id")
print(f"court={cid} status={(court or {}).get('status')} lane={(court or {}).get('lane')!r}")
if not cid:
    sys.exit(1)

print("\nSTEP 2  POST the goal. This is ALL the user does.")
t0 = time.time()
res = req("POST", f"/v1/courts/{cid}/goal", {"goal": GOAL})
print(f"goal response ({time.time() - t0:.1f}s): status={(res or {}).get('status')} "
      f"lane={(res or {}).get('lane')} error={str((res or {}).get('error'))[:120]!r}")

print("\nSTEP 3  watch lanes + runners appear with no further client action")
prev: set[str] = set()
last_status = ""
peak = 0
peak_lanes: list[str] = []
while time.time() - t0 < SECS:
    time.sleep(3)
    courts = (req("GET", "/v1/courts") or {}).get("courts", [])
    hk = req("GET", "/v1/hands") or {}
    lk = (req("GET", "/v1/link") or {}).get("links", {}).get("hands-engine", {})
    lanes = hands_lanes(hk)
    mine = next((c for c in courts if c.get("id") == cid), {})
    if len(lanes) > peak:  # concurrency the mind chose, seen while it is live
        peak, peak_lanes = len(lanes), lanes
    if lanes != prev or mine.get("status") != last_status:
        prev, last_status = set(lanes), mine.get("status")
        print(f"[t+{time.time() - t0:4.0f}s] cards={len(courts)} "
              f"this_card={mine.get('status')}/lane={str(mine.get('lane'))[:12]!r} "
              f"err={str(mine.get('error'))[:70]!r}")
        print(f"           /v1/hands connected={hk.get('connected')} runners={lanes}")
        print(f"           /v1/link  active_lanes={lk.get('active_lanes')} "
              f"total={lk.get('total_lanes')}")
    if mine.get("status") in ("done", "failed", "stopped") and lanes:
        break

print("\nSTEP 4  final state")
final_courts = (req("GET", "/v1/courts") or {}).get("courts", [])
hk = req("GET", "/v1/hands") or {}
lk = (req("GET", "/v1/link") or {}).get("links", {}).get("hands-engine", {})
for c in final_courts:
    meta = c.get("meta", {}) or {}
    print(f"  card {c['id']} status={c['status']:9s} lane={str(c['lane'])[:12]:12s} "
          f"connected={meta.get('connected')} err={str(c.get('error'))[:80]!r}")
    for tool in (meta.get("last_tools") or []):
        print(f"      tool: {str(tool)[:150]}")
    if meta.get("result"):
        print(f"      said: {str(meta['result'])[:150]}")
print(f"  PEAK concurrent runners observed while the goal ran: {peak} -> {peak_lanes}")
print(f"  /v1/hands connected={hk.get('connected')} runners={hands_lanes(hk)}")
for k, v in (hk.get("places") or {}).items():
    print(f"    {k}: host={v.get('host')} lane={v.get('lane')} tools={v.get('tools')}")
print(f"  /v1/link active_lanes={lk.get('active_lanes')} total={lk.get('total_lanes')}")

# The honesty check the whole design rests on.
print("\nSTEP 5  honesty checks")
live = {str(v.get("lane")) for v in (hk.get("places") or {}).values()}
liars = [c["id"] for c in final_courts
         if c["status"] in ("running", "waiting", "spawning") and c.get("lane") not in live]
print(f"  cards claiming work with no live runner: {liars or 'none'}")
running_hands = {k.split(':', 1)[1] for k in hands_lanes(hk)}
print(f"  /v1/link active lanes == /v1/hands runner lanes: "
      f"{sorted(set(lk.get('active_lanes') or [])) == sorted(running_hands)}")
print(f"  spawn endpoints the client called: {SPAWN_CALLS} (must be 0)")