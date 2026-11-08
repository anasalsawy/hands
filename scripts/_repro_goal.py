"""Reproduce the measured failure, then prove the fix.

Usage: python scripts/_repro_goal.py <port> <key> <goal>
Prints court view + /v1/link + /v1/hands over time. Makes NO spawn call.
"""
import json
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8802"
KEY = sys.argv[2] if len(sys.argv) > 2 else ""
GOAL = sys.argv[3] if len(sys.argv) > 3 else "Write a file called hello.txt containing hi"
SECS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
BASE = f"http://127.0.0.1:{port}"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("Authorization", f"Bearer {KEY}")
    r.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
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


court = req("POST", "/v1/courts", {"title": "goal-only", "kind": "desktop"})
print("court created:", json.dumps(court or {})[:400], flush=True)
cid = (court or {}).get("id")
if not cid:
    sys.exit(1)

res = req("POST", f"/v1/courts/{cid}/goal", {"goal": GOAL})
print("goal response:", json.dumps(res or {})[:600], flush=True)

t0 = time.time()
tick = 0
while time.time() - t0 < SECS:
    time.sleep(3)
    tick += 1
    c = req("GET", f"/v1/courts/{cid}") or {}
    hk = req("GET", "/v1/hands") or {}
    lk = req("GET", "/v1/link") or {}
    links = lk.get("links", {}).get("hands-engine", {})
    places = list((hk.get("places") or {}).keys())
    print(f"[t+{time.time() - t0:4.0f}s] status={c.get('status')} lane={c.get('lane')!r} "
          f"err={str(c.get('error'))[:90]!r} | hands.connected={hk.get('connected')} "
          f"places={places} | link.active={links.get('active_lanes')} "
          f"total={links.get('total_lanes')}", flush=True)

print("\nfinal court:", json.dumps(req("GET", f"/v1/courts/{cid}") or {}, indent=1)[:800], flush=True)
print("\nfinal hands:", json.dumps(req("GET", "/v1/hands") or {}, indent=1)[:1200], flush=True)
print("\nfinal link:", json.dumps(req("GET", "/v1/link") or {}, indent=1)[:900], flush=True)