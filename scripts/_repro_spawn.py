"""Spawn-endpoint probe (the path the user says must NOT be needed).

Usage: python scripts/_repro_spawn.py <port> <key> [secs]
"""
import json
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8802"
KEY = sys.argv[2] if len(sys.argv) > 2 else ""
SECS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
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


court = req("POST", "/v1/courts", {"title": "spawn-probe", "kind": "desktop"})
cid = (court or {}).get("id")
print("court:", cid, flush=True)
res = req("POST", f"/v1/courts/{cid}/spawn", {"goal": "Write a file called hello.txt containing hi"})
print("spawn response:", json.dumps(res or {})[:600], flush=True)

t0 = time.time()
while time.time() - t0 < SECS:
    time.sleep(3)
    c = req("GET", f"/v1/courts/{cid}") or {}
    hk = req("GET", "/v1/hands") or {}
    lk = (req("GET", "/v1/link") or {}).get("links", {}).get("hands-engine", {})
    print(f"[t+{time.time() - t0:4.0f}s] status={c.get('status')} lane={c.get('lane')!r} "
          f"err={str(c.get('error'))[:120]!r} | hands={hk.get('connected')} "
          f"{list((hk.get('places') or {}).keys())} | link.active={lk.get('active_lanes')} "
          f"total={lk.get('total_lanes')}", flush=True)