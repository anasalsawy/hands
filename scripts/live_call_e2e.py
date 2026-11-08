"""End-to-end live-call exercise against a running Hands stitch server.

Dry-run path (no Twilio): place outbound → simulate caller turns → detail → end.
Also probes /v1/voice WebSocket hello and /v1/voice-status.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request


def _req(method: str, url: str, *, headers: dict | None = None, body: dict | None = None, timeout: float = 60.0):
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype or raw[:1] in (b"{", b"["):
                return resp.status, json.loads(raw.decode("utf-8") or "null")
            return resp.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except Exception:
            payload = raw.decode("utf-8", "replace")
        return exc.code, payload


async def voice_ws_probe(base: str, key: str) -> dict:
    try:
        import websockets
    except ImportError:
        return {"ok": False, "error": "websockets not installed"}

    host = base.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    url = f"{host}/v1/voice"
    out: dict = {"ok": False, "url": url, "frames": []}
    try:
        async with websockets.connect(url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps({"type": "hello", "key": key, "playback_ack": True}))
            for _ in range(8):
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                if isinstance(msg, bytes):
                    out["frames"].append({"type": "binary", "n": len(msg)})
                    continue
                frame = json.loads(msg)
                out["frames"].append({"type": frame.get("type"), "keys": sorted(frame.keys())[:12]})
                if frame.get("type") in ("ready", "error", "done", "message"):
                    out["ready_frame"] = frame
                    out["ok"] = frame.get("type") != "error"
                    break
            # One short text turn if we got ready
            if out.get("ok"):
                await ws.send(json.dumps({"type": "text", "text": "Hi — just checking the live voice line. One short sentence."}))
                pending_seq = None
                for _ in range(40):
                    msg = await asyncio.wait_for(ws.recv(), timeout=45)
                    if isinstance(msg, bytes):
                        out["frames"].append({"type": "binary", "n": len(msg)})
                        # Server requires playback_done when hello.playback_ack=true
                        if pending_seq is not None:
                            await ws.send(json.dumps({"type": "playback_done", "seq": pending_seq}))
                            pending_seq = None
                        continue
                    frame = json.loads(msg)
                    out["frames"].append({"type": frame.get("type")})
                    if frame.get("type") == "audio":
                        pending_seq = frame.get("seq")
                    if frame.get("type") == "done":
                        out["voice_reply"] = frame.get("text") or frame.get("final")
                        break
                    if frame.get("type") == "error":
                        out["ok"] = False
                        out["error"] = frame
                        break
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("HANDS_BASE", "http://127.0.0.1:8822"))
    ap.add_argument("--admin", default=os.environ.get("SAWII_ADMIN_TOKEN", ""))
    ap.add_argument("--key", default=os.environ.get("SAWII_PERSONA_KEY", ""))
    ap.add_argument("--skip-voice-ws", action="store_true")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    report: dict = {"base": base, "steps": []}

    def step(name: str, ok: bool, detail):
        report["steps"].append({"name": name, "ok": ok, "detail": detail})
        flag = "OK" if ok else "FAIL"
        print(f"[{flag}] {name}: {json.dumps(detail, default=str)[:500]}")

    # health
    code, body = _req("GET", f"{base}/health")
    step("health", code == 200, {"status": code, "body": body})

    code, body = _req("GET", f"{base}/v1/link")
    step("v1_link", code == 200 and isinstance(body, dict), {"status": code, "body": body})

    code, body = _req("GET", f"{base}/v1/voice-status")
    step("v1_voice_status", code == 200 and isinstance(body, dict), {"status": code, "body": body})

    # persona key
    key = args.key
    if not key:
        if not args.admin:
            step("persona_key", False, {"error": "need --key or SAWII_ADMIN_TOKEN / --admin"})
            print(json.dumps(report, indent=2, default=str))
            return 2
        code, body = _req(
            "POST",
            f"{base}/admin/personas",
            headers={"Authorization": f"Bearer {args.admin}"},
            body={
                "name": "Hands Live Call Probe",
                "user_name": "Anas",
                "timezone": "Africa/Cairo",
                "definition": "You are a warm companion on a phone call. Keep replies to one short sentence.",
            },
        )
        ok = code in (200, 201) and isinstance(body, dict) and body.get("key")
        step("create_persona", ok, {"status": code, "id": (body or {}).get("id") if isinstance(body, dict) else None})
        if not ok:
            print(json.dumps(report, indent=2, default=str))
            return 1
        key = body["key"]
    report["persona_key_prefix"] = key[:7] + "…"

    auth = {"Authorization": f"Bearer {key}"}

    # calls list (twilio_live flag)
    code, body = _req("GET", f"{base}/v1/calls", headers=auth)
    step("calls_list", code == 200, {"status": code, "body": body})

    # place outbound dry-run
    code, body = _req(
        "POST",
        f"{base}/v1/calls/outbound",
        headers=auth,
        body={"to": "+15551234567", "hint": "Confirm the live call path works. Be brief."},
        timeout=90,
    )
    ok = code == 200 and isinstance(body, dict) and body.get("sid")
    step("calls_outbound", ok, {"status": code, "body": body})
    if not ok:
        print(json.dumps(report, indent=2, default=str))
        return 1
    sid = body["sid"]
    mode = body.get("mode")
    report["call_sid"] = sid
    report["call_mode"] = mode

    # simulate 2 turns
    for i, heard in enumerate([
        "Hello, is this working?",
        "Great — please say one word to confirm and we'll hang up.",
    ], start=1):
        code, body = _req(
            "POST",
            f"{base}/v1/calls/{sid}/simulate",
            headers=auth,
            body={"heard": heard},
            timeout=120,
        )
        step(f"calls_simulate_{i}", code == 200, {"status": code, "heard": heard, "body": body})

    code, body = _req("GET", f"{base}/v1/calls/{sid}", headers=auth)
    step("calls_detail", code == 200, {"status": code, "events": (body or {}).get("events") if isinstance(body, dict) else body})

    code, body = _req("POST", f"{base}/v1/calls/{sid}/end", headers=auth)
    step("calls_end", code == 200, {"status": code, "body": body})

    # voice websocket
    if not args.skip_voice_ws:
        voice = asyncio.run(voice_ws_probe(base, key))
        step("voice_ws", bool(voice.get("ok")), voice)

    # sidecar optional
    try:
        code, body = _req("GET", "http://127.0.0.1:18766/health", timeout=2)
        step("voice_sidecar_health", code == 200, {"status": code, "body": body})
    except Exception as exc:
        step("voice_sidecar_health", False, {"error": str(exc)})

    failed = [s for s in report["steps"] if not s["ok"]]
    report["passed"] = len(report["steps"]) - len(failed)
    report["failed"] = len(failed)
    report["ok"] = len(failed) == 0 or (
        # sidecar optional; Twilio live expected false without creds
        all(s["name"] != "voice_sidecar_health" for s in failed)
        and not any(s["name"].startswith("calls_") or s["name"] in ("health", "voice_ws", "create_persona") for s in failed)
    )
    # stricter: core call path must pass
    core = {"health", "calls_outbound", "calls_simulate_1", "calls_end"}
    report["core_ok"] = all(next(s for s in report["steps"] if s["name"] == n)["ok"] for n in core if any(s["name"] == n for s in report["steps"]))
    print("--- summary ---")
    print(json.dumps({k: report[k] for k in ("core_ok", "passed", "failed", "call_mode", "call_sid")}, indent=2))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["core_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
