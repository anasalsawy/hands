"""End-to-end: the REAL app, the REAL deck, two live cards over HTTP.

    python scripts/court_e2e.py

Builds the actual Starlette app (persona/api/app.py + Runtime + Mind) on a
throwaway store, creates a real persona, then drives the court endpoints over
the app's own ASGI transport:

    POST   /v1/courts            x2      -> two cards
    POST   /v1/courts/{id}/spawn x2      -> both become live at once
    GET    /v1/courts                    -> count=2, running=2
    POST   /v1/courts/{id}/goal          -> steer one
    POST   /v1/courts/{id}/voice         -> voice hook
    POST   /v1/courts/{id}/cancel        -> stop one, keep its card

The Link registry is not registered here, so spawn makes each card a live
*surface* (motor="none"); the concurrency of real engine lanes is proven
separately by scripts/court_selftest.py (peak concurrent = 2).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["SAWII_COURTS"] = "1"
os.environ["SAWII_REQUIRE_INITIAL_SEMANTIC"] = "0"

from starlette.testclient import TestClient  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hands_e2e_"))
    os.environ["SAWII_DATA_DIR"] = str(tmp)
    os.environ["SAWII_DB_PATH"] = str(tmp / "sawii.db")

    from persona.api.app import Runtime, create_app
    from persona.config import load_settings
    from persona.identity import create_persona

    settings = load_settings()
    rt = Runtime(settings)
    app = create_app(settings, rt)

    # Create the persona first (its own tiny loop + connection), then drive
    # the app over its ASGI transport.
    import asyncio
    pid, key = asyncio.run(create_persona(rt.mind.store, name="E2E"))

    with TestClient(app) as client:
        hdr = {"Authorization": f"Bearer {key}"}

        r = client.get("/v1/courts", headers=hdr)
        assert r.status_code == 200, r.text
        assert r.json()["count"] == 0, r.text
        print(f"persona id={pid} key={key[:8]}…")
        print("  GET  /v1/courts (empty)              OK")

        ids = []
        for i, kind in enumerate(("desktop", "window")):
            r = client.post("/v1/courts", headers=hdr, json={"kind": kind, "title": f"Card {i+1}"})
            assert r.status_code == 201, r.text
            ids.append(r.json()["id"])
        print(f"  POST /v1/courts x2                   OK  {ids}")

        for cid in ids:
            r = client.post(f"/v1/courts/{cid}/spawn", headers=hdr, json={"goal": "open notepad"})
            assert r.status_code == 202, r.text
            assert r.json()["status"] == "running", r.text
        print("  POST .../spawn x2 (both live)        OK")

        r = client.get("/v1/courts", headers=hdr)
        snap = r.json()
        assert snap["count"] == 2 and snap["running"] == 2, snap
        print(f"  GET  /v1/courts (count=2 running=2)  OK")

        r = client.post(f"/v1/courts/{ids[0]}/goal", headers=hdr, json={"goal": "type hello"})
        assert r.status_code == 202 and r.json()["goal"] == "type hello", r.text
        print("  POST .../goal                        OK")

        r = client.post(f"/v1/courts/{ids[0]}/voice", headers=hdr, json={"action": "speak", "text": "hi"})
        assert r.status_code == 200, r.text
        print("  POST .../voice                       OK")

        r = client.post(f"/v1/courts/{ids[1]}/cancel", headers=hdr, json={})
        assert r.status_code == 200 and r.json()["status"] == "stopped", r.text
        print("  POST .../cancel (card kept)          OK")

    print("\ncourt end-to-end (real app) PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
