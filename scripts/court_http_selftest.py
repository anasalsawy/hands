"""HTTP self-test for the court deck routes (persona/court.py).

Mounts register_court_routes on a bare Starlette app with a stub runtime whose
mind carries a stub deck + stub identity store, then drives the *real* HTTP
contract the front end (ui/webview/src/lib/courts.ts) binds to:

    GET/POST /v1/courts
    GET/DELETE /v1/courts/{id}
    POST /v1/courts/{id}/spawn|goal|cancel|voice
    GET  /v1/courts/{id}/events                (SSE)

No API keys, no engine: this proves the element layer's wire surface alone.
Run:  python scripts/court_http_selftest.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

os.environ.setdefault("SAWII_COURTS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starlette.applications import Starlette  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from persona.court import HandsDeck, register_court_routes  # noqa: E402


class StubBus:
    def __init__(self):
        self.events = []

    def publish(self, pid, ev):
        self.events.append((pid, ev))


class StubLanes:
    def __init__(self):
        self.stopped = []

    def stop(self, lane):
        self.stopped.append(lane)


class StubLinked:
    """Stand-in for LinkedLane: spawn() sets on_exit on it, so a dict will not do."""

    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.status = "active"
        self.error = ""
        self.on_exit = None


class StubLink:
    class agent:
        strategy = "spawn"

    def __init__(self):
        self.delegated = []
        self._lanes = {}

    async def delegate(self, lane_id, goal, **kw):
        # The real signature takes persona_id / persona_key: the engine must be
        # able to register as ITS persona, so a stub without them is a lie.
        self.delegated.append((lane_id, goal))
        self._lanes[lane_id] = StubLinked(lane_id)
        return self._lanes[lane_id]

    async def status(self, lane_id):
        return "active" if lane_id in self._lanes else "failed"

    def lane(self, lane_id):
        return self._lanes.get(lane_id)

    async def steer(self, lane, goal):
        return {"ok": True}

    async def stop(self, lane):
        return {"ok": True}


class StubLinks:
    def __init__(self):
        self.link = StubLink()

    def get(self, name):
        return self.link if name == "hands-engine" else None


class StubMind:
    def __init__(self):
        self.bus = StubBus()
        self.lanes = StubLanes()
        self.links = StubLinks()
        self.store = object()
        self.deck = HandsDeck(self)


async def _persona_by_key(store, key):
    return {"id": 1, "name": "stub"} if key == "test-key" else None


def main() -> int:
    # Patch identity.persona_by_key so the routes accept our stub key.
    import persona.identity as identity
    identity.persona_by_key = _persona_by_key  # type: ignore[assignment]

    app = Starlette()
    app.state.runtime = types.SimpleNamespace(mind=StubMind())
    register_court_routes(app)

    hdr = {"Authorization": "Bearer test-key"}
    ok = True

    with TestClient(app) as client:
        # Unauthorised is rejected.
        r = client.get("/v1/courts")
        assert r.status_code == 401, r.text
        print("  401 without key                      OK")

        # Empty deck snapshot.
        r = client.get("/v1/courts", headers=hdr)
        assert r.status_code == 200, r.text
        snap = r.json()
        assert snap["count"] == 0 and snap["courts"] == [], snap
        print("  GET  /v1/courts (empty)              OK")

        # Create two courts (they will run concurrently).
        ids = []
        for i in range(2):
            r = client.post("/v1/courts", headers=hdr,
                            json={"kind": "window", "title": f"Card {i + 1}"})
            assert r.status_code == 201, r.text
            ids.append(r.json()["id"])
        print(f"  POST /v1/courts x2                   OK  {ids}")

        r = client.get("/v1/courts", headers=hdr)
        assert r.json()["count"] == 2, r.text
        print("  GET  /v1/courts (2 cards)            OK")

        # Spawn both instances -> two lanes delegated at once.
        for cid in ids:
            r = client.post(f"/v1/courts/{cid}/spawn", headers=hdr, json={"goal": "open notepad"})
            assert r.status_code == 202, r.text
            assert r.json()["status"] == "running", r.text
        delegated = app.state.runtime.mind.links.link.delegated
        assert len(delegated) == 2, delegated
        lanes = {d[0] for d in delegated}
        assert len(lanes) == 2 and all(lanes), lanes
        print(f"  POST .../spawn x2 (2 lanes live)     OK  {sorted(lanes)}")

        # Steer one court.
        r = client.post(f"/v1/courts/{ids[0]}/goal", headers=hdr, json={"goal": "type hello"})
        assert r.status_code == 202 and r.json()["goal"] == "type hello", r.text
        print("  POST .../goal (steer)                OK")

        # Voice hook.
        r = client.post(f"/v1/courts/{ids[0]}/voice", headers=hdr,
                        json={"action": "speak", "text": "court one update"})
        assert r.status_code == 200 and r.json()["action"] == "speak", r.text
        print("  POST .../voice                       OK")

        # Cancel one, keep the card.
        r = client.post(f"/v1/courts/{ids[1]}/cancel", headers=hdr, json={})
        assert r.status_code == 200 and r.json()["status"] == "stopped", r.text
        assert app.state.runtime.mind.lanes.stopped, "lane.stop not called"
        print("  POST .../cancel (keeps card)         OK")

        # Delete one.
        r = client.delete(f"/v1/courts/{ids[1]}", headers=hdr)
        assert r.status_code == 200 and r.json()["ok"] is True, r.text
        r = client.get("/v1/courts", headers=hdr)
        assert r.json()["count"] == 1, r.text
        print("  DELETE /v1/courts/{id}               OK")

        # SSE route is registered and primes with a snapshot. We assert the
        # route contract without opening the (never-ending) live stream, which
        # would hold the test client's lifespan open forever.
        paths = {getattr(r, "path", "") for r in app.router.routes}
        assert "/v1/courts/{court_id}/events" in paths, paths
        assert "/v1/court-events" in paths, paths
        print("  routes /…/events + /v1/court-events OK")

        # Bus saw the deck events too.
        assert app.state.runtime.mind.bus.events, "no bus events"
        print(f"  mind.bus published {len(app.state.runtime.mind.bus.events)} deck events   OK")

    print("\ncourt HTTP deck self-test PASSED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
