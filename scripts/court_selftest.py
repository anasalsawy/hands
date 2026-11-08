"""Prove the court deck: two concurrent desktop instances, one deck.

Run:  python scripts/court_selftest.py

It uses a stand-in for the mind (bus + links + lanes) so it proves the deck
layering on its own, with no API keys, no DB and no network. Two courts are
spawned at once; each gets its own lane and its own "engine" coroutine, and
we assert they overlap in time and each finishes.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona.court import HandsDeck  # noqa: E402


class FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, pid, ev):
        self.events.append((pid, ev))


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.strategy = "spawn"


class FakeLinkedLane:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.status = "active"


class FakeLink:
    """A link that 'runs' a desktop instance: delegate returns at once and the
    work continues in a background task, exactly like the spawn strategy in
    persona/link.py (the process is launched, then monitored off the path)."""

    def __init__(self, name="hands-engine"):
        self.agent = FakeAgent(name)
        self.delegated = []
        self.stopped = []
        self.live = set()
        self.max_live = 0
        self._tasks: list[asyncio.Task] = []

    async def delegate(self, lane_id, instruction, **kw):
        self.delegated.append(lane_id)
        self.live.add(lane_id)
        self.max_live = max(self.max_live, len(self.live))

        async def work():
            await asyncio.sleep(0.3)
            self.live.discard(lane_id)

        self._tasks.append(asyncio.create_task(work()))
        return FakeLinkedLane(lane_id)

    async def steer(self, lane_id, note):
        return True

    async def status(self, lane_id):
        """The real Link answers this; deck.spawn verifies against it."""
        return "active" if lane_id in self.live else "failed"

    def lane(self, lane_id):
        return None

    async def stop(self, lane_id):
        self.stopped.append(lane_id)
        self.live.discard(lane_id)
        return True


class FakeLinks:
    def __init__(self):
        self._links = {"hands-engine": FakeLink()}

    def get(self, name):
        return self._links.get(name)


class FakeLanes:
    def __init__(self):
        self.stopped = []

    def stop(self, lane):
        self.stopped.append(lane)


class FakeMind:
    def __init__(self):
        self.bus = FakeBus()
        self.links = FakeLinks()
        self.lanes = FakeLanes()
        self.hands = None


async def main() -> int:
    mind = FakeMind()
    deck = HandsDeck(mind)
    pid = 1

    got: list[dict] = []
    q = deck.subscribe(pid)

    # Two cards, spawned at once — this is "the courts".
    a = deck.create(pid, title="Court A", kind="desktop")
    b = deck.create(pid, title="Court B", kind="window",
                    target={"kind": "window", "hwnd": 4242})

    # Spawn both concurrently.
    await asyncio.gather(
        deck.spawn(pid, a.id, goal="open the browser and search"),
        deck.spawn(pid, b.id, goal="read the open document"),
    )

    while not q.empty():
        got.append(q.get_nowait())

    # Two distinct lanes, both delegated, both live at the same moment.
    link = mind.links.get("hands-engine")
    assert len(deck.list(pid)) == 2, "expected two courts"
    assert a.lane and b.lane and a.lane != b.lane, "cards must hold separate lanes"
    assert set(link.delegated) == {a.lane, b.lane}, "both lanes must be delegated"
    assert link.max_live == 2, f"both desktop instances must run at once (saw {link.max_live})"

    kinds = [e["kind"] for e in got]
    assert "court.created" in kinds and "court.spawned" in kinds, kinds

    # Send a follow-up goal, then cancel both.
    await deck.send(pid, a.id, "now click the first result")
    assert a.goal == "now click the first result"
    await deck.cancel(pid, a.id)
    await deck.cancel(pid, b.id)
    assert a.status == "stopped" and b.status == "stopped"
    assert set(link.stopped) == {a.lane, b.lane}, "cancel must stop both workers"

    # Voice hook is a named handle, not a second voice stack.
    v = deck.voice(pid, a.id, action="speak", text="court A is done")
    assert v["action"] == "speak" and v["websocket"].startswith("/v1/voice")

    # The mind bus saw the deck too (SSE clients get them for free).
    assert any(ev.get("type") == "court" for _pid, ev in mind.bus.events)

    snap = deck.snapshot(pid)
    assert snap["count"] == 2 and snap["running"] == 0

    print("court deck self-test PASSED")
    print(f"  courts:           2  (lap count: {len(deck.list(pid))})")
    print(f"  lanes run at once: {sorted(link.delegated)}  (peak concurrent: {link.max_live})")
    print(f"  events emitted:   {len(got)}")
    print(f"  bus events:       {len(mind.bus.events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
