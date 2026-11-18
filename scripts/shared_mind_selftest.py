"""Prove Hands as the shared mind's body, offline.

Run:  python scripts/shared_mind_selftest.py

A stand-in shared-mind server sends the events the real one sends. It checks:
- one browser card becomes one shared-mind task, with no lane division;
- the task's hands act on the card's own tab: page changes run in order, reads overlap;
- two cards run at the same time on separate tabs;
- a new goal on a running card steers its live task; Stop cancels it.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["SAWII_SHARED_MIND_URL"] = "http://stand-in"

from persona import shared_mind  # noqa: E402
from persona.court import HandsDeck  # noqa: E402


@dataclass
class Snap:
    url: str
    title: str = "Page"
    visible_text: str = "Gustave Eiffel's company designed it. Construction finished in 1889."
    elements: list = field(default_factory=lambda: [{"index": 0, "tag": "a", "text": "History"}])
    screenshot_png: bytes | None = None


class Session:
    def __init__(self):
        self.url = "about:blank"
        self.busy = 0
        self.max_busy_writes = 0

    async def snapshot(self, screenshot=True, ground=True):
        return Snap(self.url)


class One:
    def __init__(self):
        self.session = Session()


class Hub:
    def __init__(self):
        self.ones: dict = {}
        self.log: list = []

    async def _get(self, key):
        return self.ones.setdefault(key, One())

    async def run(self, pid, name, args, lane=None):
        s = (await self._get((pid, lane))).session
        s.busy += 1
        s.max_busy_writes = max(s.max_busy_writes, s.busy)
        t0 = time.time()
        await asyncio.sleep(0.05)
        if name == "browser_navigate":
            s.url = args["url"]
        s.busy -= 1
        self.log.append((lane, name, t0, time.time()))
        return f"{name} ok"


class FakeMind:
    """Stand-in for the server: each task's two hands act at once, then both deliver."""

    def __init__(self):
        self.tasks: dict[str, asyncio.Queue] = {}
        self.steers: list = []
        self.cancelled: list = []
        self.observed: dict[int, list[str]] = {}
        self.started: list = []
        self._n = 0
        self.seq = 0

    def _ev(self, tid, **kw):
        self.seq += 1
        self.tasks[tid].put_nowait(dict(seq=self.seq, tid=tid, t=time.time(), **kw))

    async def start(self, goal, hands=2):
        self._n += 1
        tid = f"T{self._n}"
        self.tasks[tid] = asyncio.Queue()
        self.started.append((tid, goal, hands))
        asyncio.get_running_loop().create_task(self._play(tid, 2 * self._n, 2 * self._n + 1))
        return tid, self.tasks[tid]

    async def _play(self, tid, a, b):
        self._ev(tid, type="step", slot=a, name="Alice", text="Alice: part 1. Bob: part 2.")
        self._ev(tid, type="act", slot=a, name="Alice", tool="goto", args={"url": "en.wikipedia.org/wiki/Eiffel_Tower"})
        self._ev(tid, type="act", slot=b, name="Bob", tool="goto", args={"url": "https://en.wikipedia.org/wiki/Eiffel_Tower"})
        while len(self.observed.get(a, [])) < 1 or len(self.observed.get(b, [])) < 1:
            await asyncio.sleep(0.01)
        self._ev(tid, type="act", slot=a, name="Alice", tool="find", args={"text": "1889"})
        self._ev(tid, type="act", slot=b, name="Bob", tool="find", args={"text": "company"})
        while len(self.observed[a]) < 2 or len(self.observed[b]) < 2:
            await asyncio.sleep(0.01)
        while tid not in [s[0] for s in self.steers] and tid == "T1":
            await asyncio.sleep(0.01)
        self._ev(tid, type="done", answer="Finished in 1889 by Gustave Eiffel's company.", cancelled=False)

    async def steer(self, tid, text):
        self.steers.append((tid, text))

    async def cancel(self, tid):
        self.cancelled.append(tid)

    async def observe(self, slot, text):
        self.observed.setdefault(slot, []).append(text)

    def forget(self, tid):
        pass


class Mind:
    def __init__(self):
        self.browser_hub = Hub()


async def main() -> int:
    mind = Mind()
    deck = HandsDeck(mind)
    fake = deck._shared = FakeMind()
    a = deck.create(1, title="A", kind="browser")
    b = deck.create(1, title="B", kind="browser")
    for c in (a, b):
        c.lane = f"R{c.title}"
        c.status = "running"
    t0 = time.time()
    deck._work(1, a.id, a.lane, "Eiffel Tower: year finished and designer company")
    deck._work(1, b.id, b.lane, "Same for card B")
    while "shared_tid" not in a.meta:
        await asyncio.sleep(0.01)
    await deck.send(1, a.id, "also give the height")
    while a.status == "running" or b.status == "running":
        await asyncio.sleep(0.02)
        assert time.time() - t0 < 10, "cards never finished"
    hub = mind.browser_hub
    assert a.status == b.status == "done", (a.status, a.error, b.status, b.error)
    assert "1889" in a.meta["result"], a.meta
    assert a.meta["agent_loop"] == "shared-mind"
    assert fake.steers == [("T1", "also give the height")], fake.steers
    assert len(fake.started) == 2 and all(h == 2 for _, _, h in fake.started)
    # one tab per card, page changes on it in order
    lanes = {lane for lane, *_ in hub.log}
    assert lanes == {a.lane, b.lane}, lanes
    for one in hub.ones.values():
        assert one.session.max_busy_writes == 1, "page changes on one tab overlapped"
        assert one.session.url == "https://en.wikipedia.org/wiki/Eiffel_Tower", one.session.url
    # the two cards browsed at the same time
    ra = [x for x in hub.log if x[0] == a.lane][0]
    rb = [x for x in hub.log if x[0] == b.lane][0]
    assert ra[2] < rb[3] and rb[2] < ra[3], "cards did not run at the same time"
    view = fake.observed[2][0]
    assert view.startswith("URL: https://en.wikipedia.org/wiki/Eiffel_Tower") and "[0] a History" in view, view
    assert "Matches for '1889'" in fake.observed[2][1] and "1889" in fake.observed[2][1], fake.observed[2][1]
    # Stop cancels the live task
    c = deck.create(1, title="C", kind="browser")
    c.lane, c.status = "RC", "running"
    fake._play = lambda *a: asyncio.sleep(60)
    deck._work(1, c.id, c.lane, "long task")
    while "shared_tid" not in c.meta:
        await asyncio.sleep(0.01)
    await deck.cancel(1, c.id)
    assert fake.cancelled == [c.meta["shared_tid"]], fake.cancelled
    assert c.status == "stopped"
    print("shared mind self-test PASSED")
    print(f"  cards: 2 concurrent + 1 stopped; steer -> {fake.steers[0][0]}; tool calls: {len(hub.log)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
