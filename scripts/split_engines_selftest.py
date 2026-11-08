"""Regression test: when the mind splits a goal, every part gets a real engine.

The bug this exists for: ``HandsDeck.divide`` opened an R-lane and bound a card
to it with ``attach_lane``, which is bookkeeping only. It never started an
engine, so every automatically-split card read
``failed: lane <id> has no engine connected to /v1/hands``. The deck looked
concurrent and drove nothing.

Runs on stand-ins for the mind (the class docstring invites exactly this), so it
needs no model, no browser and no engine process: fast enough to run on a
machine that is out of memory.

    python scripts/split_engines_selftest.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona.court import HandsDeck  # noqa: E402

PID = 7


class FakeLine:
    def __init__(self, lane_id: str, label: str):
        self.id, self.label = lane_id, label
        self.status = "open"


class FakeLanes:
    """Only ``start`` is exercised: it hands out R-labels and lane ids."""

    def __init__(self):
        self.n = 0
        self.started: list[FakeLine] = []

    async def start(self, pid: int, topic: str, instruction: str) -> FakeLine:
        self.n += 1
        line = FakeLine(f"ln{self.n:08x}", f"R{self.n}")
        self.started.append(line)
        return line


class FakeLinked:
    def __init__(self, lane_id: str):
        self.lane_id = lane_id
        self.status = "active"
        self.error = ""
        self.on_exit = None


class FakeAgent:
    name = "hands-engine"
    place = "computer"
    strategy = "spawn"


class FakeLink:
    """Records every delegate call so the test can prove an engine was started."""

    agent = FakeAgent()

    def __init__(self):
        self.delegate_calls: list[tuple[str, str]] = []
        self._lanes: dict[str, FakeLinked] = {}

    async def delegate(self, lane_id, instruction, **kwargs):
        # The whole point: a split part must reach the engine launcher, with
        # the lane id the mind already chose (not a freshly minted one).
        assert kwargs.get("persona_id") == PID, "engine must run as this persona"
        self.delegate_calls.append((lane_id, instruction))
        linked = FakeLinked(lane_id)
        self._lanes[lane_id] = linked
        return linked

    async def status(self, lane_id):
        return "active" if lane_id in self._lanes else "failed"

    def lane(self, lane_id):
        return self._lanes.get(lane_id)

    async def steer(self, lane_id, goal):
        return None


class FakeHub:
    """A runner appears for every lane ``delegate`` started, as it would live."""

    def __init__(self, link: FakeLink):
        self._link = link

    @staticmethod
    def _key(pid, place, lane):
        return (pid, place, lane)

    @property
    def runners(self):
        return {self._key(PID, "computer", lid): object()
                for lid in self._link._lanes}


class FakeLinks:
    def __init__(self, link: FakeLink):
        self._link = link

    def get(self, name):
        return self._link if name in ("hands-engine", "") else None

    def persona_key(self, pid):
        return "sp_test"


class FakeMind:
    pass


async def main() -> int:
    link = FakeLink()
    mind = FakeMind()
    mind.lanes = FakeLanes()
    mind.links = FakeLinks(link)
    mind.hands = FakeHub(link)
    mind.bus = None
    # No planner lobe: divide() falls back to the goal's own punctuation, which
    # keeps this deterministic and free.
    mind.a = None

    deck = HandsDeck(mind)
    parent = deck.create(PID, title="research")
    goal = ("check the vendor inbox; "
            "read the open pull requests; "
            "summarise yesterday's deploy log")
    parent.goal = goal

    opened = await deck.divide(PID, parent)

    fails: list[str] = []
    if len(opened) != 2:
        fails.append(f"expected 2 extra lanes, got {len(opened)}: {opened}")

    delegated = [lane_id for lane_id, _ in link.delegate_calls]
    if sorted(delegated) != sorted(opened):
        fails.append(f"engines were started on {delegated}, "
                     f"but the mind opened lanes {opened}")

    cards = [c for c in deck._courts.values() if c.id != parent.id]
    if len(cards) != len(opened):
        fails.append(f"expected {len(opened)} split cards, got {len(cards)}")

    for card in cards:
        # The card must name the lane the mind opened, not a fresh ct... id.
        if card.lane not in opened:
            fails.append(f"card {card.id} is on lane {card.lane!r}, "
                         f"not one of the mind's lanes {opened}")
        if card.status == "failed":
            fails.append(f"card {card.id} ({card.title!r}) FAILED: {card.error}")
        if not card.lane:
            fails.append(f"card {card.id} has no lane bound at all")
        if not card.meta.get("lane_label"):
            fails.append(f"card {card.id} has no R-label for the deck to show")

    labels = sorted(str(c.meta.get("lane_label")) for c in cards)
    # The mind numbers the lines it opens; one per split part, no gaps.
    if labels != [f"R{i}" for i in range(1, len(opened) + 1)]:
        fails.append(f"split cards should carry one R-label each, got {labels}")

    print(f"goal        : {goal}")
    print(f"lanes open  : {opened}")
    print(f"engines ran : {delegated}")
    for card in deck._courts.values():
        print(f"  {card.view()['id'][:18]:20} lane={card.lane or '-':12} "
              f"label={card.meta.get('lane_label') or '-':4} "
              f"status={card.status}")

    # Let any scheduled R-loop task run, then re-check: the lane/engine wiring
    # must still hold once the drive loop has had a turn.
    for _ in range(6):
        await asyncio.sleep(0)

    for card in cards:
        if card.lane not in opened:
            fails.append(f"[after drive] card {card.id} lost its lane")
        if card.status == "failed" and "no engine connected" in (card.error or ""):
            fails.append(f"[after drive] card {card.id} says: {card.error}")

    # Second guard: a one-action goal must NOT fan out. A fake split is worse
    # than serial work.
    solo = deck.create(PID, title="solo")
    solo.goal = "open the settings panel"
    before = len(link.delegate_calls)
    solo_cards = len(deck._courts)
    solo_opened = await deck.divide(PID, solo)
    if solo_opened:
        fails.append(f"single-action goal was split: {solo_opened}")
    if len(link.delegate_calls) != before:
        fails.append("a single-action goal started extra engines")
    if len(deck._courts) != solo_cards:
        fails.append("a single-action goal created extra cards")

    if fails:
        print("\nFAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nPASS: every split part reached the engine launcher on its own lane")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))