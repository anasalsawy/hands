"""Live: Hands cards on the real shared-mind server and real Chromium tabs.

    SAWII_SHARED_MIND_URL=https://... python scripts/shared_mind_live.py

Two browser cards run at once (one task, two hands, one tab each), one is steered
mid-task, and L is asked about the screens while the hands work.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona import shared_mind  # noqa: E402
from persona.court import HandsDeck  # noqa: E402
from persona.gateway.browser.routes import BrowserHub  # noqa: E402

GOALS = {
    "A": "Open https://en.wikipedia.org/wiki/Eiffel_Tower and find (1) the year construction finished and "
         "(2) the name of the company that designed it. Report both.",
    "B": "On Wikipedia, find the population of Canada.",
}
STEER = "Change of plan: find the population of Australia instead, not Canada."


class Mind:
    def __init__(self):
        self.browser_hub = BrowserHub()


async def main() -> int:
    mind = Mind()
    deck = HandsDeck(mind)
    client = deck._shared = shared_mind.SharedMindClient(os.environ["SAWII_SHARED_MIND_URL"])
    print("reset", await client.reset(), file=sys.stderr, flush=True)
    events: list = []
    q = deck.subscribe(1)

    async def show():
        while True:
            ev = await q.get()
            if ev.get("kind") in ("court.hand", "court.status", "court.goal"):
                print(round(time.time() - t0, 1), ev.get("kind"), ev.get("hand") or "", ev.get("tool") or "",
                      str(ev.get("text") or ev.get("result") or ev.get("error") or "")[-120:].replace("\n", " "),
                      file=sys.stderr, flush=True)
    t0 = time.time()
    asyncio.get_running_loop().create_task(show())
    cards = {}
    t0 = time.time()
    for name, goal in GOALS.items():
        c = deck.create(1, title=name, kind="browser")
        c.lane, c.status, c.goal = f"R{name}", "running", goal
        cards[name] = c
        deck._work(1, c.id, c.lane, goal)
    while not all("shared_tid" in c.meta for c in cards.values()):
        await asyncio.sleep(0.05)
    while not cards["B"].meta.get("last_tools"):
        await asyncio.sleep(0.1)
    await deck.send(1, cards["B"].id, STEER)
    l0 = time.time()
    seen = await client.chat("Don't start a task. What page is each hand on right now?")
    l_secs = round(time.time() - l0, 2)
    while any(c.status == "running" for c in cards.values()):
        await asyncio.sleep(0.1)
        if time.time() - t0 > 300:
            break
    out = dict(total_secs=round(time.time() - t0, 2), l_reply_secs=l_secs, l_reply=seen.get("reply") or seen.get("text") or seen,
               l_first_token=seen.get("ttft"),
               cards={n: dict(status=c.status, result=c.meta.get("result"), error=c.error,
                              actions=c.meta.get("actions"), act_to_act_median=c.meta.get("act_to_act_median"),
                              think_median=c.meta.get("think_median"), tool_median=c.meta.get("tool_median"),
                              done_secs=round(c.updated_at - t0, 2) if isinstance(c.updated_at, float) else None)
                      for n, c in cards.items()},
               tabs=len(mind.browser_hub._ones))
    print(json.dumps(out, indent=1))
    await client.close()
    for key in list(mind.browser_hub._ones):
        await mind.browser_hub._ones[key].session.close()
    ok = all(c.status == "done" for c in cards.values()) and "Australia" in str(cards["B"].meta.get("result"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
