"""Does the fast responder's new conversation window cost any latency?

Runs the SAME utterances twice against the live model: once with no history
(the old two-message call) and once with a realistic 3-exchange window. The
only difference is how many tokens ride along, so the delta IS the cost of the
window. Also prints what the model answered, because a window that makes
"and you?" resolve to something is the whole point.

    python persona/voice/bench/fast_l_history_latency.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from persona.voice.fast_L import FastL, fast_l_config, needs_deep_path  # noqa: E402

# Each case is a short spoken line plus the exchange that came before it in the
# same call. Several are deliberately unresolvable without that window.
CASES = [
    ("how are you doing",
     [{"role": "user", "content": "hey"},
      {"role": "assistant", "content": "Hey, good to hear you."}]),
    ("and you?",
     [{"role": "user", "content": "how was your day"},
      {"role": "assistant", "content": "Busy but good, I got some work done."}]),
    ("tell me more about that",
     [{"role": "user", "content": "I finished the deck today"},
      {"role": "assistant", "content": "Nice, that deck was taking ages."}]),
    ("what do you think",
     [{"role": "user", "content": "I might ship it tomorrow"},
      {"role": "assistant", "content": "Sounds like a plan."}]),
    ("that's funny",
     [{"role": "user", "content": "my cat sat on the keyboard again"},
      {"role": "assistant", "content": "Classic cat behaviour."}]),
]

REPEATS = 3
# The window must not be allowed to eat the latency it was meant to protect.
BUDGET_MS = 150.0


async def one(fl: FastL, text: str, history):
    res = await fl.reply(text, history=history)
    ms = res.total_ms if res.total_ms is not None else -1
    if res.abstained:
        return ms, "ABSTAINED (deep path will answer)"
    return ms, (res.text or f"error: {res.error}")[:72]


async def main() -> int:
    base, model, key = fast_l_config()
    print(f"target : {base}")
    print(f"model  : {model}")
    print(f"window : SAWII_FAST_L_HISTORY="
          f"{os.environ.get('SAWII_FAST_L_HISTORY', '(default 6)')}\n")

    fl = FastL()
    if not await fl.warm():
        print(f"FAIL: no model reachable at {base} (key={key!r}).\n"
              f"      start it with: persona\\voice\\bench\\start_fast_l.ps1")
        await fl.aclose()
        return 1

    bare, hist_ms = [], []
    for text, hist in CASES:
        gated = needs_deep_path(text)
        print(f"  {text!r}" + ("   [gated to deep path]" if gated else ""))
        # Alternate the order every repeat so thermal throttling or background
        # load on this box cannot quietly favour one arm of the comparison.
        for rep in range(REPEATS):
            order = [(None, "bare"), (hist, "hist")] if rep % 2 == 0 \
                else [(hist, "hist"), (None, "bare")]
            got = {}
            for h, tag in order:
                got[tag] = await one(fl, text, h)
            bare.append(got["bare"][0])
            hist_ms.append(got["hist"][0])
            if rep == 0:
                print(f"      no window -> {got['bare'][1]}")
                print(f"      3 turns   -> {got['hist'][1]}")

    await fl.aclose()
    lb = [m for m in bare if m >= 0]
    lh = [m for m in hist_ms if m >= 0]
    if not lb or not lh:
        print("\nFAIL: nothing measurable came back")
        return 1

    pb, ph = statistics.median(lb), statistics.median(lh)
    print(f"\n{'=' * 60}")
    print(f"fast-L reply, median of {len(lb)} samples per arm:")
    print(f"  without window : {pb:7.0f} ms")
    print(f"  with 3 turns   : {ph:7.0f} ms")
    print(f"  delta          : {ph - pb:+7.0f} ms   (budget {BUDGET_MS:.0f} ms)")
    print(f"{'=' * 60}")
    if ph - pb > BUDGET_MS:
        print(f"\nFAIL: the window costs {ph - pb:.0f} ms. Shrink "
              f"SAWII_FAST_L_HISTORY before shipping it.")
        return 1
    print("\nPASS: window fits the latency budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))