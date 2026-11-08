"""The fast L against a stand-in deep path, so its latency is not swamped.

Why this exists: the full VoiceSession turn costs 8-36 s through the remote mind
(Featherless), and this box currently runs several agents at once, so an A/B on
the live pipeline reports the remote provider's variance, not the fast path.
This isolates the two racers: a real local llama.cpp call versus a fixed
constant that stands in for the deep turn, using the production race code.

Run: python persona/voice/bench/fast_l_race.py [--deep-ms 6500] [--turns 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from persona.voice.fast_L import FastL, needs_deep_path  # noqa: E402

UTTERANCES = [
    "hey", "hi there", "how are you", "how is your day going", "good morning",
    "what's up", "thanks", "that's nice", "okay cool", "bye for now",
    "how are you feeling right now", "good evening", "I'm back", "still there",
]


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return {}
    s = sorted(v)
    return {"n": len(v), "p50": round(statistics.median(v)),
            "p95": s[int(round(0.95 * (len(s) - 1)))], "min": round(min(v)),
            "max": round(max(v))}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-ms", type=float, default=6500.0,
                    help="measured p50 first-word latency of the remote deep path")
    ap.add_argument("--turns", type=int, default=14)
    args = ap.parse_args()

    fast = FastL()
    await fast.warm()
    rows = []
    for i in range(args.turns):
        u = UTTERANCES[i % len(UTTERANCES)]
        if needs_deep_path(u):
            continue
        t0 = asyncio.get_event_loop().time()
        deep = asyncio.create_task(asyncio.sleep(args.deep_ms / 1000))

        async def deep_words() -> float:
            await deep
            return (asyncio.get_event_loop().time() - t0) * 1000

        dw = asyncio.create_task(deep_words())
        # The production rule: the fast line is spoken only if it beats the deep
        # path's first clause. Here that is a plain race on wall-clock arrival.
        fr = asyncio.create_task(fast.reply(u))
        done, _ = await asyncio.wait({fr, dw}, return_when=asyncio.FIRST_COMPLETED)
        winner = "fast" if fr in done else "deep"
        res = fr.result() if fr.done() else None
        row = {"in": u, "winner": winner,
               "fast_ttft_ms": None, "fast_line_ms": None,
               "deep_first_word_ms": round(args.deep_ms)}
        if res is not None:
            row["fast_ttft_ms"] = res.first_token_ms
            row["fast_line_ms"] = res.total_ms
            if winner == "deep":
                fr.cancel()
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not dw.done():
            dw.cancel()
        if not deep.done():
            deep.cancel()
        await asyncio.sleep(0.05)

    winners = [r for r in rows if r["winner"] == "fast"]
    print("\n=== FAST L RACE (local llama.cpp vs a %.0f ms stand-in deep path) ===" % args.deep_ms)
    print(json.dumps({
        "turns": len(rows), "fast_wins": len(winners),
        "win_share": round(len(winners) / max(1, len(rows)), 3),
        "fast_ttft_ms": stats([r["fast_ttft_ms"] for r in rows]),
        "fast_line_ms": stats([r["fast_line_ms"] for r in rows]),
        "deep_first_word_ms": round(args.deep_ms),
    }, indent=2))
    await fast.aclose()


if __name__ == "__main__":
    asyncio.run(main())