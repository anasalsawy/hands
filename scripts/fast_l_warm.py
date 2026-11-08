"""Measure WARM L1 (local fast mouth) latency against the running llama.cpp server.

L1 is the fast mouth of lane L: same mind as L2 (Featherless deep), just a tiny
local model that speaks first. This runs real spoken turns and reports TTFT and
first-clause latency after the model is resident.
"""
import asyncio
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persona.voice.fast_L import FastL


async def main() -> None:
    fast = FastL()
    print("warming...", flush=True)
    await fast.warm()
    turns = [
        "hey, how are you?",
        "good morning",
        "thanks so much",
        "what's up?",
        "how's it going today?",
        "nice to meet you",
        "ok cool",
        "good night",
    ]
    ttfts, totals = [], []
    for t in turns:
        r = await fast.reply(t)
        if r.ok and not r.abstained:
            ttfts.append(r.first_token_ms or 0)
            totals.append(r.total_ms or 0)
        print(json.dumps({"in": t, "out": r.text, "abstain": r.abstained,
                          "ttft_ms": r.first_token_ms, "total_ms": r.total_ms}), flush=True)
    if ttfts:
        print(json.dumps({
            "n": len(ttfts),
            "ttft_p50_ms": round(statistics.median(ttfts)),
            "ttft_min_ms": min(ttfts),
            "total_p50_ms": round(statistics.median(totals)),
        }), flush=True)
    await fast.aclose()


if __name__ == "__main__":
    asyncio.run(main())
