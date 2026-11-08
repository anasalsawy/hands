"""Prices the fast L hop by itself, over many turns, cold vs warm.

Two numbers that matter:
  speakable_share  how much of real traffic the safety gate lets the fast L answer
  first_audio_ms   gate + local TTFT + streaming TTS head, i.e. what the person waits

Run: python persona/voice/bench/fast_l_bench.py [--turns 20] [--tts]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from persona.config import load_settings  # noqa: E402
from persona.voice.fast_L import FastL, needs_deep_path  # noqa: E402

# A slice of ordinary spoken traffic, not just the easy stuff.
UTTERANCES = [
    "hey", "hi there", "how are you", "how is your day going", "good morning",
    "what's up", "thanks", "that's nice", "okay cool", "bye for now",
    "how are you feeling right now", "good evening", "yeah exactly", "I'm back",
    "still there", "what should I do this afternoon", "any ideas for dinner",
    "give me a quick plan for tomorrow", "what do you think about the weather",
    "remind me what we talked about last time", "tell me something you noticed about me",
    "open chrome and search for flights", "what time is it", "did I tell you my birthday",
    "translate this to french", "how much is a flight to Lisbon", "play something calm",
]


def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {}
    s = sorted(v)
    return {"n": len(v), "p50": round(statistics.median(v)),
            "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
            "min": round(min(v)), "max": round(max(v))}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=21)
    ap.add_argument("--tts", action="store_true", help="also pay the streaming TTS head")
    args = ap.parse_args()

    allowed = [u for u in UTTERANCES if not needs_deep_path(u)]
    blocked = [u for u in UTTERANCES if needs_deep_path(u)]
    print(json.dumps({"event": "gate", "total": len(UTTERANCES),
                      "fast_allowed": len(allowed), "deep_path": len(blocked),
                      "share_allowed": round(len(allowed) / len(UTTERANCES), 3),
                      "allowed": allowed, "blocked": blocked[:8]}))

    s = load_settings()
    fast = FastL()
    # Cold first: the first live turn must never be the one that loads the model.
    cold = await fast.reply(allowed[0])
    print(json.dumps({"event": "cold", "ms": cold.total_ms, "first_token_ms": cold.first_token_ms,
                      "ok": cold.ok, "error": cold.error}))

    tts_head = None
    if args.tts and s.tts.api_key:
        from persona.voice.stream_tts import GeminiStreamTTS
        gtts = GeminiStreamTTS(s.tts.base_url, s.tts.api_key, model=s.tts.model,
                               voice=s.tts_voice or "Puck")
        heads = []
        for text in ("Good, thanks for asking.", "Hey there, what's up?",
                     "Morning. How are you?"):
            t0 = asyncio.get_event_loop().time()
            try:
                n = 0
                async for sl in gtts.stream(text):
                    if n == 0:
                        heads.append(round((asyncio.get_event_loop().time() - t0) * 1000))
                    n += 1
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "tts_skip", "error": f"{type(exc).__name__}"}))
                break
        tts_head = stats(heads)
        print(json.dumps({"event": "tts_head", **tts_head}))
        await gtts.aclose()

    rows = []
    for i in range(args.turns):
        u = allowed[i % len(allowed)]
        r = await fast.reply(u)
        rows.append(r)
        print(json.dumps({"event": "turn", "in": u, "first_token_ms": r.first_token_ms,
                          "total_ms": r.total_ms, "ok": r.ok, "text": r.text[:60]}), flush=True)
    ok = [r for r in rows if r.ok and r.text]
    print("\n=== FAST L ALONE (warm, local llama.cpp) ===")
    print("first_token_ms:", stats([r.first_token_ms for r in ok]))
    print("full_line_ms :", stats([r.total_ms for r in ok]))
    if tts_head:
        combo = [ (r.first_token_ms or 0) + (tts_head["p50"] or 0) for r in ok ]
        print("est_first_audio_ms (ttft + tts head p50):", stats(combo))
    print("ok:", len(ok), "/", len(rows))
    await fast.aclose()


if __name__ == "__main__":
    asyncio.run(main())