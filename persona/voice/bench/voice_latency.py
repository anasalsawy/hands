"""End-to-end first-audio harness on the REAL VoiceSession critical path.

Production-shaped: real Mind (kernel, lobes, memory), real SpeechIO/TTS, real
SentenceChunker, real _dispatch_speech. The only thing simulated is the socket,
which is a timestamped collector.

Clocks (all ms from the moment the user's words arrive):
  t_words      0          <- on_text()/on_partial() entry (== end of their speech)
  first_delta  A's first visible token
  first_chunk  the first clause the chunker released
  tts_rtt      round trip of that clause's synthesis call
  first_audio  the first audio/say frame handed to the socket   <-- THE METRIC

Run:  python persona/voice/bench/voice_latency.py --turns 8 [--text-only]
Env:  see README section "Voice latency bench".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from persona.config import load_settings  # noqa: E402
from persona.mind import Mind  # noqa: E402
from persona.voice.realtime import SpeechIO, VoiceSession  # noqa: E402

# Turns the fast gate will actually answer (no memory/fact/tool cue). Used with
# --prompts social to measure the fast path's own end-to-end number.
SOCIAL_PROMPTS = [
    "hey", "hi there", "how are you", "how is your day going", "good morning",
    "what's up", "thanks a lot", "that's nice", "okay cool", "bye for now",
    "how are you feeling right now", "good evening", "I'm back", "still there",
]

PROMPTS = [
    "hey, how is your day going?",
    "what should I do this afternoon?",
    "tell me something you noticed about me",
    "give me a quick plan for tomorrow morning",
    "what do you think about the weather today?",
    "remind me what we talked about last time",
    "how are you feeling right now?",
    "any ideas for a quick dinner?",
]


def _stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {}
    s = sorted(v)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    return {"n": len(v), "p50": round(statistics.median(v)), "p95": round(p95),
            "min": round(min(v)), "max": round(max(v))}


class Recorder:
    """Stand-in for the websocket: keeps an ordered, timestamped frame log."""

    def __init__(self):
        self.frames: list[tuple[float, dict]] = []
        self.t0 = time.monotonic()
        self.blobs = 0

    async def send_json(self, obj) -> None:
        self.frames.append(((time.monotonic() - self.t0) * 1000, dict(obj or {})))

    async def send_bytes(self, blob) -> None:
        self.blobs += 1
        self.frames.append(((time.monotonic() - self.t0) * 1000,
                            {"type": "_bytes", "n": len(blob or b"")}))

    def reset(self) -> None:
        self.frames.clear()
        self.blobs = 0
        self.t0 = time.monotonic()


async def one_turn(session: VoiceSession, rec: Recorder, text: str) -> dict:
    rec.reset()
    session._heard = ""
    session.interrupted = False
    task = asyncio.create_task(session._turn(text))
    session.turn_task = task
    first_audio = first_chunk = first_delta = None
    first_server = None
    err = ""
    deadline = time.monotonic() + 180
    while not task.done() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        for t, fr in rec.frames:
            k = fr.get("type")
            if first_delta is None and k == "delta" and str(fr.get("text", "")).strip():
                first_delta = t
            if k == "audio" and first_server is None and fr.get("lobe") != "L1":
                # The deep (DualLoop) path's own first clip, so L1's (the fast
                # mouth's) line is never mistaken for the mind answering.
                first_server = t
            if first_audio is None and k in ("audio", "say"):
                first_audio = t
                first_chunk = str(fr.get("text", ""))
    if task.done() and task.exception() is not None:
        err = type(task.exception()).__name__
    if not task.done():
        task.cancel()
        err = err or "timeout"
    for t, fr in rec.frames:
        if fr.get("type") == "error":
            err = err or str(fr.get("detail", ""))[:80]
    metrics = dict(session.last_metrics or {})
    audio_frames = [t for t, fr in rec.frames if fr.get("type") in ("audio", "say")]
    # What the fast responder actually did this turn, straight off the session:
    # "ok"/"abstained"/error, its own latency, and whether its line reached us.
    fast_frames = [t for t, fr in rec.frames
                   if fr.get("type") == "kernel"
                   and (fr.get("event") or {}).get("type") == "fast_l"]
    return {
        "text": text[:40],
        "words": len(text.split()),
        "t_first_delta_ms": round(first_delta) if first_delta is not None else None,
        "t_first_audio_ms": round(first_audio) if first_audio is not None else None,
        "first_chunk": (first_chunk or "")[:60],
        "first_chunk_chars": len(first_chunk or ""),
        "t_second_audio_ms": round(audio_frames[1]) if len(audio_frames) > 1 else None,
        "n_audio_frames": len(audio_frames),
        "server_first_audio_ms": metrics.get("first_audio_ms"),
        "first_server_clip_ms": round(first_server) if first_server is not None else None,
        "server_first_delta_ms": metrics.get("first_delta_ms"),
        "tts_first_rtt_ms": metrics.get("tts_first_rtt_ms"),
        "ttft_ms": metrics.get("ttft_ms"),
        "reply_ms": metrics.get("reply_ms"),
        "ttfa_filler_ms": metrics.get("ttfa_filler"),
        "lobe_a_model_ms": metrics.get("lobe_a_model_ms"),
        "lobe_a_host": metrics.get("lobe_a_host", ""),
        "fast_l_note": session._fast_l_note,
        "fast_l_ms": session._fast_l_ms,
        "fast_l_spoke": bool(session._fast_l_spoke),
        "fast_l_frame_ms": round(fast_frames[0]) if fast_frames else None,
        "tts_streams": metrics.get("tts_streams"),
        "error": err,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--partials", action="store_true",
                    help="feed on_partial first, then on_text (client ASR already had the words)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--prompts", default="mixed", choices=["mixed", "social"])
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    prompts = SOCIAL_PROMPTS if args.prompts == "social" else PROMPTS

    if args.text_only:
        os.environ["SAWII_VOICE_OUTPUT"] = "text"
    s = load_settings()
    if args.text_only:
        s.voice_output = "text"
    mind = Mind(s)
    store = mind.store
    from persona import identity
    rows = await store.all("SELECT * FROM personas ORDER BY id LIMIT 1")
    persona = (rows or [None])[0]
    if persona is None:
        pid, _key = await identity.create_persona(store, name="Latency Bench")
        persona = await identity.get_persona(store, pid)
    speech = SpeechIO(s.asr, s.tts, s.tts_voice)
    rec = Recorder()
    session = VoiceSession(mind, persona, speech, rec.send_json, rec.send_bytes,
                           dual=True, playback_ack=False)
    print(json.dumps({"event": "config", "label": args.label, "gate": session.gate,
                      "dual_lobe_voice": session.dual_lobe_voice,
                      "realtime_voice": session.realtime_voice,
                      "text_only": session.text_only,
                      "has_tts": speech.has_tts, "has_asr": speech.has_asr,
                      "a": f"{s.a.base_url} {s.a.model}",
                      "tts": f"{s.tts.base_url} {s.tts.model}",
                      "lobe_a": (f"{session._lobe_a.base} {session._lobe_a.model}"
                                 if getattr(session, "_lobe_a", None) else ""),
                      "env_FAST_L": os.environ.get("SAWII_FAST_L", "<unset>"),
                      "fast_l": (f"{session.fast_l.base} {session.fast_l.model}"
                                 if getattr(session, "fast_l", None) else "DISABLED"),
                      }))
    # let the session's own warmups (TTS + lobe A load) land before timing
    async def _warmed() -> bool:
        lobe = getattr(session, "_lobe_a", None)
        lobe_ready = lobe is None or lobe.last is not None
        # A cold local model costs ~14 s to first token and would otherwise be
        # paid on the very turn we are trying to measure.
        fast = getattr(session, "fast_l", None)
        fast_ready = fast is None or fast.last is not None
        return lobe_ready and fast_ready

    for _ in range(240):
        if await _warmed():
            break
        await asyncio.sleep(0.25)
    await asyncio.sleep(1.0)
    rows = []
    for i in range(args.turns):
        text = PROMPTS[i % len(PROMPTS)]
        if args.partials:
            await session.on_partial(text[: max(16, len(text) // 2)])
            await asyncio.sleep(0.05)
        r = await one_turn(session, rec, text)
        rows.append(r)
        print(json.dumps({"event": "turn", "i": i, **r}), flush=True)
    out = {
        "label": args.label,
        "turns": len(rows),
        "errors": [r["error"] for r in rows if r["error"]],
        "first_audio_ms": _stats([r["t_first_audio_ms"] for r in rows]),
        "first_audio_ms_warm": _stats([r["t_first_audio_ms"] for r in rows[1:]]),
        "first_delta_ms": _stats([r["t_first_delta_ms"] for r in rows]),
        "first_server_clip_ms": _stats([r["first_server_clip_ms"] for r in rows]),
        "tts_first_rtt_ms": _stats([r["tts_first_rtt_ms"] for r in rows]),
        "ttft_ms": _stats([r["ttft_ms"] for r in rows]),
        "reply_ms": _stats([r["reply_ms"] for r in rows]),
        "second_audio_ms": _stats([r["t_second_audio_ms"] for r in rows]),
        "first_chunk_chars": _stats([r["first_chunk_chars"] for r in rows]),
        "cold": rows[0],
        "rows": rows,
    }
    print("\n=== VOICE LATENCY:", args.label, "===")
    for k in ("first_audio_ms", "first_audio_ms_warm", "first_delta_ms",
              "first_server_clip_ms", "tts_first_rtt_ms",
              "ttft_ms", "reply_ms", "second_audio_ms", "first_chunk_chars"):
        print(f"{k:22s} {out[k]}")
    print("errors:", out["errors"][:6])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    await session.finish()
    try:
        await mind.drain(timeout=10)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    asyncio.run(main())