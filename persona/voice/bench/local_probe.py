"""Local-runner benchmark: real TTFT + tokens/sec on THIS CPU box.

Hits an OpenAI-compatible /v1/chat/completions (llama.cpp llama-server or
Ollama), streaming, and reports cold (first call after load) vs warm.

Env: LOCAL_BASE (default http://127.0.0.1:8080/v1), LOCAL_MODEL, LOCAL_N
Run: python persona/voice/bench/local_probe.py
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import httpx  # noqa: E402

BRIDGE_PROMPT = ("The person just said:\n{utterance}\n\n"
                 "Speak your short spoken bridge now.")
BRIDGE_SYSTEM = (
    "You are a fast voice micro-responder. The deep answer is being prepared in "
    "parallel. Reply with ONE short conversational bridge only (2-8 words). "
    "Output ONLY the spoken bridge phrase, nothing else.")

UTTERANCES = [
    "hey, how is your day going?",
    "what time is it for you right now?",
    "remind me what we talked about last time",
    "tell me something you noticed about me",
    "give me a quick plan for tomorrow morning",
    "did you hear that noise outside a minute ago",
]


async def one(client: httpx.AsyncClient, base: str, model: str, utterance: str,
              *, max_tokens: int, temperature: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": BRIDGE_SYSTEM},
                     {"role": "user", "content": BRIDGE_PROMPT.format(utterance=utterance)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # llama.cpp: keep-alive so the model stays resident between calls
        "keepalive": "30m",
    }
    t0 = time.perf_counter()
    ttft = None
    out = ""
    tok_evt = 0
    try:
        async with client.stream("POST", f"{base}/chat/completions", json=body,
                                 timeout=httpx.Timeout(90, connect=5)) as r:
            if r.status_code >= 400:
                raw = (await r.aread())[:200]
                return {"ok": False, "status": r.status_code,
                        "error": raw.decode("utf-8", "replace")[:150]}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                d = (obj.get("choices") or [{}])[0].get("delta") or {}
                piece = d.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    out += piece
                    tok_evt += 1
        total = (time.perf_counter() - t0) * 1000
        words = len(out.split())
        gen_ms = max(1.0, total - (ttft or total))
        return {"ok": True, "status": 200, "ttft_ms": round(ttft or total),
                "total_ms": round(total), "text": out.strip()[:90], "words": words,
                "tok_s": round(words / (gen_ms / 1000), 1)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": 0,
                "error": f"{type(exc).__name__}:{exc}"[:150],
                "total_ms": round((time.perf_counter() - t0) * 1000)}


def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {}
    v_sorted = sorted(v)
    p95 = v_sorted[min(len(v_sorted) - 1, int(round(0.95 * (len(v_sorted) - 1))))]
    return {"n": len(v), "p50": round(statistics.median(v)), "p95": round(p95),
            "min": round(min(v)), "max": round(max(v))}


async def main() -> None:
    base = os.environ.get("LOCAL_BASE", "http://127.0.0.1:8080/v1").rstrip("/")
    model = os.environ.get("LOCAL_MODEL", "qwen2.5-1.5b-instruct-q4_k_m")
    n = int(os.environ.get("LOCAL_N", "12"))
    max_tokens = int(os.environ.get("LOCAL_MAXTOK", "24"))
    label = os.environ.get("LOCAL_LABEL", "llama.cpp qwen2.5:1.5b-instruct-q4_K_M")
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=4)) as client:
        cold = await one(client, base, model, UTTERANCES[0],
                         max_tokens=max_tokens, temperature=0.0)
        print("COLD:", json.dumps(cold), flush=True)
        samples = []
        for i in range(n):
            s = await one(client, base, model, UTTERANCES[i % len(UTTERANCES)],
                          max_tokens=max_tokens, temperature=0.7)
            samples.append(s)
            print(f"{i+1:2d}", json.dumps(s), flush=True)
    ok = [s for s in samples if s.get("ok")]
    print("\n=== LOCAL:", label, "===")
    if not ok:
        print("ALL FAILED:", samples[0].get("error"))
        return
    print("cold_total_ms:", cold.get("total_ms"))
    print("ttft :", stats([s["ttft_ms"] for s in ok]))
    print("total:", stats([s["total_ms"] for s in ok]))
    print("tok_s:", stats([s["tok_s"] for s in ok]))
    print("sample texts:", [s["text"] for s in ok[:6]])


if __name__ == "__main__":
    asyncio.run(main())