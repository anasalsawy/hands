"""Live provider probe: chat TTFT + availability, one shot, no secrets printed.

Run:  python persona/voice/bench/provider_probe.py [chat|tts|all]

Prints JSON lines to stdout and a table at the end. Keys come from env only.
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

from persona.config import load_dotenv  # noqa: E402

load_dotenv()

GEM = "https://generativelanguage.googleapis.com/v1beta/openai"
GEM_NATIVE = "https://generativelanguage.googleapis.com/v1beta/models"
FEATHER = "https://api.featherless.ai/v1"
DI = "https://api.deepinfra.com/v1/openai"

PROMPT = "hey, how is your day going?"


def _key(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _redact(cand: dict) -> dict:
    """Never echo credentials into logs/files."""
    out = dict(cand)
    out.pop("key", None)
    return out


def chat_candidates() -> list[dict]:
    out = []
    if _key("COMPUSE_LLM_API_KEY"):
        for m in ("gemini-flash-lite-latest", "gemini-3.1-flash-lite",
                  "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash-lite"):
            out.append({"hop": "chat", "provider": "gemini", "model": m,
                        "base": GEM, "key": _key("COMPUSE_LLM_API_KEY")})
    if _key("FEATHERLESS_API_KEY") or _key("SAWII_FEATHERLESS_API_KEY"):
        for m in ("Qwen/Qwen3.8-Flash-Next",):
            out.append({"hop": "chat", "provider": "featherless", "model": m,
                        "base": FEATHER, "key": _key("SAWII_FEATHERLESS_API_KEY") or _key("FEATHERLESS_API_KEY")})
    if _key("DEEPINFRA_API_KEY"):
        for m in ("XiaomiMiMo/MiMo-V2.6-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"):
            out.append({"hop": "chat", "provider": "deepinfra", "model": m,
                        "base": DI, "key": _key("DEEPINFRA_API_KEY")})
    if _key("OPENAI_API_KEY"):
        for m in ("gpt-4o-mini",):
            out.append({"hop": "chat", "provider": "openai", "model": m,
                        "base": "https://api.openai.com/v1", "key": _key("OPENAI_API_KEY")})
    if _key("OPENROUTER_API_KEY"):
        for m in ("qwen/qwen-2.5-0.5b-instruct",):
            out.append({"hop": "chat", "provider": "openrouter", "model": m,
                        "base": "https://openrouter.ai/api/v1", "key": _key("OPENROUTER_API_KEY")})
    return out


async def stream_chat_ttft(client: httpx.AsyncClient, cand: dict) -> dict:
    """OpenAI-compatible streaming TTFT + total + tokens/sec."""
    body = {
        "model": cand["model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 40,
        "temperature": 0.7,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {cand['key']}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    ttft = None
    chars = 0
    try:
        async with client.stream("POST", f"{cand['base']}/chat/completions",
                                 headers=headers, json=body, timeout=45) as r:
            if r.status_code >= 400:
                raw = (await r.aread())[:200]
                return {**_redact(cand), "ok": False, "status": r.status_code,
                        "error": raw.decode("utf-8", "replace")[:160]}
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
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
                    chars += len(piece)
        total = (time.perf_counter() - t0) * 1000
        tps = (chars / 4.0) / (total / 1000) if ttft else 0
        return {**_redact(cand), "ok": True, "status": 200, "ttft_ms": round(ttft or total),
                "total_ms": round(total), "chars": chars, "tok_s": round(tps, 1)}
    except Exception as exc:  # noqa: BLE001
        return {**_redact(cand), "ok": False, "status": 0,
                "error": f"{type(exc).__name__}: {exc}"[:160],
                "total_ms": round((time.perf_counter() - t0) * 1000)}


async def main(n: int) -> None:
    cands = chat_candidates()
    rows = []
    async with httpx.AsyncClient(http2=False, limits=httpx.Limits(max_connections=16)) as client:
        # warm handshake once per provider, then N timed calls
        by_provider: dict[str, list[dict]] = {}
        for c in cands:
            by_provider.setdefault(c["provider"], []).append(c)
        for prov, group in by_provider.items():
            for c in group:
                warm = await stream_chat_ttft(client, c)
                print(json.dumps({"phase": "cold", **warm}), flush=True)
                samples = [warm] if not warm["ok"] else []
                if warm["ok"]:
                    for _ in range(n - 1):
                        s = await stream_chat_ttft(client, c)
                        samples.append(s)
                        print(json.dumps({"phase": "warm", **s}), flush=True)
                ok = [s for s in samples if s.get("ok")]
                rows.append({
                    "provider": prov, "model": c["model"], "ok": bool(ok),
                    "n": len(samples),
                    "cold_ms": samples[0].get("total_ms"),
                    "error": "" if ok else str(samples[0].get("error", ""))[:120],
                    "ttft_p50": round(statistics.median([s["ttft_ms"] for s in ok])) if ok else None,
                    "ttft_max": max([s["ttft_ms"] for s in ok]) if ok else None,
                    "total_p50": round(statistics.median([s["total_ms"] for s in ok])) if ok else None,
                    "tok_s": round(statistics.median([s["tok_s"] for s in ok]), 1) if ok else None,
                })
    print("\n=== CHAT TTFT MATRIX ===")
    print(f"{'provider':12s} {'model':34s} {'ok':3s} {'ttft p50':>9s} {'ttft max':>9s} {'tot p50':>8s} {'tok/s':>6s}  err")
    for r in sorted(rows, key=lambda x: (not x["ok"], x["ttft_p50"] or 9e9)):
        print(f"{r['provider']:12s} {r['model'][:34]:34s} {'Y' if r['ok'] else 'N':3s} "
              f"{str(r['ttft_p50']):>9s} {str(r['ttft_max']):>9s} {str(r['total_p50']):>8s} "
              f"{str(r['tok_s']):>6s}  {r['error'][:70]}")


if __name__ == "__main__":
    asyncio.run(main(int(os.environ.get("PROBE_N", "5"))))