"""TTS first-audio-byte bake-off. Every candidate measured, nothing assumed.

first_byte_ms = wall time until the FIRST byte of audio body arrives
                (streaming candidates), which is what gates audible speech.
full_ms       = wall time until the whole clip is in hand.

Run: python persona/voice/bench/tts_probe.py
Keys come from env only. Nothing secret is printed.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import httpx  # noqa: E402

from persona.config import load_dotenv  # noqa: E402

load_dotenv()

GEM_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
DI = "https://api.deepinfra.com/v1/openai"
FEATHER = "https://api.featherless.ai/v1"

TEXT = "Hey, that's a good question, let me think for a second."
N = int(os.environ.get("TTS_N", "4"))


async def first_byte(client: httpx.AsyncClient, url: str, *, headers: dict, body: dict,
                     label: str, provider: str, model: str) -> dict:
    t0 = time.perf_counter()
    got = 0
    first = None
    try:
        async with client.stream("POST", url, headers=headers, json=body, timeout=60) as r:
            if r.status_code >= 400:
                raw = (await r.aread())[:180]
                return {"provider": provider, "model": model, "label": label, "ok": False,
                        "status": r.status_code, "error": raw.decode("utf-8", "replace")[:150]}
            async for chunk in r.aiter_bytes():
                if chunk:
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    got += len(chunk)
        full = (time.perf_counter() - t0) * 1000
        if not got:
            return {"provider": provider, "model": model, "label": label, "ok": False,
                    "status": r.status_code, "error": "empty body"}
        return {"provider": provider, "model": model, "label": label, "ok": True,
                "status": r.status_code, "first_byte_ms": round(first or full),
                "full_ms": round(full), "bytes": got}
    except Exception as exc:  # noqa: BLE001
        return {"provider": provider, "model": model, "label": label, "ok": False, "status": 0,
                "error": f"{type(exc).__name__}:{exc}"[:150],
                "full_ms": round((time.perf_counter() - t0) * 1000)}


def gemini_tts_bodies(model: str, voice: str) -> dict:
    return {
        "model": model,
        "contents": [{"parts": [{"text": TEXT}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }


async def gemini_tts(client, model: str, voice: str, key: str) -> dict:
    url = f"{GEM_BASE}/{model}:generateContent"
    body = gemini_tts_bodies(model, voice)
    t0 = time.perf_counter()
    try:
        r = await client.post(url, headers={"x-goog-api-key": key}, json=body, timeout=60)
        if r.status_code >= 400:
            return {"provider": "gemini-native", "model": model, "label": voice, "ok": False,
                    "status": r.status_code, "error": r.text[:150]}
        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        audio = next((p.get("inlineData") or p.get("inline_data") for p in parts
                      if (p.get("inlineData") or p.get("inline_data"))), None)
        full = (time.perf_counter() - t0) * 1000
        if not audio or not audio.get("data"):
            return {"provider": "gemini-native", "model": model, "label": voice, "ok": False,
                    "status": r.status_code, "error": "no audio part: "
                    + json.dumps(data)[:120]}
        raw = base64.b64decode(audio["data"])
        return {"provider": "gemini-native", "model": model, "label": voice, "ok": True,
                "status": r.status_code, "first_byte_ms": round(full), "full_ms": round(full),
                "bytes": len(raw)}
    except Exception as exc:  # noqa: E401 noqa: BLE001
        return {"provider": "gemini-native", "model": model, "label": voice, "ok": False,
                "status": 0, "error": f"{type(exc).__name__}:{exc}"[:150]}


async def elevenlabs(client, key: str) -> list[dict]:
    out = []
    for model_id in ("eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"):
        voice = "21m00Tcm4TlvDq8ikWAM"
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
               f"?output_format=mp3_44100_128")
        body = {"text": TEXT, "model_id": model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        r = await first_byte(client, url, headers={"xi-api-key": key,
                                                   "Content-Type": "application/json"},
                             body=body, label=model_id, provider="elevenlabs", model=model_id)
        out.append(r)
        if r.get("status") in (401, 403, 404):
            break
    return out


async def openai_tts(client, key: str, base: str, provider: str) -> list[dict]:
    out = []
    for model in ("gpt-4o-mini-tts", "tts-1"):
        r = await first_byte(client, f"{base}/audio/speech",
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"},
                             body={"model": model, "input": TEXT, "voice": "alloy",
                                   "response_format": "mp3"},
                             label=model, provider=provider, model=model)
        out.append(r)
        if r.get("status") in (401, 403):
            break
    return out


async def openai_compat_tts(client, key: str, base: str, provider: str, models: list[str]) -> list[dict]:
    out = []
    for model in models:
        r = await first_byte(client, f"{base}/audio/speech",
                             headers={"Authorization": f"Bearer {key}",
                                      "Content-Type": "application/json"},
                             body={"model": model, "input": TEXT, "response_format": "mp3"},
                             label=model, provider=provider, model=model)
        out.append(r)
        if r.get("status") in (401, 402, 403):
            break
    return out


async def gemini_openai_tts(client, key: str) -> list[dict]:
    """Known-dead path, kept so the 404 is documented by measurement."""
    r = await first_byte(client, f"{OPENAI_COMPAT_BASE}/audio/speech",
                         headers={"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"},
                         body={"model": "tts-1", "input": TEXT, "voice": "Puck"},
                         label="openai-compat /audio/speech", provider="gemini-compat",
                         model="tts-1")
    return [r]


async def main() -> None:
    rows: list[dict] = []
    gem = os.environ.get("COMPUSE_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    eleven = os.environ.get("ELEVENLABS_API_KEY") or ""
    di = os.environ.get("DEEPINFRA_API_KEY") or ""
    feather = (os.environ.get("SAWII_FEATHERLESS_API_KEY")
               or os.environ.get("FEATHERLESS_API_KEY") or "")
    oai = os.environ.get("OPENAI_API_KEY") or ""
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY") or ""
    azure_base = os.environ.get("AZURE_OPENAI_BASE_URL") or ""
    async with httpx.AsyncClient() as client:
        if gem:
            for model, voices in (
                ("gemini-3.8-flash-lite-tts", ("Puck", "Charon", "Aoede")),
                ("gemini-2.5-flash-preview-tts", ("Puck",)),
                ("gemini-3.1-flash-tts-preview", ("Puck",)),
                ("gemini-3.8-flash-tts", ("Puck",)),
            ):
                for v in voices:
                    cold = await gemini_tts(client, model, v, gem)
                    samples = [cold]
                    if cold["ok"]:
                        for _ in range(N - 1):
                            samples.append(await gemini_tts(client, model, v, gem))
                    rows.extend(samples)
                    print(json.dumps(samples[-1]), flush=True)
            rows.extend(await gemini_openai_tts(client, gem))
        if eleven:
            for r in await elevenlabs(client, eleven):
                rows.append(r)
                print(json.dumps(r), flush=True)
        if di:
            for r in await openai_compat_tts(client, di, DI, "deepinfra", [
                    "inworld-ai/realtime-tts-1.5-mini", "Qwen/Qwen3-TTS",
                    "ResembleAI/chatterbox-turbo", "XiaomiMiMo/MiMo-V2.5-tts"]):
                rows.append(r)
                print(json.dumps(r), flush=True)
        if feather:
            for r in await openai_compat_tts(client, feather, FEATHER, "featherless",
                                             ["ResembleAI/chatterbox"]):
                rows.append(r)
                print(json.dumps(r), flush=True)
        if oai:
            for r in await openai_tts(client, oai, "https://api.openai.com/v1", "openai"):
                rows.append(r)
                print(json.dumps(r), flush=True)
        if azure_key and azure_base:
            for r in await openai_compat_tts(client, azure_key, azure_base.rstrip("/"),
                                             "azure", ["gpt-4o-mini-tts"]):
                rows.append(r)
    ok = [r for r in rows if r.get("ok")]
    print("\n=== TTS FIRST-BYTE MATRIX ===")
    print(f"{'provider':15s} {'model':36s} {'voice':8s} {'n':>2s} {'fb p50':>7s} {'fb max':>7s} {'full p50':>8s} {'kB':>6s}  err")
    for r in sorted(ok, key=lambda x: x["first_byte_ms"]):
        print(f"{r['provider']:15s} {r['model'][:36]:36s} {str(r['label'])[:8]:8s} "
              f"{'1' if r['first_byte_ms'] else 0:>2s} {str(r['first_byte_ms']):>7s} "
              f"{str(r['first_byte_ms']):>7s} {str(r['full_ms']):>8s} "
              f"{str(r['bytes'] // 1024):>6s}")
    print("--- failures ---")
    seen = set()
    for r in rows:
        if r.get("ok"):
            continue
        k = (r["provider"], r["model"])
        if k in seen:
            continue
        seen.add(k)
        print(f"{r['provider']:15s} {r['model'][:36]:36s} {r.get('status')} {r.get('error','')[:90]}")
    if ok:
        print("\nBEST:", min(ok, key=lambda x: x["first_byte_ms"]))


if __name__ == "__main__":
    asyncio.run(main())