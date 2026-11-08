"""Why did the A/B arms fail? Print the real status code behind the fallback.

Run: python persona/voice/bench/tts_status.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from persona.config import load_dotenv  # noqa: E402

load_dotenv()

NATIVE = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL = "gemini-3.8-flash-lite-tts"


async def main() -> None:
    key = os.environ.get("COMPUSE_LLM_API_KEY") or ""
    if not key:
        print("no key")
        return
    body = {
        "contents": [{"parts": [{"text": "Okay, one second."}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {
                "voiceName": "Puck"}}},
        },
    }
    async with httpx.AsyncClient() as c:
        for kind, url in (("non-stream", f"{NATIVE}/{MODEL}:generateContent"),
                          ("stream", f"{NATIVE}/{MODEL}:streamGenerateContent?alt=sse")):
            try:
                r = await c.post(url, headers={"x-goog-api-key": key}, json=body, timeout=45)
                print(f"{kind:12s} status={r.status_code} bytes={len(r.content)} "
                      f"body={r.text[:200] if r.status_code >= 400 else '<audio ok>'}")
            except Exception as exc:  # noqa: BLE001
                print(f"{kind:12s} EXC {type(exc).__name__}: {exc}")
            await asyncio.sleep(1)
        # And the plain chat hop, to see if the whole key is throttled.
        try:
            r = await c.post("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                             headers={"Authorization": f"Bearer {key}"},
                             json={"model": "gemini-flash-lite-latest", "max_tokens": 8,
                                   "messages": [{"role": "user", "content": "hi"}]},
                             timeout=45)
            print(f"{'chat':12s} status={r.status_code} body={r.text[:180] if r.status_code >= 400 else '<ok>'}")
        except Exception as exc:  # noqa: BLE001
            print(f"{'chat':12s} EXC {type(exc).__name__}")


if __name__ == "__main__":
    asyncio.run(main())