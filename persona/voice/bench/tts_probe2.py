"""TTS first-audio-byte: STREAMING variants + local engines.

1. Gemini native TTS with streamGenerateContent (SSE) -> first audio chunk time
   vs the non-streaming number already measured (~2.2 s p50).
2. Windows SAPI (HoloSpeech/David/Zira) through PowerShell System.Speech.
3. Piper local (once the binary is unpacked).

Prints one JSON line per sample. Keys from env only.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import httpx  # noqa: E402

from persona.config import load_dotenv  # noqa: E402

load_dotenv()

NATIVE = "https://generativelanguage.googleapis.com/v1beta/models"
TEXT = "Hey, that's a good question, let me think for a second."
N = int(os.environ.get("T2_N", "5"))


def row(provider, model, label, ok, **kw):
    out = {"provider": provider, "model": model, "label": label, "ok": ok}
    out.update(kw)
    print(json.dumps(out), flush=True)
    return out


async def gemini_stream(client, model: str, voice: str, key: str) -> dict:
    url = f"{NATIVE}/{model}:streamGenerateContent?alt=sse"
    body = {
        "contents": [{"parts": [{"text": TEXT}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    t0 = time.perf_counter()
    first = None
    audio_bytes = 0
    try:
        async with client.stream("POST", url, headers={"x-goog-api-key": key}, json=body,
                                 timeout=60) as r:
            if r.status_code >= 400:
                raw = (await r.aread())[:160]
                return row("gemini-stream", model, voice, False, status=r.status_code,
                           error=raw.decode("utf-8", "replace")[:120])
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except ValueError:
                    continue
                for cand in obj.get("candidates") or []:
                    for p in (cand.get("content") or {}).get("parts") or []:
                        inline = p.get("inlineData") or p.get("inline_data")
                        if inline and inline.get("data"):
                            chunk = base64.b64decode(inline["data"])
                            if first is None:
                                first = (time.perf_counter() - t0) * 1000
                            audio_bytes += len(chunk)
        full = (time.perf_counter() - t0) * 1000
        if first is None:
            return row("gemini-stream", model, voice, False, status=r.status_code,
                       error="no audio part")
        return row("gemini-stream", model, voice, True,
                   first_byte_ms=round(first), full_ms=round(full), bytes=audio_bytes)
    except Exception as exc:  # noqa: BLE001
        return row("gemini-stream", model, voice, False, status=0,
                   error=f"{type(exc).__name__}:{exc}"[:120])


SAPI_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$ms = New-Object System.IO.MemoryStream
$s.SetOutputToWave($ms)
$s.Rate = 1
$s.Speak($env:BENCH_TEXT)
[IO.File]::WriteAllBytes($env:BENCH_OUT, $ms.ToArray())
"""


def sapi_once(text: str, out_path: str) -> dict:
    t0 = time.perf_counter()
    env = dict(os.environ, BENCH_TEXT=text, BENCH_OUT=out_path)
    p = subprocess.run(["powershell", "-NoProfile", "-Command", SAPI_PS],
                       env=env, capture_output=True, text=True)
    wall = (time.perf_counter() - t0) * 1000
    if p.returncode != 0 or not Path(out_path).exists():
        return {"provider": "windows-sapi", "model": "System.Speech", "label": "powershell",
                "ok": False, "error": (p.stderr or "")[:120], "full_ms": round(wall)}
    size = Path(out_path).stat().st_size
    return {"provider": "windows-sapi", "model": "System.Speech", "label": "powershell",
            "ok": size > 1000, "full_ms": round(wall), "first_byte_ms": round(wall),
            "bytes": size, "error": "" if size > 1000 else "empty wav"}


def piper_once(binary: str, voice: str, text: str, out_path: str) -> dict:
    """Piper writes the wav when it finishes; process wall time == first-byte floor."""
    t0 = time.perf_counter()
    p = subprocess.run([binary, "-m", voice, "-f", out_path], input=text.encode(),
                       capture_output=True)
    wall = (time.perf_counter() - t0) * 1000
    if p.returncode != 0 or not Path(out_path).exists():
        return {"provider": "piper-local", "model": Path(voice).name, "label": "subprocess",
                "ok": False, "error": (p.stderr or b"")[:120].decode("utf-8", "replace"),
                "full_ms": round(wall)}
    size = Path(out_path).stat().st_size
    return {"provider": "piper-local", "model": Path(voice).name, "label": "subprocess",
            "ok": size > 1000, "first_byte_ms": round(wall), "full_ms": round(wall),
            "bytes": size}


async def main() -> None:
    key = os.environ.get("COMPUSE_LLM_API_KEY") or ""
    rows = []
    async with httpx.AsyncClient() as client:
        if key:
            for model in ("gemini-3.8-flash-lite-tts", "gemini-2.5-flash-preview-tts"):
                for voice in ("Puck",):
                    for _ in range(N):
                        rows.append(await gemini_stream(client, model, voice, key))
        else:
            print("no COMPUSE_LLM_API_KEY; skipping gemini stream")

    # local engines
    out = os.path.join(os.environ.get("TEMP", "."), "t2_local.wav")
    short = "Okay, one second."
    long_ = TEXT
    sapi = [sapi_once(long_, out) for _ in range(2)]
    rows.extend(sapi)
    for r in sapi:
        print(json.dumps(r), flush=True)

    piper_dir = Path(r"C:\Users\anasa\localtts\piper")
    binary = str(piper_dir / "piper.exe")
    voice = r"C:\Users\anasa\localtts\voices\en_US-lessac-low.onnx"
    if Path(binary).exists() and Path(voice).exists():
        for txt, lbl in ((long_, "clause"), (short, "short")):
            for _ in range(3):
                r = piper_once(binary, voice, txt, out)
                r["label"] = lbl
                rows.append(r)
                print(json.dumps(r), flush=True)
    else:
        print(json.dumps({"provider": "piper-local", "model": "-", "ok": False,
                          "error": f"missing binary or voice: {Path(binary).exists()} "
                                   f"{Path(voice).exists()}"}))

    ok = [r for r in rows if r.get("ok")]
    print("\n=== TTS ROUND 2 ===")
    groups: dict[tuple, list] = {}
    for r in ok:
        groups.setdefault((r["provider"], r["model"], r.get("label")), []).append(r)
    for k, items in sorted(groups.items(), key=lambda kv: min(i["first_byte_ms"] for i in kv[1])):
        fb = [i["first_byte_ms"] for i in items]
        print(f"{k[0]:16s} {k[1][:36]:36s} {str(k[2]):8s} n={len(items)} "
              f"fb p50={round(statistics.median(fb))} min={min(fb)} max={max(fb)}")


if __name__ == "__main__":
    asyncio.run(main())