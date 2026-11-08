"""Piper as a pip package on Python 3.14, then its real first-byte latency.

onnxruntime is already installed here, so only the piper code is needed: the
official piper-tts wheel ships no 3.14 wheel and its binary bundle is what
usually fails to build. We install with --no-deps and reuse the existing
onnxruntime.

Run: powershell -NoProfile -ExecutionPolicy Bypass -File persona\voice\bench\piper_pip.ps1
"""
$ErrorActionPreference = 'Continue'
$VoiceDir = "C:\Users\anasa\localtts\voices"
$Voice = Join-Path $VoiceDir "en_US-lessac-low.onnx"

python -m pip install --no-deps --only-binary :all: piper-tts 2>&1 | Select-Object -Last 3
if (-not (Test-Path $Voice)) {
    Write-Host "voice missing: $Voice"
    exit 1
}
Write-Host "=== piper python smoke ==="
python -c @"
import time, json, io
from piper import PiperVoice
v = PiperVoice.load(r'$Voice')
import wave
for text in ('Okay, one second.', 'Hey, that is a good question, let me think for a second.'):
    buf = io.BytesIO()
    t0 = time.perf_counter()
    first = None
    for audio in v.synthesize(text):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        buf.write(audio.audio_int16_bytes)
    full = (time.perf_counter() - t0) * 1000
    print(json.dumps({'provider':'piper-pip','text':text[:28],'first_byte_ms':round(first),
                      'full_ms':round(full),'bytes':buf.tell()}), flush=True)
"@