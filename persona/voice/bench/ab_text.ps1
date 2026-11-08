# Isolates the FAST_L hop with TTS out of the picture.
#
# Voice output is text, so no speech call happens at all: first_audio_ms stays
# null and first_delta_ms measures the LLM path alone, with no provider quota
# inside the numbers. This is the honest way to price the fast responder while
# the Gemini speech quota is exhausted.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File persona\voice\bench\ab_text.ps1

$ErrorActionPreference = 'Continue'
$Turns = if ($env:AB_TURNS) { [int]$env:AB_TURNS } else { 8 }
$Out   = if ($env:AB_OUT) { $env:AB_OUT } else { Join-Path $env:TEMP "sawii_voice_ab_text" }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$env:SAWII_DUAL_LOBE_VOICE = "0"
$env:SAWII_FAST_L_BASE_URL = "http://127.0.0.1:8080/v1"
$env:SAWII_FAST_L_MODEL    = "qwen2.5-1.5b-instruct-q4_k_m"
$env:PYTHONIOENCODING      = "utf-8"

foreach ($arm in @("off", "on")) {
    if ($arm -eq "off") { $env:SAWII_FAST_L = "0" } else { $env:SAWII_FAST_L = "1" }
    Write-Host "=== TEXT ARM: fast_l=$env:SAWII_FAST_L ==="
    python persona\voice\bench\voice_latency.py --turns $Turns --text-only --label "text_$arm" `
        --json (Join-Path $Out "text_$arm.json")
}
Write-Host "wrote $Out"