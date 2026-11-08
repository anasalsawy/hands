# One real end-to-end first-AUDIO run of the fast path, on the live provider.
#
# TTS comes from the environment; nothing here overrides it, so this measures the
# provider the repo actually resolves today. --prompts social keeps every turn
# inside the population the fast responder is allowed to answer.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File persona\voice\bench\speech_after.ps1

$ErrorActionPreference = 'Continue'
$Turns = if ($env:AB_TURNS) { [int]$env:AB_TURNS } else { 5 }
$Out   = if ($env:AB_OUT) { $env:AB_OUT } else { Join-Path $env:TEMP "sawii_after" }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$env:SAWII_DUAL_LOBE_VOICE = "0"
$env:SAWII_FAST_L_BASE_URL = "http://127.0.0.1:8080/v1"
$env:SAWII_FAST_L_MODEL    = "qwen2.5-1.5b-instruct-q4_k_m"
$env:SAWII_FAST_L          = "1"
$env:PYTHONIOENCODING      = "utf-8"

Write-Host "=== AFTER: fast L on, TTS=$env:SAWII_TTS_MODEL stream=$env:SAWII_TTS_STREAM ==="
python persona\voice\bench\voice_latency.py --turns $Turns --prompts social `
    --label after_fast --json (Join-Path $Out "after_fast.json")