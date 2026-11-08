# End-to-end first-AUDIO on turns the fast gate admits, fast L off vs on.
#
# Uses TTS from the environment (do not set SAWII_TTS_* here; the caller picks
# the provider). --prompts social selects utterances the safety gate lets the
# fast responder answer, which is the population the fast path can actually win.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File persona\voice\bench\ab_speech_social.ps1

$ErrorActionPreference = 'Continue'
$Turns = if ($env:AB_TURNS) { [int]$env:AB_TURNS } else { 6 }
$Out   = if ($env:AB_OUT) { $env:AB_OUT } else { Join-Path $env:TEMP "sawii_social" }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$env:SAWII_DUAL_LOBE_VOICE = "0"
$env:SAWII_FAST_L_BASE_URL = "http://127.0.0.1:8080/v1"
$env:SAWII_FAST_L_MODEL    = "qwen2.5-1.5b-instruct-q4_k_m"
$env:PYTHONIOENCODING      = "utf-8"

foreach ($arm in @("off", "on")) {
    if ($arm -eq "off") { $env:SAWII_FAST_L = "0" } else { $env:SAWII_FAST_L = "1" }
    Write-Host "=== SPEECH SOCIAL fast_l=$env:SAWII_FAST_L (tts=$env:SAWII_TTS_MODEL stream=$env:SAWII_TTS_STREAM) ==="
    python persona\voice\bench\voice_latency.py --turns $Turns --prompts social `
        --label "social_$arm" --json (Join-Path $Out "social_$arm.json")
}
Write-Host "wrote $Out"