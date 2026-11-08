# Before/after run of the voice first-audio harness, one hop at a time.
#
# ARM        FAST_L   TTS                        what it isolates
# baseline    0      production default         today's code, today's config
# tts         0      gemini native STREAMING    the TTS hop alone
# fast        1      gemini native STREAMING    + the fast local responder
#
# Nothing else changes between arms: same mind, kernel, DualLoop, prompts and
# turn order. Keys come from env only.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File persona\voice\bench\ab_latency.ps1
# Env: AB_TURNS=8  AB_OUT=<dir>

$ErrorActionPreference = 'Continue'
$Turns = if ($env:AB_TURNS) { [int]$env:AB_TURNS } else { 8 }
$Out   = if ($env:AB_OUT) { $env:AB_OUT } else { Join-Path $env:TEMP "sawii_voice_ab" }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$env:SAWII_DUAL_LOBE_VOICE = "0"   # off in every arm: the old bridge never worked
$env:SAWII_FAST_L_BASE_URL = "http://127.0.0.1:8080/v1"
$env:SAWII_FAST_L_MODEL    = "qwen2.5-1.5b-instruct-q4_k_m"
$env:PYTHONIOENCODING      = "utf-8"

foreach ($arm in @("baseline", "tts", "fast")) {
    # Baseline inherits whatever persona/config.py resolves today.
    Remove-Item Env:SAWII_TTS_BASE_URL, Env:SAWII_TTS_MODEL, Env:SAWII_TTS_API_KEY,
                Env:SAWII_TTS_VOICE, Env:SAWII_TTS_STREAM -ErrorAction SilentlyContinue
    if ($arm -eq "baseline") {
        $env:SAWII_FAST_L = "0"
    } else {
        $env:SAWII_TTS_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
        $env:SAWII_TTS_MODEL    = "gemini-3.8-flash-lite-tts"
        $env:SAWII_TTS_API_KEY  = $env:COMPUSE_LLM_API_KEY
        $env:SAWII_TTS_VOICE    = "Puck"
        $env:SAWII_TTS_STREAM   = "1"
        if ($arm -eq "fast") { $env:SAWII_FAST_L = "1" } else { $env:SAWII_FAST_L = "0" }
    }
    Write-Host "=== ARM: $arm (fast_l=$env:SAWII_FAST_L tts=$env:SAWII_TTS_MODEL) ==="
    python persona\voice\bench\voice_latency.py --turns $Turns --label $arm `
        --json (Join-Path $Out "$arm.json")
}
Write-Host "wrote $Out\baseline.json $Out\tts.json $Out\fast.json"