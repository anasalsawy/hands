# Text-only A/B on turns the fast gate admits: the fast L hop, TTS-free.
#
# --text-only means no speech call at all, so no provider quota sits inside the
# numbers, and --prompts social selects the utterances the safety gate lets the
# fast responder answer. first_delta_ms is then "time to first audible-equivalent
# word", and fast_l_* on every row says exactly what the fast path did.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File persona\voice\bench\ab_text_social.ps1

$ErrorActionPreference = 'Continue'
$Turns = if ($env:AB_TURNS) { [int]$env:AB_TURNS } else { 6 }
$Out   = if ($env:AB_OUT) { $env:AB_OUT } else { Join-Path $env:TEMP "sawii_soc_text" }

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$env:SAWII_DUAL_LOBE_VOICE = "0"
$env:SAWII_FAST_L_BASE_URL = "http://127.0.0.1:8080/v1"
$env:SAWII_FAST_L_MODEL    = "qwen2.5-1.5b-instruct-q4_k_m"
$env:PYTHONIOENCODING      = "utf-8"

foreach ($arm in @("off", "on")) {
    if ($arm -eq "off") { $env:SAWII_FAST_L = "0" } else { $env:SAWII_FAST_L = "1" }
    Write-Host "=== SOCIAL TEXT ARM fast_l=$env:SAWII_FAST_L ==="
    python persona\voice\bench\voice_latency.py --turns $Turns --prompts social --text-only `
        --label "soc_$arm" --json (Join-Path $Out "soc_$arm.json")
}
Write-Host "wrote $Out"