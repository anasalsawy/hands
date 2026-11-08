<#
    start_fast_l.ps1 - bring up the reactive-L server for the voice fast path.

    Loopback ONLY (--host 127.0.0.1) and keyless-by-accident is NOT allowed: it
    always passes --api-key, so nothing on the LAN can use this box as a free
    inference endpoint. Threads are capped low because this machine is usually
    near its memory ceiling (a PowerShell call has already thrown
    OutOfMemoryException while llama.cpp held 8 threads).

    Point the persona at it with env only (never a tracked file):
        $env:SAWII_FAST_L_BASE_URL = "http://127.0.0.1:8080/v1"
        $env:SAWII_FAST_L_MODEL    = "qwen2.5-1.5b-instruct-q4_k_m"
        $env:SAWII_FAST_L_API_KEY  = $env:SAWII_LOCAL_L_API_KEY

    Measured on this box (2026-11-08, persona/voice/bench numbers):
        warm first token p50 307 ms  (n=10, min 205, p95 1424)
        cold process first token ~6.8 s - which is why VoiceSession warms it.
#>
param(
    [string]$Model   = "C:\Users\anasa\llama\models\qwen2.5-1.5b-instruct-q4_k_m.gguf",
    [string]$Exe     = "C:\Users\anasa\llama\llama-server.exe",
    [int]   $Port    = 8080,
    [int]   $Threads = 4,
    [int]   $Ctx     = 2048,
    [string]$Host_   = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

if ($Host_ -ne "127.0.0.1" -and $Host_ -ne "localhost") {
    throw "refusing to bind $Host_ : the reactive-L listener must stay on loopback"
}
if (-not (Test-Path $Exe))   { throw "llama-server not found: $Exe" }
if (-not (Test-Path $Model)) { throw "gguf not found: $Model" }

# A key is mandatory. It is read from the environment, never hardcoded.
$Key = $env:SAWII_LOCAL_L_API_KEY
if ([string]::IsNullOrWhiteSpace($Key)) {
    $Key = "local-voice-key"
    Write-Warning "SAWII_LOCAL_L_API_KEY unset; using the placeholder 'local-voice-key'. Set a real value for anything shared."
}

$ModelName = [IO.Path]::GetFileNameWithoutExtension($Model)

$args = @(
    '-m', $Model,
    '--host', $Host_,          # loopback only
    '--port', "$Port",
    '-t', "$Threads",
    '-c', "$Ctx",
    '--api-key', $Key,         # never a keyless listener
    '--keep-alive', '30m',     # weights + slots stay resident between turns
    '--no-webui'
)

$err = Join-Path $env:TEMP "llama_server_$Port.err"
$out = Join-Path $env:TEMP "llama_server_$Port.out"
$p = Start-Process -FilePath $Exe -ArgumentList $args -PassThru -WindowStyle Hidden `
        -RedirectStandardError $err -RedirectStandardOutput $out
Write-Host "llama-server pid=$($p.Id) on ${Host_}:$Port model=$ModelName threads=$Threads ctx=$Ctx (api-key set)"

$health = "http://127.0.0.1:$Port/health"
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    try {
        if ((Invoke-WebRequest $health -TimeoutSec 5 -UseBasicParsing).StatusCode -eq 200) {
            Write-Host "healthy after $i s"
            Write-Host "export: SAWII_FAST_L_BASE_URL=http://127.0.0.1:$Port/v1"
            Write-Host "export: SAWII_FAST_L_MODEL=$ModelName"
            Write-Host "export: SAWII_FAST_L_API_KEY=`$env:SAWII_LOCAL_L_API_KEY"
            exit 0
        }
    } catch { }
}
Write-Error "llama-server did not become healthy; see $err"
exit 1