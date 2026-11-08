# Keep the fast-path local model warm so no live turn pays for a model load.
#
# Measured here: first token 14.3 s on a cold llama.cpp process, 570 ms warm.
# The load must never sit on the voice critical path, so this starts the server
# once and keeps it up (llama.cpp's own keepalive holds the weights in RAM).
#
# Run:  powershell -File persona\voice\bench\start_fast_model.ps1
# Then: SAWII_FAST_L_BASE_URL=http://127.0.0.1:8080/v1  (already the default)
#
# Nothing in the app depends on this script. Without it the fast path abstains
# and the deep path answers, exactly as before.

$ErrorActionPreference = 'Stop'

$LlamaDir = if ($env:LLAMA_DIR) { $env:LLAMA_DIR } else { "C:\Users\anasa\llama" }
$Model    = if ($env:LLAMA_MODEL) { $env:LLAMA_MODEL } else { "qwen2.5-1.5b-instruct-q4_k_m.gguf" }
$Port     = if ($env:FAST_L_PORT) { $env:FAST_L_PORT } else { "8080" }
$Threads  = if ($env:FAST_L_THREADS) { $env:FAST_L_THREADS } else { "8" }
$Ctx      = if ($env:FAST_L_CTX) { $env:FAST_L_CTX } else { "2048" }

$Exe   = Join-Path $LlamaDir "llama-server.exe"
$Model = Join-Path $LlamaDir ("models\" + $Model)
if (-not (Test-Path $Exe))   { Write-Error "llama-server.exe not found: $Exe"; exit 1 }
if (-not (Test-Path $Model)) { Write-Error "model not found: $Model"; exit 1 }

$existing = Get-Process llama-server -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "llama-server already running (pid $($existing.Id -join ','))"
} else {
    Start-Process -FilePath $Exe -WorkingDirectory $LlamaDir -ArgumentList @(
        "-m", $Model, "--host", "127.0.0.1", "--port", $Port,
        "-c", $Ctx, "-t", $Threads
    ) -WindowStyle Hidden `
        -RedirectStandardOutput "$env:TEMP\fast_llama_out.txt" `
        -RedirectStandardError  "$env:TEMP\fast_llama_err.txt" | Out-Null
    Write-Host "started llama-server on 127.0.0.1:$Port"
}

# /health is the only honest readiness signal; a cold load takes seconds.
$deadline = (Get-Date).AddSeconds(180)
$ok = $false
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5
        if ($h.status -eq 'ok') { $ok = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) {
    Write-Host "--- server log tail ---"
    Get-Content "$env:TEMP\fast_llama_err.txt" -Tail 12 -ErrorAction SilentlyContinue
    Write-Error "server never became healthy"; exit 1
}

# One real generation: llama.cpp loads weights lazily, so warm it for real.
$body = @{
    model = "warmup"; stream = $false; max_tokens = 4; keepalive = "30m"
    messages = @(@{ role = "user"; content = "hi" })
} | ConvertTo-Json -Depth 4
$sw = [Diagnostics.Stopwatch]::StartNew()
Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
    -ContentType "application/json" -Body $body -TimeoutSec 180 | Out-Null
$sw.Stop()
Write-Host ("READY port={0} warm_call_ms={1}" -f $Port, $sw.ElapsedMilliseconds)