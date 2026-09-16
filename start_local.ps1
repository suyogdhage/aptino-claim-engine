param(
    [string]$RenderToken = $env:RENDER_API_KEY,
    [string]$UiServiceId = "srv-dal9b4ajnfac73cto3j0",
    [string]$ApiPort = "8000"
)

$ErrorActionPreference = "Stop"
$log = "$env:TEMP\aptino_restart.log"
"==== Aptino restart at $(Get-Date) ====" | Out-File -FilePath $log

function Log($msg) { Write-Output $msg; $msg | Out-File -FilePath $log -Append }

# Load .env for keys (GROQ_API_KEY, GROQ_MODEL, CORS_ORIGINS)
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.*)$") {
            Set-Item -Path "Env:$($matches[1].Trim())" -Value $matches[2].Trim()
        }
    }
    Log "Loaded .env"
}
$env:GROQ_MODEL = if ($env:GROQ_MODEL) { $env:GROQ_MODEL } else { "openai/gpt-oss-20b" }
$env:CORS_ORIGINS = if ($env:CORS_ORIGINS) { $env:CORS_ORIGINS } else { "https://aptino-claim-ui.onrender.com,http://localhost:8501" }

# 1. Stop old processes
Get-Process -Name uvicorn,python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*aptino*"
} | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Log "Stopped old processes"

# 2. Start backend
$stdout = "$env:TEMP\local_api_stdout.log"
$stderr = "$env:TEMP\local_api_stderr.log"
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", $ApiPort `
    -WorkingDirectory (Get-Location) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru |
    ForEach-Object { Log "Backend PID=$($_.Id)" }

$ready = $false
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 5
        if ($h.status -eq "healthy") { $ready = $true; Log "Backend healthy (${i}0s)"; break }
    } catch {}
}
if (-not $ready) { Log "Backend did not become healthy - see $stderr"; exit 1 }

# 3. Start tunnel
$cf = "$env:USERPROFILE\cloudflared.exe"
$cfLog = "$env:TEMP\cf_tunnel.log"
$cfErr = "$env:TEMP\cf_tunnel_err.log"
Start-Process -FilePath $cf -ArgumentList "tunnel", "--url", "http://127.0.0.1:$ApiPort", "--no-autoupdate" `
    -RedirectStandardOutput $cfLog -RedirectStandardError $cfErr -PassThru |
    ForEach-Object { Log "Tunnel PID=$($_.Id)" }

$tunnelUrl = ""
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Seconds 5
    $u = (Select-String -Path $cfErr -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue).Matches.Value | Select-Object -Last 1
    if ($u) { $tunnelUrl = $u; break }
}
if (-not $tunnelUrl) { Log "Tunnel URL not found - see $cfErr"; exit 1 }
Log "Tunnel URL: $tunnelUrl"

# 4. Verify through tunnel
Start-Sleep -Seconds 3
$h2 = Invoke-RestMethod -Uri "$tunnelUrl/health" -TimeoutSec 30
Log "Tunnel health: $($h2.status)"

# 5. Update Render UI env var + redeploy
if ($RenderToken) {
    try {
        curl.exe -s -X PUT "https://api.render.com/v1/services/$UiServiceId/env-vars" `
            -H "Authorization: Bearer $RenderToken" -H "Content-Type: application/json" `
            -d "[{\"key\":\"API_BASE_URL\",\"value\":\"$tunnelUrl\"}]" | Out-Null
        Log "Render env var updated"
        $dep = curl.exe -s -X POST "https://api.render.com/v1/services/$UiServiceId/deploys" `
            -H "Authorization: Bearer $RenderToken" -H "Content-Type: application/json" -d '{}' | ConvertFrom-Json
        Log "Render redeploy triggered: $($dep.id) ($($dep.status))"
    } catch {
        Log "NOTE: could not update Render env var - set API_BASE_URL=$tunnelUrl manually in the Render dashboard."
    }
} else {
    Log "No RenderToken - do NOT forget to set API_BASE_URL=$tunnelUrl in the Render dashboard (Service > Environment > Values and redeploy)."
}

Log "ALL DONE"
Log "Backend : http://127.0.0.1:$ApiPort"
Log "Tunnel  : $tunnelUrl"
Log "UI      : https://aptino-claim-ui.onrender.com"