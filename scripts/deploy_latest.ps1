$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$deployScript = Join-Path $PSScriptRoot "deploy_to_azure.ps1"
if (-not (Test-Path $deployScript)) {
    throw "Missing deploy script: $deployScript"
}

Write-Host "Starting Azure deploy with default settings..." -ForegroundColor Cyan
& $deployScript

Write-Host "Done." -ForegroundColor Green
Read-Host "Press Enter to close"
