param(
    [string]$ResourceGroup = "RG-DEMO-UKW",
    [string]$AppName = "NHS-EDA-demo",
    [string]$AcrName = "crmhcdemo",
    [string]$ImageName = "streamlit-eda",
    [string]$ImageTag
)

$ErrorActionPreference = "Stop"

if (-not $ImageTag) {
    $ImageTag = Get-Date -Format "yyyyMMddHHmmss"
}

Write-Host "Using image tag: $ImageTag" -ForegroundColor Cyan

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is not installed or not on PATH. Install it first, then rerun this script."
}

az account show 1>$null 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "You are not logged into Azure CLI. Running az login..." -ForegroundColor Yellow
    az login
}

Write-Host "Enabling container continuous deployment..." -ForegroundColor Green
az webapp deployment container config --enable-cd true -g $ResourceGroup -n $AppName

Write-Host "Building and pushing image to ACR..." -ForegroundColor Green
az acr build -r $AcrName -t "$ImageName`:$ImageTag" .

$containerImage = "$AcrName.azurecr.io/$ImageName`:$ImageTag"
Write-Host "Pointing App Service to image: $containerImage" -ForegroundColor Green
az webapp config container set `
  -g $ResourceGroup -n $AppName `
  --container-image-name $containerImage `
  --docker-registry-server-url "https://$AcrName.azurecr.io"

Write-Host "Restarting App Service..." -ForegroundColor Green
az webapp restart -g $ResourceGroup -n $AppName

Write-Host "Configured container:" -ForegroundColor Green
az webapp config container show -g $ResourceGroup -n $AppName

Write-Host "Deployment complete." -ForegroundColor Cyan
