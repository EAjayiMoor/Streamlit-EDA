# Deployment guide (Azure App Service via ACR)

This project deploys to Azure App Service using an image built in Azure Container Registry (ACR).

## Target environment

- Resource Group: `RG-DEMO-UKW`
- App Service: `NHS-EDA-demo`
- ACR: `crmhcdemo`

## Prerequisites

- Azure CLI installed and available as `az`
- Logged in to Azure (`az login`)
- Access to resource group, App Service, and ACR

## Quick deploy (recommended)

From the repository root:

```powershell
.\scripts\deploy_latest.ps1
```

This will:

1. Enable container CD on the App Service
2. Build and push a timestamp-tagged image to ACR
3. Update App Service to the new image tag
4. Restart App Service
5. Print final container configuration

## Custom deploy options

Use the main script directly if needed:

```powershell
.\scripts\deploy_to_azure.ps1 `
  -ResourceGroup "RG-DEMO-UKW" `
  -AppName "NHS-EDA-demo" `
  -AcrName "crmhcdemo" `
  -ImageName "streamlit-eda" `
  -ImageTag "20260824153000"
```

If `-ImageTag` is omitted, a timestamp is used automatically.

## Verify deployment

```powershell
az webapp config container show -g RG-DEMO-UKW -n NHS-EDA-demo
az webapp restart -g RG-DEMO-UKW -n NHS-EDA-demo
```

## Troubleshooting

- `az` not found: install Azure CLI and reopen terminal.
- Not logged in: run `az login`.
- Permission errors: ensure your account has access to App Service and ACR.
- App not updating: rerun deploy and confirm the configured image tag changed.
