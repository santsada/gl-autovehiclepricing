#!/usr/bin/env bash
# Creates the Azure resources and GitHub secret needed by the pipeline.
set -euo pipefail

SUBSCRIPTION_ID="78407db7-ee1b-45db-86c3-dfe2aeb169cf"      # az account list -o table
LOCATION="eastus"                            # any region with ML quota
RESOURCE_GROUP="gl-resourcegrp"   # must equal config-infra-prod.yml
WORKSPACE="gl-workspace"         # must equal config-infra-prod.yml
GITHUB_REPO="santsada/autovehiclepricing"  # e.g. my-user/used-cars-mlops
SP_NAME="Azure-ARM-Prod-AutoVehiclePricing"             # service principal display name

if [[ "$SUBSCRIPTION_ID" == \<* || "$RESOURCE_GROUP" == \<* || "$WORKSPACE" == \<* || "$GITHUB_REPO" == \<* ]]; then
  echo "Edit the variables at the top of this script first." >&2
  exit 1
fi

az account set --subscription "$SUBSCRIPTION_ID"
az extension add --name ml --yes >/dev/null 2>&1 || az extension update --name ml >/dev/null

# Resource providers can be unregistered in a fresh subscription.
for ns in Microsoft.MachineLearningServices Microsoft.ContainerRegistry \
          Microsoft.Storage Microsoft.KeyVault Microsoft.Insights \
          Microsoft.OperationalInsights Microsoft.Cache; do
  az provider register --namespace "$ns" --wait
done

az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none
az ml workspace create --name "$WORKSPACE" --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" -o none
echo "Workspace ready: $WORKSPACE"

# The service principal is scoped to the resource group only (least privilege that
# still lets the workflow create compute, register assets and submit jobs).
# The JSON (which contains a secret) is piped straight into the GitHub secret and
# never written to disk or printed.
# NOTE: `--sdk-auth` is deprecated in newer Azure CLI versions but still works. If

az ad sp create-for-rbac \
  --name "$SP_NAME" \
  --role contributor \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  --sdk-auth | gh secret set AZURE_CREDENTIALS --repo "$GITHUB_REPO"

echo "Secret AZURE_CREDENTIALS stored in $GITHUB_REPO."
echo "Next: set resource_group/aml_workspace in config-infra-prod.yml, then push to main."
