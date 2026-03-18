#!/usr/bin/env bash
set -euo pipefail

# Sync Azure deploy credentials into GitHub Actions secrets.
# Requires:
# - az logged into the target subscription
# - gh authenticated for the target GitHub repo

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-reasonlens}"
FUNCTION_APP="${FUNCTION_APP:-reasonlens-api}"
STATIC_WEB_APP="${STATIC_WEB_APP:-reasonlens-app}"
GITHUB_REPO="${GITHUB_REPO:-AI-For-Global-Education/reasonlens}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd az
require_cmd gh

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login -h github.com" >&2
  exit 1
fi

tmp_publish_profile="$(mktemp)"
trap 'rm -f "$tmp_publish_profile"' EXIT

az functionapp deployment list-publishing-profiles \
  -g "$RESOURCE_GROUP" \
  -n "$FUNCTION_APP" \
  --xml > "$tmp_publish_profile"

swa_token="$(az staticwebapp secrets list \
  -g "$RESOURCE_GROUP" \
  -n "$STATIC_WEB_APP" \
  --query properties.apiKey \
  -o tsv)"

if [ -z "$swa_token" ]; then
  echo "Failed to fetch Static Web App deployment token." >&2
  exit 1
fi

gh secret set AZURE_FUNCTIONS_PUBLISH_PROFILE --repo "$GITHUB_REPO" < "$tmp_publish_profile"
printf "%s" "$swa_token" | gh secret set SWA_DEPLOYMENT_TOKEN --repo "$GITHUB_REPO"

echo "GitHub Actions secrets updated for $GITHUB_REPO"
