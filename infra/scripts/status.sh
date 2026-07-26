#!/usr/bin/env bash
set -euo pipefail

environment="${1:?usage: status.sh <environment> [--json] [--env-file FILE]}"
json=false
while [[ $# -gt 1 ]]; do
  shift
  case "$1" in
    --json) json=true ;;
    --env-file) shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if "$json"; then
  printf '{"environment":"%s","status":"not_configured"}\n' "$environment"
else
  printf 'Environment: %s\nStatus: not_configured (Terraform configuration pending)\n' "$environment"
fi
