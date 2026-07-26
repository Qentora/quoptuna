#!/usr/bin/env bash
set -euo pipefail
action="${1:?missing action}"
environment="${2:?usage: <action> <environment> [--env-file FILE]}"
case "$environment" in
  dev|production) ;;
  *) echo "Unsupported environment: $environment" >&2; exit 2 ;;
esac
echo "$action is not configured yet for $environment. Add Terraform configuration under infra/environments/$environment." >&2
exit 2
