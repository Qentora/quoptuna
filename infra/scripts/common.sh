#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$INFRA_DIR/.." && pwd)"
FOUNDATION_DIR="$INFRA_DIR/terraform/foundation"
APPLICATION_DIR="$INFRA_DIR/terraform/application"
# Consumed by the operation dispatcher after sourcing this library.
# shellcheck disable=SC2034
RUNTIME_DIR="$INFRA_DIR/runtime"

ENVIRONMENT=""
ENV_FILE="$REPO_DIR/.env"
JSON_OUTPUT=false
FORCE=false
DELETE_DATA=false
PLAN_ONLY=false
CONFIRM_DESTROY=false

log() {
  if [[ "$JSON_OUTPUT" != true ]]; then
    printf '%s\n' "$*"
  fi
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_var() {
  [[ -n "${!1:-}" ]] || die "Set $1 in $ENV_FILE or the process environment"
}

# Flags assigned here are consumed by the operation dispatcher.
# shellcheck disable=SC2034
parse_args() {
  [[ $# -ge 1 ]] || die "usage: <script> <dev|production> [options]"
  ENVIRONMENT="$1"
  shift
  case "$ENVIRONMENT" in
    dev|production) ;;
    *) die "Unsupported environment: $ENVIRONMENT" ;;
  esac
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env-file) [[ $# -ge 2 ]] || die "--env-file requires a path"; ENV_FILE="$2"; shift 2 ;;
      --json) JSON_OUTPUT=true; shift ;;
      --force) FORCE=true; shift ;;
      --delete-data) DELETE_DATA=true; shift ;;
      --plan-only) PLAN_ONLY=true; shift ;;
      --confirm-destroy) CONFIRM_DESTROY=true; shift ;;
      *) die "Unknown option: $1" ;;
    esac
  done
}

load_deployment_env() {
  require_command python3
  [[ -f "$ENV_FILE" ]] || die "Environment file not found: $ENV_FILE"
  eval "$(python3 "$SCRIPT_DIR/envfile.py" export --file "$ENV_FILE")"
  PROJECT_NAME="${PROJECT_NAME:-quoptuna}"
  INSTANCE_TYPE="${INSTANCE_TYPE:-t3.large}"
  ROOT_VOLUME_SIZE="${ROOT_VOLUME_SIZE:-50}"
  require_var AWS_REGION
  require_var TF_STATE_BUCKET
  require_var DOMAIN_NAME
  require_var ROUTE53_ZONE_ID
  AWS_DEFAULT_REGION="$AWS_REGION"
  export AWS_REGION AWS_DEFAULT_REGION
}

check_tools() {
  for tool in aws terraform jq; do
    require_command "$tool"
  done
  aws sts get-caller-identity >/dev/null || die "AWS credentials are unavailable"
}

bootstrap_state() {
  if ! aws s3api head-bucket --bucket "$TF_STATE_BUCKET" >/dev/null 2>&1; then
    log "Creating Terraform state bucket $TF_STATE_BUCKET"
    if [[ "$AWS_REGION" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "$TF_STATE_BUCKET" --region "$AWS_REGION" >/dev/null
    else
      aws s3api create-bucket --bucket "$TF_STATE_BUCKET" --region "$AWS_REGION" \
        --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
    fi
  fi
  aws s3api put-bucket-versioning --bucket "$TF_STATE_BUCKET" \
    --versioning-configuration Status=Enabled
  aws s3api put-public-access-block --bucket "$TF_STATE_BUCKET" \
    --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$TF_STATE_BUCKET" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
}

terraform_init() {
  local directory="$1"
  local state_name="$2"
  terraform -chdir="$directory" init -reconfigure \
    -backend-config="bucket=$TF_STATE_BUCKET" \
    -backend-config="key=quoptuna/$ENVIRONMENT/$state_name.tfstate" \
    -backend-config="region=$AWS_REGION" \
    -backend-config="encrypt=true" \
    -backend-config="use_lockfile=true"
}

foundation_output() {
  terraform -chdir="$FOUNDATION_DIR" output -raw "$1"
}

application_var_file() {
  local target="$1"
  jq -n \
    --arg region "$AWS_REGION" \
    --arg environment "$ENVIRONMENT" \
    --arg project "$PROJECT_NAME" \
    --arg domain "$DOMAIN_NAME" \
    --arg zone "$ROUTE53_ZONE_ID" \
    --arg instance "$INSTANCE_TYPE" \
    --argjson disk "$ROOT_VOLUME_SIZE" \
    --arg bucket "$(foundation_output artifact_bucket)" \
    --arg bucket_arn "$(foundation_output artifact_bucket_arn)" \
    --arg ecr_name "$(foundation_output ecr_repository_name)" \
    --arg ecr_url "$(foundation_output ecr_repository_url)" \
    --arg ecr_arn "$(foundation_output ecr_repository_arn)" \
    --arg secret_name "$(foundation_output runtime_secret_name)" \
    --arg secret_arn "$(foundation_output runtime_secret_arn)" \
    '{
      aws_region: $region, environment: $environment, project_name: $project,
      domain_name: $domain, route53_zone_id: $zone, instance_type: $instance,
      root_volume_size: $disk, artifact_bucket: $bucket,
      artifact_bucket_arn: $bucket_arn, ecr_repository_name: $ecr_name,
      ecr_repository_url: $ecr_url, ecr_repository_arn: $ecr_arn,
      runtime_secret_name: $secret_name, runtime_secret_arn: $secret_arn
    }' >"$target"
}

sync_runtime_secret() {
  local temp_dir secret_file secret_name bucket
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' RETURN
  secret_file="$temp_dir/runtime.json"
  secret_name="$(foundation_output runtime_secret_name)"
  bucket="$(foundation_output artifact_bucket)"
  python3 "$SCRIPT_DIR/envfile.py" secret --file "$ENV_FILE" \
    --environment "$ENVIRONMENT" --bucket "$bucket" --region "$AWS_REGION" \
    --domain "$DOMAIN_NAME" >"$secret_file"
  aws secretsmanager put-secret-value --secret-id "$secret_name" \
    --secret-string "file://$secret_file" >/dev/null
  log "Runtime secret updated"
}

instance_output() {
  terraform -chdir="$APPLICATION_DIR" output -raw "$1"
}

wait_for_ssm() {
  local instance_id="$1"
  local deadline=$((SECONDS + 600))
  log "Waiting for SSM..."
  while (( SECONDS < deadline )); do
    if [[ "$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=$instance_id" \
      --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)" == "Online" ]]; then
      return
    fi
    sleep 5
  done
  die "Instance did not become available in SSM"
}

run_ssm() {
  local instance_id="$1"
  local command="$2"
  local temp_dir input_file command_id status
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' RETURN
  input_file="$temp_dir/command.json"
  jq -n --arg instance "$instance_id" --arg command "$command" '{
    DocumentName: "AWS-RunShellScript",
    InstanceIds: [$instance],
    Parameters: {commands: [$command]},
    TimeoutSeconds: 900
  }' >"$input_file"
  command_id="$(aws ssm send-command --cli-input-json "file://$input_file" \
    --query 'Command.CommandId' --output text)"
  aws ssm wait command-executed --command-id "$command_id" --instance-id "$instance_id" || true
  status="$(aws ssm get-command-invocation --command-id "$command_id" \
    --instance-id "$instance_id" --query Status --output text)"
  aws ssm get-command-invocation --command-id "$command_id" \
    --instance-id "$instance_id" --query StandardOutputContent --output text
  if [[ "$status" != "Success" ]]; then
    aws ssm get-command-invocation --command-id "$command_id" \
      --instance-id "$instance_id" --query StandardErrorContent --output text >&2
    return 1
  fi
}

active_work_total() {
  local instance_id="$1"
  local state output
  state="$(aws ec2 describe-instances --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  if [[ "$state" != "running" ]]; then
    printf '0\n'
    return
  fi
  output="$(run_ssm "$instance_id" \
    "docker exec quoptuna /app/.venv/bin/quoptuna active-work" 2>/dev/null || printf '{"total":0}')"
  printf '%s' "$output" | jq -r '.total // 0' 2>/dev/null || printf '0\n'
}

assert_no_active_work() {
  local instance_id="$1"
  local total
  total="$(active_work_total "$instance_id")"
  if [[ "$total" -gt 0 && "$FORCE" != true ]]; then
    die "$total active job(s) found; wait for completion or explicitly use --force"
  fi
}

require_running_instance() {
  local instance_id="$1"
  local state
  state="$(aws ec2 describe-instances --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  [[ "$state" == "running" ]] || die "Instance is $state; run resume before deploying or updating"
}

delete_dns_record() {
  local record_file
  record_file="$(mktemp)"
  if aws route53 list-resource-record-sets --hosted-zone-id "$ROUTE53_ZONE_ID" \
    --query "ResourceRecordSets[?Name == '$DOMAIN_NAME.' && Type == 'A'] | [0]" \
    --output json >"$record_file" &&
    [[ "$(jq -r '.Type // empty' "$record_file")" == "A" ]]; then
    jq -n --slurpfile record "$record_file" \
      '{Changes:[{Action:"DELETE",ResourceRecordSet:$record[0]}]}' \
      >"$record_file.change"
    aws route53 change-resource-record-sets --hosted-zone-id "$ROUTE53_ZONE_ID" \
      --change-batch "file://$record_file.change" >/dev/null
  fi
  rm -f "$record_file" "$record_file.change"
}

wait_for_https() {
  local deadline=$((SECONDS + 600))
  log "Waiting for https://$DOMAIN_NAME/api/v1/health"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 10 "https://$DOMAIN_NAME/api/v1/health" >/dev/null 2>&1; then
      log "Application is healthy at https://$DOMAIN_NAME"
      return
    fi
    sleep 10
  done
  die "HTTPS health check timed out"
}
