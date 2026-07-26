#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:?missing action}"
shift

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
parse_args "$@"
load_deployment_env

apply_foundation() {
  local tf_args=(
    "-var=aws_region=$AWS_REGION"
    "-var=environment=$ENVIRONMENT"
    "-var=project_name=$PROJECT_NAME"
  )
  terraform_init "$FOUNDATION_DIR" foundation
  if [[ "$PLAN_ONLY" == true ]]; then
    terraform -chdir="$FOUNDATION_DIR" plan "${tf_args[@]}"
  else
    terraform -chdir="$FOUNDATION_DIR" apply -auto-approve "${tf_args[@]}"
  fi
}

apply_application() {
  local temp_dir var_file
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' RETURN
  var_file="$temp_dir/application.tfvars.json"
  application_var_file "$var_file"
  terraform_init "$APPLICATION_DIR" application
  if [[ "$PLAN_ONLY" == true ]]; then
    terraform -chdir="$APPLICATION_DIR" plan -var-file="$var_file"
  else
    terraform -chdir="$APPLICATION_DIR" apply -auto-approve -var-file="$var_file"
  fi
}

BUILT_IMAGE_URI=""

build_and_push() {
  local repository tag registry password
  require_command docker
  repository="$(foundation_output ecr_repository_url)"
  registry="${repository%%/*}"
  tag="$(git -C "$REPO_DIR" rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)"
  log "Building $repository:$tag"
  password="$(aws ecr get-login-password --region "$AWS_REGION")"
  docker login --username AWS --password-stdin "$registry" <<<"$password" >/dev/null
  docker buildx build --platform linux/amd64 --load \
    -f "$REPO_DIR/Dockerfile.production" -t "$repository:$tag" "$REPO_DIR"
  docker push "$repository:$tag"
  BUILT_IMAGE_URI="$repository:$tag"
  log "Published $BUILT_IMAGE_URI"
}

deploy_image() {
  local image_uri="$1"
  local instance_id bucket secret_name repository command
  instance_id="$(instance_output instance_id)"
  bucket="$(foundation_output artifact_bucket)"
  secret_name="$(foundation_output runtime_secret_name)"
  repository="$(foundation_output ecr_repository_url)"
  aws s3 cp "$RUNTIME_DIR/docker-compose.prod.yml" \
    "s3://$bucket/deployment/$ENVIRONMENT/docker-compose.yml" >/dev/null
  aws s3 cp "$RUNTIME_DIR/Caddyfile" \
    "s3://$bucket/deployment/$ENVIRONMENT/Caddyfile" >/dev/null
  wait_for_ssm "$instance_id"
  command="set -euo pipefail
mkdir -p /opt/quoptuna
aws s3 cp s3://$bucket/deployment/$ENVIRONMENT/docker-compose.yml /opt/quoptuna/docker-compose.yml
aws s3 cp s3://$bucket/deployment/$ENVIRONMENT/Caddyfile /opt/quoptuna/Caddyfile
aws secretsmanager get-secret-value --secret-id '$secret_name' --query SecretString --output text > /opt/quoptuna/runtime.json
jq -r 'to_entries[] | \"\\(.key)=\\(.value|tostring)\"' /opt/quoptuna/runtime.json > /opt/quoptuna/runtime.env
chmod 600 /opt/quoptuna/runtime.json /opt/quoptuna/runtime.env
printf 'IMAGE_URI=%s\\nDOMAIN_NAME=%s\\n' '$image_uri' '$DOMAIN_NAME' > /opt/quoptuna/compose.env
aws ecr get-login-password --region '$AWS_REGION' | docker login --username AWS --password-stdin '${repository%%/*}'
cd /opt/quoptuna
docker compose --env-file compose.env -f docker-compose.yml pull
docker compose --env-file compose.env -f docker-compose.yml up -d --remove-orphans
docker image prune -f"
  run_ssm "$instance_id" "$command"
  wait_for_https
}

create_infrastructure() {
  local image_uri
  check_tools
  require_command git
  bootstrap_state
  apply_foundation
  [[ "$PLAN_ONLY" == true ]] && return
  sync_runtime_secret
  build_and_push
  image_uri="$BUILT_IMAGE_URI"
  apply_application
  deploy_image "$image_uri"
}

deploy_application() {
  local instance_id image_uri
  check_tools
  require_command git
  terraform_init "$FOUNDATION_DIR" foundation
  terraform_init "$APPLICATION_DIR" application
  instance_id="$(instance_output instance_id)"
  require_running_instance "$instance_id"
  assert_no_active_work "$instance_id"
  sync_runtime_secret
  build_and_push
  image_uri="$BUILT_IMAGE_URI"
  deploy_image "$image_uri"
}

update_infrastructure() {
  local image_uri
  check_tools
  require_command git
  bootstrap_state
  terraform_init "$APPLICATION_DIR" application
  if terraform -chdir="$APPLICATION_DIR" output -raw instance_id >/dev/null 2>&1; then
    local existing_instance
    existing_instance="$(instance_output instance_id)"
    require_running_instance "$existing_instance"
    assert_no_active_work "$existing_instance"
  fi
  apply_foundation
  [[ "$PLAN_ONLY" == true ]] && { apply_application; return; }
  sync_runtime_secret
  build_and_push
  image_uri="$BUILT_IMAGE_URI"
  apply_application
  deploy_image "$image_uri"
}

pause_infrastructure() {
  local instance_id state
  check_tools
  terraform_init "$APPLICATION_DIR" application
  instance_id="$(instance_output instance_id)"
  assert_no_active_work "$instance_id"
  state="$(aws ec2 describe-instances --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  [[ "$state" == "stopped" ]] && { log "Instance is already stopped"; return; }
  delete_dns_record
  aws ec2 stop-instances --instance-ids "$instance_id" >/dev/null
  aws ec2 wait instance-stopped --instance-ids "$instance_id"
  log "Paused $ENVIRONMENT; persistent Supabase and S3 data were preserved"
}

resume_infrastructure() {
  local instance_id
  check_tools
  terraform_init "$APPLICATION_DIR" application
  instance_id="$(instance_output instance_id)"
  aws ec2 start-instances --instance-ids "$instance_id" >/dev/null
  aws ec2 wait instance-running --instance-ids "$instance_id"
  wait_for_ssm "$instance_id"
  run_ssm "$instance_id" \
    "/usr/local/bin/quoptuna-refresh-dns && systemctl restart quoptuna-start.service"
  wait_for_https
}

status_infrastructure() {
  local instance_id state public_ip health active image
  check_tools
  if ! aws s3api head-bucket --bucket "$TF_STATE_BUCKET" >/dev/null 2>&1; then
    if [[ "$JSON_OUTPUT" == true ]]; then
      jq -n --arg environment "$ENVIRONMENT" --arg url "https://$DOMAIN_NAME" \
        '{environment:$environment,state:"not_created",health:"unavailable",
          url:$url,active_work:0}'
    else
      printf 'Environment: %s\nState: not_created\nURL: https://%s\n' \
        "$ENVIRONMENT" "$DOMAIN_NAME"
    fi
    return
  fi
  terraform_init "$APPLICATION_DIR" application >/dev/null
  if ! instance_id="$(terraform -chdir="$APPLICATION_DIR" output -raw instance_id 2>/dev/null)"; then
    if [[ "$JSON_OUTPUT" == true ]]; then
      jq -n --arg environment "$ENVIRONMENT" --arg url "https://$DOMAIN_NAME" \
        '{environment:$environment,state:"not_created",health:"unavailable",
          url:$url,active_work:0}'
    else
      printf 'Environment: %s\nState: not_created\nURL: https://%s\n' \
        "$ENVIRONMENT" "$DOMAIN_NAME"
    fi
    return
  fi
  state="$(aws ec2 describe-instances --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)"
  public_ip="$(aws ec2 describe-instances --instance-ids "$instance_id" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
  health="unavailable"
  active=0
  image=""
  if [[ "$state" == "running" ]]; then
    curl -fsS --max-time 5 "https://$DOMAIN_NAME/api/v1/health" >/dev/null 2>&1 && health="healthy"
    active="$(active_work_total "$instance_id")"
    image="$(run_ssm "$instance_id" \
      "docker inspect quoptuna --format '{{.Config.Image}}'" 2>/dev/null || true)"
  fi
  if [[ "$JSON_OUTPUT" == true ]]; then
    jq -n --arg environment "$ENVIRONMENT" --arg instance_id "$instance_id" \
      --arg state "$state" --arg public_ip "$public_ip" \
      --arg url "https://$DOMAIN_NAME" --arg health "$health" \
      --arg image "$image" --argjson active "$active" \
      '{environment:$environment,instance_id:$instance_id,state:$state,
        public_ip:$public_ip,url:$url,health:$health,active_work:$active,image:$image}'
  else
    printf 'Environment: %s\nInstance: %s\nState: %s\nURL: https://%s\nHealth: %s\nActive work: %s\nImage: %s\n' \
      "$ENVIRONMENT" "$instance_id" "$state" "$DOMAIN_NAME" "$health" "$active" "$image"
  fi
}

empty_versioned_bucket() {
  local bucket="$1"
  local response objects delete_file
  delete_file="$(mktemp)"
  trap 'rm -f "$delete_file"' RETURN
  while true; do
    response="$(aws s3api list-object-versions --bucket "$bucket" --output json)"
    objects="$(printf '%s' "$response" | jq '[
      (.Versions // [])[], (.DeleteMarkers // [])[]
    ] | map({Key, VersionId})')"
    [[ "$(printf '%s' "$objects" | jq 'length')" -eq 0 ]] && break
    jq -n --argjson objects "$objects" '{Objects:$objects,Quiet:true}' >"$delete_file"
    aws s3api delete-objects --bucket "$bucket" --delete "file://$delete_file" >/dev/null
  done
}

destroy_infrastructure() {
  local instance_id temp_dir var_file bucket
  local tf_args=(
    "-var=aws_region=$AWS_REGION"
    "-var=environment=$ENVIRONMENT"
    "-var=project_name=$PROJECT_NAME"
  )
  check_tools
  terraform_init "$FOUNDATION_DIR" foundation
  terraform_init "$APPLICATION_DIR" application
  instance_id="$(instance_output instance_id)"
  assert_no_active_work "$instance_id"
  if [[ "$CONFIRM_DESTROY" != true ]]; then
    printf 'Type %s to destroy its compute infrastructure: ' "$ENVIRONMENT" >&2
    read -r confirmation
    [[ "$confirmation" == "$ENVIRONMENT" ]] || die "Confirmation did not match"
  fi
  delete_dns_record
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' RETURN
  var_file="$temp_dir/application.tfvars.json"
  application_var_file "$var_file"
  terraform -chdir="$APPLICATION_DIR" destroy -auto-approve -var-file="$var_file"
  [[ "$DELETE_DATA" != true ]] && {
    log "Compute deleted; Supabase, S3, ECR, secret, and state were preserved"
    return
  }
  printf 'Type DELETE-%s to delete persistent AWS data: ' "$ENVIRONMENT" >&2
  read -r confirmation
  [[ "$confirmation" == "DELETE-$ENVIRONMENT" ]] || die "Confirmation did not match"
  bucket="$(foundation_output artifact_bucket)"
  empty_versioned_bucket "$bucket"
  terraform -chdir="$FOUNDATION_DIR" destroy -auto-approve "${tf_args[@]}"
  log "All environment AWS resources deleted; Supabase and Terraform state remain"
}

case "$ACTION" in
  create) create_infrastructure ;;
  deploy) deploy_application ;;
  update) update_infrastructure ;;
  pause) pause_infrastructure ;;
  resume) resume_infrastructure ;;
  status) status_infrastructure ;;
  destroy) destroy_infrastructure ;;
  *) die "Unsupported action: $ACTION" ;;
esac
