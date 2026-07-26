# QuOptuna AWS deployment

This deployment runs QuOptuna and Caddy on one stoppable EC2 instance. Supabase
stores application and Optuna data; S3 stores datasets and analysis artifacts.
There is no Kubernetes cluster, load balancer, NAT Gateway, RDS instance, or SSH
port.

## Prerequisites

- Terraform 1.10 or newer
- AWS CLI v2 authenticated with an AWS profile
- Docker with Buildx
- `jq`, `curl`, Git, and Python 3
- A Supabase PostgreSQL URL
- A domain registered anywhere and delegated to an existing Route 53 hosted zone

Copy the deployment template and fill it in:

```bash
cp .env.deploy.example .env.deploy
aws sts get-caller-identity
```

Use a globally unique `TF_STATE_BUCKET`. For Supabase, use an IPv4-compatible
pooler URL with `sslmode=require` if the direct endpoint is IPv6-only.

## Auth0 approved emails

In the Auth0 application, set:

- Allowed Callback URL: `https://YOUR_DOMAIN/auth/callback`
- Allowed Logout URL: `https://YOUR_DOMAIN`
- Allowed Web Origin: `https://YOUR_DOMAIN`

Then open **Actions → Library → Build Custom**, create a Post-Login Action, and
paste `infra/auth0/approved-emails.js`. Add these Action secrets:

- `QUOPTUNA_CLIENT_ID`: the QuOptuna Auth0 client ID
- `ALLOWED_EMAILS`: the same comma-separated list as `AUTH_ALLOWED_EMAILS`

Deploy the Action and add it to **Actions → Flows → Login**. QuOptuna repeats the
same allowlist and verified-email checks in the API, so removing the Action does
not open application access.

## Run from the TUI

```bash
uv sync
uv run quoptuna infra --environment dev --env-file .env.deploy
```

The actions mean:

- **Create**: state bootstrap, persistent resources, image build, EC2, DNS, and HTTPS.
- **Deploy**: build and deploy a new immutable image.
- **Update**: apply Terraform changes and deploy the new image.
- **Pause**: refuse while work is active, remove DNS, and stop EC2.
- **Resume**: start EC2, restore DNS, and wait for HTTPS.
- **Status**: report EC2, app health, image, and active work.
- **Destroy**: delete compute/network resources while preserving Supabase and AWS data.

The same operations can be run directly:

```bash
infra/scripts/create.sh dev --env-file .env.deploy
infra/scripts/status.sh dev --env-file .env.deploy --json
infra/scripts/pause.sh dev --env-file .env.deploy
infra/scripts/resume.sh dev --env-file .env.deploy
infra/scripts/deploy.sh dev --env-file .env.deploy
infra/scripts/update.sh dev --env-file .env.deploy
infra/scripts/destroy.sh dev --env-file .env.deploy
```

Use `--plan-only` with `create` or `update` to inspect Terraform changes. Use
`--force` only when you intentionally accept interrupting active work.

To delete the persistent AWS foundation too:

```bash
infra/scripts/destroy.sh dev --env-file .env.deploy --delete-data
```

This requires two typed confirmations. It deletes the artifact bucket, images,
and runtime secret. It never deletes Supabase or the Terraform-state bucket.

## Cost controls

- Pause EC2 whenever trials are not running.
- The default `t3.large` uses standard CPU credits, preventing unlimited-credit charges.
- No Elastic IP is retained while paused; DNS is restored to the new address on resume.
- ECR keeps only five images.
- Old S3 artifacts transition to Glacier Instant Retrieval.
- Container logs rotate locally; CloudWatch log ingestion is not enabled.
- Increase `INSTANCE_TYPE` only for trials that need more CPU or memory.

If a deployment fails, inspect status and use SSM without opening SSH:

```bash
aws ssm start-session --target INSTANCE_ID
```
