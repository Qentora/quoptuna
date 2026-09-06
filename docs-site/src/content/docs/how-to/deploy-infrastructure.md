---
title: Deploy the AWS infrastructure
description: Prerequisites, the quoptuna infra console, and the create/deploy/pause/resume/destroy operations.
---

QuOptuna deploys as a single stoppable EC2 instance running the application
container behind Caddy. Supabase stores application and Optuna data, and S3
stores datasets and analysis artifacts. There is no Kubernetes cluster, load
balancer, NAT Gateway, RDS instance, or open SSH port.

Terraform lives in `infra/terraform/` (a `foundation` stack for persistent
resources and an `application` stack for compute), and every operation is driven
by the scripts in `infra/scripts/`.

## Prerequisites

### Tooling

| Requirement | Notes |
| --- | --- |
| Terraform | `>= 1.10.0, < 2.0.0` (pinned in `versions.tf`) |
| AWS CLI v2 | Authenticated; `aws sts get-caller-identity` must succeed |
| Docker with Buildx | Used to build and push the immutable application image |
| `jq` | Required by the operation scripts |
| `git` | Used to stamp the deployed image |
| `curl`, Python 3 | Health checks and deployment-file parsing |

The scripts check for `aws`, `terraform`, `jq`, `docker`, and `git`, and fail
with `Required command not found` if any is missing.

### Accounts and resources

- **AWS account** with permissions for EC2, S3, ECR, Route 53, Secrets Manager,
  IAM, and SSM.
- **Supabase PostgreSQL URL**. The EC2 network is dual-stack, so Supabase's
  IPv6-only direct endpoint works. An IPv4-compatible session-pooler URL with
  `sslmode=require` is also supported.
- **A domain** registered anywhere and delegated to an existing Route 53 hosted
  zone. You need the zone ID.
- **An Auth0 application** (see [Auth0 setup](#auth0-setup) below).
- **A globally unique S3 bucket name** for Terraform state. The scripts create
  the bucket if it does not exist, with versioning, encryption, and public-access
  blocking enabled.

### The deployment file

Copy the template and fill it in:

```bash
cp .env.deploy.example .env.deploy
aws sts get-caller-identity
```

`infra/scripts/envfile.py` parses this file as a conservative `KEY=VALUE` subset
— it never evaluates shell syntax. Matching variables already present in your
process environment take precedence over the file.

**Deployment keys** (control where and how infrastructure is built):

| Variable | Example | Purpose |
| --- | --- | --- |
| `AWS_PROFILE` | `default` | Profile for the AWS credential chain |
| `AWS_REGION` | `us-east-2` | Target region |
| `TF_STATE_BUCKET` | — | Globally unique Terraform state bucket |
| `PROJECT_NAME` | `quoptuna` | Resource name prefix |
| `DOMAIN_NAME` | `quoptuna.example.com` | Public hostname served over HTTPS |
| `ROUTE53_ZONE_ID` | `Z0000...` | Hosted zone for the domain |
| `INSTANCE_TYPE` | `t3.large` | EC2 instance size |
| `ROOT_VOLUME_SIZE` | `50` | Root EBS volume in GB |

**Runtime keys** (written into the AWS Secrets Manager runtime secret):

`DATABASE_URL`, `OPTUNA_DATABASE_URL`, `OPTUNA_DB_SCHEMA`, `AUTH0_DOMAIN`,
`AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`, `AUTH0_SECRET`, `AUTH_ALLOWED_EMAILS`,
`AUTH_REQUIRE_VERIFIED_EMAIL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`.

See the [Configuration reference](/reference/configuration/) for what each
runtime variable does.

:::note
The deployment derives `APP_ENV=production`, `APP_BASE_URL`, `CORS_ORIGINS`,
`ARTIFACT_STORAGE=s3`, and the `S3_*` settings automatically from your domain,
bucket, and region. Do not set them in `.env.deploy`.
:::

### Auth0 setup

In the Auth0 application, set:

- Allowed Callback URL: `https://YOUR_DOMAIN/auth/callback`
- Allowed Logout URL: `https://YOUR_DOMAIN`
- Allowed Web Origin: `https://YOUR_DOMAIN`

Then open **Actions → Library → Build Custom**, create a Post-Login Action, and
paste `infra/auth0/approved-emails.js`. Add these Action secrets:

- `QUOPTUNA_CLIENT_ID` — the QuOptuna Auth0 client ID
- `ALLOWED_EMAILS` — the same comma-separated list as `AUTH_ALLOWED_EMAILS`

Deploy the Action and add it to **Actions → Flows → Login**. QuOptuna repeats the
same allowlist and verified-email checks in the API, so removing the Action does
not open application access.

## Run from the console

```bash
uv sync
uv run quoptuna infra --environment dev --env-file .env.deploy
```

This opens a [Textual](https://textual.textualize.io/) console with an action
sidebar, a live status panel, and a streaming log. Secrets matching
`password`, `secret`, `token`, `database_url`, `access_key`, or `private_key`
are redacted from the log output.

| Option | Default | Purpose |
| --- | --- | --- |
| `--environment`, `-e` | `dev` | Target environment (`dev` or `production`) |
| `--env-file` | none | Deployment `.env` file; falls back to the process environment |
| `--terraform-dir` | `infra` | Directory holding `scripts/` and `terraform/` |

Key bindings: `r` refreshes status, `q` quits. Only one operation runs at a
time. **Pause** asks for confirmation; **Destroy** requires you to type the
environment name exactly.

## Operations

| Action | What it does |
| --- | --- |
| **Create** | State bootstrap, persistent resources, image build, EC2, DNS, and HTTPS |
| **Deploy** | Build and deploy a new immutable image |
| **Update** | Apply Terraform changes and deploy the new image |
| **Pause** | Refuse while work is active, remove DNS, and stop EC2 |
| **Resume** | Start EC2, restore DNS, and wait for HTTPS |
| **Status** | Report EC2 state, app health, image, and active work |
| **Destroy** | Delete compute/network resources, preserving Supabase and AWS data |

The same operations run directly as scripts:

```bash
infra/scripts/create.sh dev --env-file .env.deploy
infra/scripts/status.sh dev --env-file .env.deploy --json
infra/scripts/pause.sh dev --env-file .env.deploy
infra/scripts/resume.sh dev --env-file .env.deploy
infra/scripts/deploy.sh dev --env-file .env.deploy
infra/scripts/update.sh dev --env-file .env.deploy
infra/scripts/destroy.sh dev --env-file .env.deploy
```

:::caution
The environment argument accepts only `dev` or `production`. Any other value
fails with `Unsupported environment`.
:::

### Script options

| Flag | Applies to | Purpose |
| --- | --- | --- |
| `--env-file PATH` | all | Deployment file to read |
| `--json` | `status` | Emit machine-readable status (used by the console) |
| `--plan-only` | `create`, `update` | Show the Terraform plan without applying |
| `--force` | `pause`, `destroy` | Proceed even though work is active |
| `--confirm-destroy` | `destroy` | Skip the interactive destroy confirmation |
| `--delete-data` | `destroy` | Also delete the persistent AWS foundation |

Use `--force` only when you intentionally accept interrupting active trials.

To delete the persistent AWS foundation too:

```bash
infra/scripts/destroy.sh dev --env-file .env.deploy --delete-data
```

This requires two typed confirmations. It deletes the artifact bucket, images,
and runtime secret. It never deletes Supabase or the Terraform-state bucket.

## Check deployment health

Two CLI commands report on a running deployment, both emitting JSON:

```bash
quoptuna active-work        # active optimization and analysis counts
quoptuna deployment-check   # readiness checks; exits 1 when unhealthy
```

`pause` calls `active-work` on the instance over SSM to refuse stopping EC2
while trials are still running. See the [CLI reference](/reference/cli/).

## Cost controls

- Pause EC2 whenever trials are not running.
- The default `t3.large` uses standard CPU credits, preventing unlimited-credit
  charges.
- No Elastic IP is retained while paused; DNS is restored to the new address on
  resume.
- ECR keeps only five images.
- Old S3 artifacts transition to Glacier Instant Retrieval.
- Container logs rotate locally; CloudWatch log ingestion is not enabled.
- Increase `INSTANCE_TYPE` only for trials that need more CPU or memory.

## Troubleshoot

If a deployment fails, inspect status and use SSM without opening SSH:

```bash
aws ssm start-session --target INSTANCE_ID
```

| Message | Cause |
| --- | --- |
| `Required command not found: X` | Install the missing tool from [Tooling](#tooling) |
| `AWS credentials are unavailable` | `aws sts get-caller-identity` fails; check `AWS_PROFILE` |
| `Set X in ...` | A required deployment or runtime key is missing |
| `Script is missing or not executable` | Run `chmod +x infra/scripts/*.sh` |
| `Environment file not found` | The `--env-file` path is wrong |

## See also

- [Configuration reference](/reference/configuration/)
- [CLI reference](/reference/cli/)
- [Move persistence to Supabase and S3](/how-to/migrate-to-supabase/)
