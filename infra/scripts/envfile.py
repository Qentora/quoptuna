#!/usr/bin/env python3
"""Read deployment dotenv files without evaluating shell syntax."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

QUOTED_VALUE_LENGTH = 2

DEPLOYMENT_KEYS = {
    "AWS_PROFILE",
    "AWS_REGION",
    "TF_STATE_BUCKET",
    "PROJECT_NAME",
    "DOMAIN_NAME",
    "ROUTE53_ZONE_ID",
    "INSTANCE_TYPE",
    "ROOT_VOLUME_SIZE",
}

RUNTIME_KEYS = {
    "DATABASE_URL",
    "OPTUNA_DATABASE_URL",
    "OPTUNA_DB_SCHEMA",
    "AUTH0_DOMAIN",
    "AUTH0_CLIENT_ID",
    "AUTH0_CLIENT_SECRET",
    "AUTH0_SECRET",
    "AUTH_ALLOWED_EMAILS",
    "AUTH_REQUIRE_VERIFIED_EMAIL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
}


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse the conservative KEY=VALUE subset used by deployment files."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            message = f"{path}:{number}: expected KEY=VALUE"
            raise ValueError(message)
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in DEPLOYMENT_KEYS | RUNTIME_KEYS:
            continue
        value = value.strip()
        if len(value) >= QUOTED_VALUE_LENGTH and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def merged_values(path: Path) -> dict[str, str]:
    values = read_dotenv(path)
    for key in DEPLOYMENT_KEYS | RUNTIME_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def export_values(values: dict[str, str]) -> None:
    for key in sorted(DEPLOYMENT_KEYS):
        if key in values:
            sys.stdout.write(f"export {key}={shlex.quote(values[key])}\n")


def runtime_secret(
    values: dict[str, str], *, environment: str, bucket: str, region: str, domain: str
) -> dict[str, str]:
    result = {key: values.get(key, "") for key in sorted(RUNTIME_KEYS)}
    result.update(
        {
            "APP_ENV": "production",
            "APP_BASE_URL": f"https://{domain}",
            "CORS_ORIGINS": f"https://{domain}",
            "ARTIFACT_STORAGE": "s3",
            "ARTIFACT_ROOT": "db/analysis",
            "S3_BUCKET": bucket,
            "S3_REGION": region,
            "S3_ENDPOINT_URL": "",
            "S3_ACCESS_KEY_ID": "",
            "S3_SECRET_ACCESS_KEY": "",
            "S3_PREFIX": f"quoptuna/{environment}",
            "S3_SIGNED_URL_TTL": "900",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "secret"])
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--environment")
    parser.add_argument("--bucket")
    parser.add_argument("--region")
    parser.add_argument("--domain")
    args = parser.parse_args()
    values = merged_values(args.file)
    if args.command == "export":
        export_values(values)
        return
    if not all((args.environment, args.bucket, args.region, args.domain)):
        parser.error("secret requires --environment, --bucket, --region, and --domain")
    payload = json.dumps(
        runtime_secret(
            values,
            environment=args.environment,
            bucket=args.bucket,
            region=args.region,
            domain=args.domain,
        )
    )
    sys.stdout.write(f"{payload}\n")


if __name__ == "__main__":
    main()
