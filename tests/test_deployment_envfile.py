"""Deployment dotenv parsing and secret generation tests."""

import importlib.util
from pathlib import Path


def load_envfile_module():
    path = Path(__file__).parents[1] / "infra" / "scripts" / "envfile.py"
    spec = importlib.util.spec_from_file_location("deployment_envfile", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dotenv_parser_ignores_unknown_keys(tmp_path: Path):
    module = load_envfile_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_REGION=us-east-2\nUNKNOWN=value\nDOMAIN_NAME='app.example.com'\n",
        encoding="utf-8",
    )
    assert module.read_dotenv(env_file) == {
        "AWS_REGION": "us-east-2",
        "DOMAIN_NAME": "app.example.com",
    }


def test_runtime_secret_uses_iam_for_s3(tmp_path: Path):
    module = load_envfile_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql://example\nAUTH_ALLOWED_EMAILS=user@example.com\n",
        encoding="utf-8",
    )
    values = module.read_dotenv(env_file)
    secret = module.runtime_secret(
        values,
        environment="dev",
        bucket="artifact-bucket",
        region="us-east-2",
        domain="app.example.com",
    )
    assert secret["ARTIFACT_STORAGE"] == "s3"
    assert secret["S3_BUCKET"] == "artifact-bucket"
    assert secret["S3_ACCESS_KEY_ID"] == ""
    assert secret["APP_BASE_URL"] == "https://app.example.com"
