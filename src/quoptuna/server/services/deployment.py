"""Deployment readiness checks used by AWS lifecycle automation."""

from __future__ import annotations

from typing import Any

import boto3
from sqlalchemy import text
from sqlmodel import col, select

from quoptuna.server.core.config import settings
from quoptuna.server.services.database import get_engine, session_scope
from quoptuna.server.services.models import AnalysisJob
from quoptuna.server.services.run_store import count_active_runs


def active_work() -> dict[str, int]:
    """Count work that would be interrupted by a restart or instance stop."""
    with session_scope() as session:
        analyses = len(
            session.exec(
                select(AnalysisJob).where(
                    col(AnalysisJob.status).in_(["pending", "running"])
                )
            ).all()
        )
    runs = count_active_runs()
    return {"runs": runs, "analyses": analyses, "total": runs + analyses}


def deployment_check() -> dict[str, Any]:
    """Check database, artifact storage, and active work without exposing secrets."""
    checks: dict[str, Any] = {}
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": type(exc).__name__}

    if settings.ARTIFACT_STORAGE.lower() == "s3":
        try:
            boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                region_name=settings.S3_REGION or None,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID or None,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY or None,
            ).head_bucket(Bucket=settings.S3_BUCKET)
            checks["artifacts"] = {"ok": True, "backend": "s3"}
        except Exception as exc:
            checks["artifacts"] = {
                "ok": False,
                "backend": "s3",
                "error": type(exc).__name__,
            }
    else:
        checks["artifacts"] = {"ok": True, "backend": "local"}

    checks["active_work"] = active_work()
    return {"ok": all(item.get("ok", True) for item in checks.values()), "checks": checks}
