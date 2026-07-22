"""Durable analysis snapshots, reports, and local/S3 artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import select

from quoptuna.server.core.config import settings
from quoptuna.server.services.database import session_scope
from quoptuna.server.services.models import (
    AnalysisArtifact,
    AnalysisJob,
    AnalysisReport,
    AnalysisSnapshot,
)
from quoptuna.server.services import run_store

ARTIFACT_ROOT = Path(settings.ARTIFACT_ROOT)


def _now() -> str:
    return datetime.now().isoformat()


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_number": config.get("trial_number"),
        "use_proba": bool(config.get("use_proba", True)),
        "subset_size": int(config.get("subset_size", 50)),
        "class_index": int(config.get("class_index", 0)),
        "sample_index": int(config.get("sample_index", 0)),
    }


def _config_key(config: dict[str, Any]) -> str:
    encoded = json.dumps(normalize_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def create_job(optimization_id: str, config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_config(config)
    key = _config_key(config)
    now = _now()
    with session_scope() as session:
        snapshot = session.exec(
            select(AnalysisSnapshot).where(
                AnalysisSnapshot.optimization_id == optimization_id,
                AnalysisSnapshot.config_key == key,
            )
        ).first()
        if snapshot is None:
            snapshot = AnalysisSnapshot(
                id=str(uuid.uuid4()), optimization_id=optimization_id,
                config_key=key, config_json=json.dumps(config), created_at=now,
            )
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
        active = session.exec(
            select(AnalysisJob).where(
                AnalysisJob.snapshot_id == snapshot.id,
                AnalysisJob.status.in_(["pending", "running"]),
            ).order_by(AnalysisJob.created_at.desc())
        ).first()
        if active:
            return {"id": active.id, "snapshot_id": snapshot.id, "status": active.status, "created": False}
        job = AnalysisJob(id=str(uuid.uuid4()), snapshot_id=snapshot.id, status="pending", created_at=now)
        session.add(job)
        session.commit()
        return {"id": job.id, "snapshot_id": snapshot.id, "status": job.status, "created": True}


def create_revision_job(snapshot_id: str) -> dict[str, Any]:
    with session_scope() as session:
        snapshot = session.get(AnalysisSnapshot, snapshot_id)
        if not snapshot or snapshot.revision <= 0:
            raise KeyError(snapshot_id)
        job = AnalysisJob(id=str(uuid.uuid4()), snapshot_id=snapshot_id, status="running", created_at=_now())
        session.add(job)
        session.commit()
        return {"id": job.id, "snapshot_id": snapshot_id, "status": "running"}


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"status", "current_section", "error", "completed_at"}
    if set(fields) - allowed:
        raise ValueError("Invalid analysis job field")
    with session_scope() as session:
        job = session.get(AnalysisJob, job_id)
        if job:
            for key, value in fields.items():
                setattr(job, key, value)
            session.add(job)
            session.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        job = session.get(AnalysisJob, job_id)
        if not job:
            return None
        snapshot = session.get(AnalysisSnapshot, job.snapshot_id)
        result = job.model_dump()
        result.update({"optimization_id": snapshot.optimization_id, "revision": snapshot.revision,
                       "config": json.loads(snapshot.config_json)})
        return result


def _extract_artifacts(value: Any, directory: Path, prefix: str = "payload") -> Any:
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        header, encoded = value.split(",", 1)
        mime = header[5:].split(";", 1)[0]
        extension = {"image/png": ".png", "image/jpeg": ".jpg"}.get(mime, ".bin")
        filename = f"{hashlib.sha256(prefix.encode()).hexdigest()}{extension}"
        (directory / filename).write_bytes(base64.b64decode(encoded))
        return {"$artifact": filename, "mime_type": mime}
    if isinstance(value, dict):
        return {key: _extract_artifacts(item, directory, f"{prefix}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_extract_artifacts(item, directory, f"{prefix}.{i}") for i, item in enumerate(value)]
    return value


class S3ArtifactStore:
    def __init__(self):
        import boto3
        self.client = boto3.client("s3", endpoint_url=settings.S3_ENDPOINT_URL or None,
                                   region_name=settings.S3_REGION or None,
                                   aws_access_key_id=settings.S3_ACCESS_KEY_ID or None,
                                   aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY or None)
        self.bucket = settings.S3_BUCKET

    def publish_tree(self, payload: Any, directory: Path, prefix: str, run_id: str, snapshot_id: str, revision: int) -> Any:
        def replace(value: Any) -> Any:
            if isinstance(value, dict) and "$artifact" in value:
                filename = value["$artifact"]
                key = f"{settings.S3_PREFIX.strip('/')}/{prefix}/{filename}"
                path = directory / filename
                self.client.upload_file(str(path), self.bucket, key,
                                        ExtraArgs={"ContentType": value.get("mime_type", "application/octet-stream")})
                with session_scope() as session:
                    session.add(AnalysisArtifact(id=str(uuid.uuid4()), optimization_id=run_id,
                        snapshot_id=snapshot_id, revision=revision, filename=filename,
                        object_key=key, storage_backend="s3", mime_type=value.get("mime_type"),
                        size_bytes=path.stat().st_size, checksum=hashlib.sha256(path.read_bytes()).hexdigest()))
                    session.commit()
                return {**value, "$object_key": key}
            if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
            if isinstance(value, list): return [replace(v) for v in value]
            return value
        return replace(payload)

    def get_url(self, key: str) -> str:
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=settings.S3_SIGNED_URL_TTL)

    def upload_file(self, path: Path, key: str, mime_type: str = "application/octet-stream") -> str:
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": mime_type})
        return key


def get_artifact_store() -> S3ArtifactStore | None:
    if settings.ARTIFACT_STORAGE.lower() != "s3":
        return None
    if not settings.S3_BUCKET:
        raise RuntimeError("S3_BUCKET is required when ARTIFACT_STORAGE=s3")
    try:
        return S3ArtifactStore()
    except ImportError as exc:
        raise RuntimeError("Install boto3 to use S3 artifact storage") from exc


def complete_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise KeyError(job_id)
    revision = int(job["revision"]) + 1
    final_dir = ARTIFACT_ROOT / job["optimization_id"] / job["snapshot_id"] / str(revision)
    temp_dir = final_dir.with_name(f".{revision}-{job_id}.tmp")
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        stored_payload = _extract_artifacts(payload, temp_dir)
        backend = get_artifact_store()
        prefix = f"runs/{job['optimization_id']}/analysis/{job['snapshot_id']}/revisions/{revision}"
        if backend:
            stored_payload = backend.publish_tree(stored_payload, temp_dir, prefix, job["optimization_id"], job["snapshot_id"], revision)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(final_dir)
        with session_scope() as session:
            snapshot = session.get(AnalysisSnapshot, job["snapshot_id"])
            snapshot.revision = revision
            snapshot.payload_json = json.dumps(stored_payload)
            snapshot.artifact_dir = str(final_dir)
            snapshot.storage_backend = "s3" if backend else "local"
            snapshot.artifact_prefix = prefix if backend else None
            snapshot.completed_at = _now()
            analysis_job = session.get(AnalysisJob, job_id)
            analysis_job.status = "completed"
            analysis_job.current_section = "complete"
            analysis_job.completed_at = _now()
            session.add(snapshot)
            session.add(analysis_job)
            session.commit()
        for old in final_dir.parent.iterdir():
            if old != final_dir and old.is_dir(): shutil.rmtree(old, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"snapshot_id": job["snapshot_id"], "revision": revision}


def _hydrate(value: Any, directory: Path) -> Any:
    if isinstance(value, dict) and "$object_key" in value:
        store = get_artifact_store()
        return store.get_url(value["$object_key"]) if store else value
    if isinstance(value, dict) and "$artifact" in value:
        path = directory / value["$artifact"]
        return "data:%s;base64,%s" % (value.get("mime_type", "application/octet-stream"), base64.b64encode(path.read_bytes()).decode())
    if isinstance(value, dict): return {k: _hydrate(v, directory) for k, v in value.items()}
    if isinstance(value, list): return [_hydrate(v, directory) for v in value]
    return value


def get_snapshot(snapshot_id: str, hydrate: bool = True) -> dict[str, Any] | None:
    with session_scope() as session:
        row = session.get(AnalysisSnapshot, snapshot_id)
        if not row: return None
        result = row.model_dump()
        result["config"] = json.loads(result.pop("config_json"))
        raw = json.loads(result.pop("payload_json") or "null")
        result["payload"] = _hydrate(raw, Path(result["artifact_dir"])) if hydrate and raw is not None and result.get("artifact_dir") else raw
        return result


def list_snapshots(optimization_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.exec(select(AnalysisSnapshot).where(AnalysisSnapshot.optimization_id == optimization_id, AnalysisSnapshot.revision > 0).order_by(AnalysisSnapshot.completed_at.desc())).all()
        return [{"id": r.id, "optimization_id": r.optimization_id, "revision": r.revision, "config": json.loads(r.config_json), "created_at": r.created_at, "completed_at": r.completed_at} for r in rows]


def create_report(snapshot: dict[str, Any], provider: str, model_name: str, description: str | None) -> str:
    report = AnalysisReport(id=str(uuid.uuid4()), optimization_id=snapshot["optimization_id"], snapshot_id=snapshot["id"], snapshot_revision=snapshot["revision"], status="running", provider=provider, model_name=model_name, dataset_description=description, created_at=_now())
    with session_scope() as session:
        session.add(report); session.commit()
    return report.id


def complete_report(report_id: str, markdown: str) -> None:
    with session_scope() as session:
        row = session.get(AnalysisReport, report_id)
        if row: row.status, row.markdown, row.completed_at = "completed", markdown, _now(); session.add(row); session.commit()


def fail_report(report_id: str, error: str) -> None:
    with session_scope() as session:
        row = session.get(AnalysisReport, report_id)
        if row: row.status, row.error, row.completed_at = "failed", error, _now(); session.add(row); session.commit()


def list_reports(snapshot_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        return [r.model_dump() for r in session.exec(select(AnalysisReport).where(AnalysisReport.snapshot_id == snapshot_id).order_by(AnalysisReport.created_at.desc())).all()]


def artifact_url(snapshot_id: str, filename: str) -> str | None:
    with session_scope() as session:
        row = session.exec(select(AnalysisArtifact).where(AnalysisArtifact.snapshot_id == snapshot_id, AnalysisArtifact.filename == filename)).first()
    if not row or not row.object_key: return None
    store = get_artifact_store()
    return store.get_url(row.object_key) if store else None


def delete_for_run(optimization_id: str) -> None:
    with session_scope() as session:
        snapshots = session.exec(select(AnalysisSnapshot).where(AnalysisSnapshot.optimization_id == optimization_id)).all()
        snapshot_ids = [s.id for s in snapshots]
        for snapshot_id in snapshot_ids:
            for row in session.exec(select(AnalysisJob).where(AnalysisJob.snapshot_id == snapshot_id)).all(): session.delete(row)
            for row in session.exec(select(AnalysisReport).where(AnalysisReport.snapshot_id == snapshot_id)).all(): session.delete(row)
            for row in session.exec(select(AnalysisArtifact).where(AnalysisArtifact.snapshot_id == snapshot_id)).all(): session.delete(row)
            session.delete(session.get(AnalysisSnapshot, snapshot_id))
        session.commit()
    shutil.rmtree(ARTIFACT_ROOT / optimization_id, ignore_errors=True)
