"""Durable analysis snapshots, binary artifacts, jobs, and generated reports."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from quoptuna.server.services import run_store
from quoptuna.server.services.storage import ensure_db_dir

ARTIFACT_ROOT = Path("db/analysis")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_snapshots (
    id TEXT PRIMARY KEY,
    optimization_id TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_json TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT,
    artifact_dir TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(optimization_id, config_key)
);
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    current_section TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS analysis_reports (
    id TEXT PRIMARY KEY,
    optimization_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    snapshot_revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dataset_description TEXT,
    markdown TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS analysis_snapshots_run_idx
    ON analysis_snapshots(optimization_id, completed_at DESC);
CREATE INDEX IF NOT EXISTS analysis_reports_snapshot_idx
    ON analysis_reports(snapshot_id, completed_at DESC);
"""


def _connect():
    conn = run_store._connect()  # noqa: SLF001 - shares the app database intentionally
    conn.executescript(_SCHEMA)
    return conn


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
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM analysis_snapshots WHERE optimization_id = ? AND config_key = ?",
            (optimization_id, key),
        ).fetchone()
        snapshot_id = row["id"] if row else str(uuid.uuid4())
        if not row:
            conn.execute(
                "INSERT INTO analysis_snapshots "
                "(id, optimization_id, config_key, config_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, optimization_id, key, json.dumps(config), now),
            )
        active = conn.execute(
            "SELECT id, status FROM analysis_jobs WHERE snapshot_id = ? "
            "AND status IN ('pending', 'running') ORDER BY created_at DESC LIMIT 1",
            (snapshot_id,),
        ).fetchone()
        if active:
            return {
                "id": active["id"],
                "snapshot_id": snapshot_id,
                "status": active["status"],
                "created": False,
            }
        job_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO analysis_jobs (id, snapshot_id, status, created_at) VALUES (?, ?, ?, ?)",
            (job_id, snapshot_id, "pending", now),
        )
    return {"id": job_id, "snapshot_id": snapshot_id, "status": "pending", "created": True}


def create_revision_job(snapshot_id: str) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM analysis_snapshots WHERE id = ? AND revision > 0", (snapshot_id,)
        ).fetchone()
        if not row:
            raise KeyError(snapshot_id)
        conn.execute(
            "INSERT INTO analysis_jobs (id, snapshot_id, status, created_at) VALUES (?, ?, 'running', ?)",
            (job_id, snapshot_id, _now()),
        )
    return {"id": job_id, "snapshot_id": snapshot_id, "status": "running"}


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"status", "current_section", "error", "completed_at"}
    if set(fields) - allowed:
        raise ValueError("Invalid analysis job field")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE analysis_jobs SET {assignments} WHERE id = ?",  # noqa: S608
            (*fields.values(), job_id),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT j.*, s.optimization_id, s.revision, s.config_json "
            "FROM analysis_jobs j JOIN analysis_snapshots s ON s.id = j.snapshot_id "
            "WHERE j.id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["config"] = json.loads(result.pop("config_json"))
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


def complete_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise KeyError(job_id)
    revision = int(job["revision"]) + 1
    ensure_db_dir()
    final_dir = ARTIFACT_ROOT / job["optimization_id"] / job["snapshot_id"] / str(revision)
    temp_dir = final_dir.with_name(f".{revision}-{job_id}.tmp")
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        stored_payload = _extract_artifacts(payload, temp_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(final_dir)
        now = _now()
        with _connect() as conn:
            conn.execute(
                "UPDATE analysis_snapshots SET revision = ?, payload_json = ?, "
                "artifact_dir = ?, completed_at = ? WHERE id = ?",
                (revision, json.dumps(stored_payload), str(final_dir), now, job["snapshot_id"]),
            )
            conn.execute(
                "UPDATE analysis_jobs SET status = 'completed', current_section = 'complete', "
                "completed_at = ? WHERE id = ?",
                (now, job_id),
            )
        for old_revision in final_dir.parent.iterdir():
            if old_revision != final_dir and old_revision.is_dir():
                shutil.rmtree(old_revision, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"snapshot_id": job["snapshot_id"], "revision": revision}


def _hydrate_artifacts(value: Any, directory: Path) -> Any:
    if isinstance(value, dict) and "$artifact" in value:
        path = directory / value["$artifact"]
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:{value.get('mime_type', 'application/octet-stream')};base64,{encoded}"
    if isinstance(value, dict):
        return {key: _hydrate_artifacts(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [_hydrate_artifacts(item, directory) for item in value]
    return value


def get_snapshot(snapshot_id: str, hydrate: bool = True) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM analysis_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["config"] = json.loads(result.pop("config_json"))
    raw_payload = json.loads(result.pop("payload_json") or "null")
    result["payload"] = (
        _hydrate_artifacts(raw_payload, Path(result["artifact_dir"]))
        if hydrate and raw_payload is not None and result.get("artifact_dir")
        else raw_payload
    )
    return result


def list_snapshots(optimization_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, optimization_id, revision, config_json, created_at, completed_at "
            "FROM analysis_snapshots WHERE optimization_id = ? "
            "AND revision > 0 ORDER BY completed_at DESC",
            (optimization_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "optimization_id": row["optimization_id"],
            "revision": row["revision"],
            "config": json.loads(row["config_json"]),
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }
        for row in rows
    ]


def create_report(snapshot: dict[str, Any], provider: str, model_name: str, description: str | None) -> str:
    report_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO analysis_reports (id, optimization_id, snapshot_id, snapshot_revision, "
            "status, provider, model_name, dataset_description, created_at) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?)",
            (report_id, snapshot["optimization_id"], snapshot["id"], snapshot["revision"], provider, model_name, description, _now()),
        )
    return report_id


def complete_report(report_id: str, markdown: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE analysis_reports SET status = 'completed', markdown = ?, completed_at = ? WHERE id = ?",
            (markdown, _now(), report_id),
        )


def fail_report(report_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE analysis_reports SET status = 'failed', error = ?, completed_at = ? WHERE id = ?",
            (error, _now(), report_id),
        )


def list_reports(snapshot_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_reports WHERE snapshot_id = ? ORDER BY created_at DESC",
            (snapshot_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_for_run(optimization_id: str) -> None:
    with _connect() as conn:
        snapshot_ids = [row["id"] for row in conn.execute(
            "SELECT id FROM analysis_snapshots WHERE optimization_id = ?", (optimization_id,)
        ).fetchall()]
        for snapshot_id in snapshot_ids:
            conn.execute("DELETE FROM analysis_jobs WHERE snapshot_id = ?", (snapshot_id,))
            conn.execute("DELETE FROM analysis_reports WHERE snapshot_id = ?", (snapshot_id,))
        conn.execute("DELETE FROM analysis_snapshots WHERE optimization_id = ?", (optimization_id,))
    shutil.rmtree(ARTIFACT_ROOT / optimization_id, ignore_errors=True)
