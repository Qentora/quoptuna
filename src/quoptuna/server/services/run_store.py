"""Durable SQLModel store for runs and datasets."""

import json
from typing import Any, Dict, List, Optional

from sqlmodel import select

from quoptuna.server.services.database import session_scope
from quoptuna.server.services.models import Dataset, Run

# Retained as a compatibility hook for existing tests and local deployments.
APP_DB_PATH = "db/quoptuna_app.db"


def _run_dict(row: Run) -> Dict[str, Any]:
    result = row.model_dump()
    result["request"] = json.loads(result.pop("request_json") or "null")
    result["best_params"] = json.loads(result.pop("best_params_json") or "null")
    result["id"] = result["job_id"]
    return result


def save_run(job: Dict[str, Any]) -> None:
    request = job.get("request") or {}
    values = Run(
        job_id=job["id"],
        study_name=request.get("study_name"),
        db_name=request.get("database_name"),
        status=job.get("status"),
        request_json=json.dumps(request),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error"),
        best_value=job.get("best_value"),
        best_params_json=json.dumps(job.get("best_params")),
        total_trials=job.get("total_trials"),
        current_trial=job.get("current_trial"),
        session_id=job.get("session_id"),
        user_id=job.get("user_id"),
        dataset_id=request.get("dataset_id"),
        source_file_id=job.get("source_file_id") or request.get("dataset_id"),
        source_file_path=job.get("source_file_path"),
        study_storage_key=job.get("study_storage_key"),
        artifact_storage=job.get("artifact_storage"),
    )
    with session_scope() as session:
        existing = session.get(Run, values.job_id)
        if existing:
            for key, value in values.model_dump(exclude={"job_id", "created_at"}).items():
                setattr(existing, key, value)
            session.add(existing)
        else:
            session.add(values)
        session.commit()


def update_run(job_id: str, **fields: Any) -> None:
    if "best_params" in fields:
        fields["best_params_json"] = json.dumps(fields.pop("best_params"))
    allowed = set(Run.model_fields) - {"job_id", "created_at"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown run columns: {unknown}")
    with session_scope() as session:
        row = session.get(Run, job_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        session.add(row)
        session.commit()


def get_run(job_id: str) -> Optional[Dict[str, Any]]:
    with session_scope() as session:
        row = session.get(Run, job_id)
        return _run_dict(row) if row else None


def list_runs() -> List[Dict[str, Any]]:
    with session_scope() as session:
        rows = session.exec(select(Run).order_by(Run.started_at.desc())).all()
        return [_run_dict(row) for row in rows]


def delete_run(job_id: str) -> None:
    with session_scope() as session:
        row = session.get(Run, job_id)
        if row:
            session.delete(row)
            session.commit()


def mark_stale_runs_interrupted() -> None:
    with session_scope() as session:
        rows = session.exec(select(Run).where(Run.status.in_(["running", "pending"]))).all()
        for row in rows:
            row.status = "interrupted"
            session.add(row)
        session.commit()


def save_dataset(record: Dict[str, Any]) -> None:
    values = Dataset(
        id=record["id"],
        name=record.get("name"),
        source=record.get("source"),
        file_path=record.get("file_path"),
        object_key=record.get("object_key"),
        checksum=record.get("checksum"),
        rows=record.get("rows"),
        columns_json=json.dumps(record.get("columns")),
    )
    with session_scope() as session:
        existing = session.get(Dataset, values.id)
        if existing:
            for key, value in values.model_dump(exclude={"id", "created_at"}).items():
                setattr(existing, key, value)
            session.add(existing)
        else:
            session.add(values)
        session.commit()


def all_datasets() -> List[Dict[str, Any]]:
    with session_scope() as session:
        records = []
        for row in session.exec(select(Dataset)).all():
            value = row.model_dump()
            value["columns"] = json.loads(value.pop("columns_json") or "[]")
            records.append(value)
        return records
