"""Idempotent migration of the legacy application SQLite store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from quoptuna.server.services.models import AnalysisJob, AnalysisReport, AnalysisSnapshot, Dataset, Run


def migrate_app_store(source: str | Path, target_url: str, dry_run: bool = False) -> dict[str, int | bool]:
    source = str(source)
    with sqlite3.connect(source) as source_conn:
        source_conn.row_factory = sqlite3.Row
        runs = source_conn.execute("SELECT * FROM runs").fetchall()
        datasets = source_conn.execute("SELECT * FROM datasets").fetchall()
        table_names = {row[0] for row in source_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        snapshots = source_conn.execute("SELECT * FROM analysis_snapshots").fetchall() if "analysis_snapshots" in table_names else []
        jobs = source_conn.execute("SELECT * FROM analysis_jobs").fetchall() if "analysis_jobs" in table_names else []
        reports = source_conn.execute("SELECT * FROM analysis_reports").fetchall() if "analysis_reports" in table_names else []
    if dry_run:
        return {"dry_run": True, "runs": len(runs), "datasets": len(datasets), "snapshots": len(snapshots), "reports": len(reports)}
    if target_url.startswith("postgresql://"):
        target_url = target_url.replace("postgresql://", "postgresql+psycopg://", 1)
    target_engine = create_engine(target_url, connect_args={"check_same_thread": False} if target_url.startswith("sqlite") else {}, pool_pre_ping=True if not target_url.startswith("sqlite") else False)
    SQLModel.metadata.create_all(target_engine)
    with Session(target_engine, expire_on_commit=False) as session:
        for row in runs:
            get = row.keys().__contains__
            request_json = row["request_json"] if get("request_json") else "{}"
            session.merge(Run(
                job_id=row["job_id"], study_name=row["study_name"], db_name=row["db_name"],
                status=row["status"], request_json=request_json, started_at=row["started_at"],
                completed_at=row["completed_at"], error=row["error"], best_value=row["best_value"],
                best_params_json=row["best_params_json"], total_trials=row["total_trials"],
                current_trial=row["current_trial"], session_id=row["session_id"] if get("session_id") else None,
                user_id=row["user_id"] if get("user_id") else None,
                dataset_id=row["dataset_id"] if get("dataset_id") else None,
                source_file_id=row["source_file_id"] if get("source_file_id") else None,
                source_file_path=row["source_file_path"] if get("source_file_path") else None,
                study_storage_key=row["study_storage_key"] if get("study_storage_key") else row["db_name"],
                artifact_storage=row["artifact_storage"] if get("artifact_storage") else "local",
            ))
        for row in datasets:
            columns = row["columns_json"] if "columns_json" in row.keys() else "[]"
            session.merge(Dataset(id=row["id"], name=row["name"], source=row["source"], file_path=row["file_path"], rows=row["rows"], columns_json=columns))
        for row in snapshots:
            get = row.keys().__contains__
            session.merge(AnalysisSnapshot(
                id=row["id"], optimization_id=row["optimization_id"], config_key=row["config_key"],
                config_json=row["config_json"], revision=row["revision"], payload_json=row["payload_json"],
                artifact_dir=row["artifact_dir"], storage_backend=row["storage_backend"] if get("storage_backend") else "local",
                artifact_prefix=row["artifact_prefix"] if get("artifact_prefix") else None,
                created_at=row["created_at"], completed_at=row["completed_at"],
            ))
        for row in jobs:
            session.merge(AnalysisJob(id=row["id"], snapshot_id=row["snapshot_id"], status=row["status"],
                                      current_section=row["current_section"], error=row["error"],
                                      created_at=row["created_at"], completed_at=row["completed_at"]))
        for row in reports:
            session.merge(AnalysisReport(id=row["id"], optimization_id=row["optimization_id"],
                snapshot_id=row["snapshot_id"], snapshot_revision=row["snapshot_revision"], status=row["status"],
                provider=row["provider"], model_name=row["model_name"], dataset_description=row["dataset_description"],
                markdown=row["markdown"], error=row["error"], created_at=row["created_at"], completed_at=row["completed_at"]))
        session.commit()
    return {"dry_run": False, "runs": len(runs), "datasets": len(datasets), "snapshots": len(snapshots), "reports": len(reports)}
