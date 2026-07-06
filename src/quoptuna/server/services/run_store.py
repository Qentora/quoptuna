"""
Durable store for optimization runs and dataset records.

Persists the state that previously lived only in in-memory dicts
(``optimization_jobs`` and the dataset registry) into a small app-level
SQLite database, so runs survive backend restarts and can be listed,
rehydrated, and replayed from the UI.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from quoptuna.server.services.storage import ensure_db_dir

APP_DB_PATH = "db/quoptuna_app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    job_id TEXT PRIMARY KEY,
    study_name TEXT,
    db_name TEXT,
    status TEXT,
    request_json TEXT,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    best_value REAL,
    best_params_json TEXT,
    total_trials INTEGER,
    current_trial INTEGER
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT,
    source TEXT,
    file_path TEXT,
    rows INTEGER,
    columns_json TEXT
);
"""

_RUN_COLUMNS = (
    "job_id",
    "study_name",
    "db_name",
    "status",
    "request_json",
    "started_at",
    "completed_at",
    "error",
    "best_value",
    "best_params_json",
    "total_trials",
    "current_trial",
)


def _connect() -> sqlite3.Connection:
    ensure_db_dir()
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
    run = dict(row)
    run["request"] = json.loads(run.pop("request_json") or "null")
    run["best_params"] = json.loads(run.pop("best_params_json") or "null")
    return run


def save_run(job: Dict[str, Any]) -> None:
    """Insert or replace a run row from a job dict (as kept in memory)."""
    request = job.get("request") or {}
    values = (
        job["id"],
        request.get("study_name"),
        request.get("database_name"),
        job.get("status"),
        json.dumps(request),
        job.get("started_at"),
        job.get("completed_at"),
        job.get("error"),
        job.get("best_value"),
        json.dumps(job.get("best_params")),
        job.get("total_trials"),
        job.get("current_trial"),
    )
    with _connect() as conn:
        placeholders = ",".join("?" for _ in _RUN_COLUMNS)
        conn.execute(
            # Column names come from the module-level _RUN_COLUMNS constant.
            f"INSERT OR REPLACE INTO runs ({','.join(_RUN_COLUMNS)}) VALUES ({placeholders})",  # noqa: S608
            values,
        )


def update_run(job_id: str, **fields: Any) -> None:
    """Update selected columns of a run row."""
    if "best_params" in fields:
        fields["best_params_json"] = json.dumps(fields.pop("best_params"))
    if not fields:
        return
    unknown = set(fields) - set(_RUN_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown run columns: {unknown}")
    assignments = ",".join(f"{key} = ?" for key in fields)
    with _connect() as conn:
        conn.execute(
            # Column names are validated against _RUN_COLUMNS above.
            f"UPDATE runs SET {assignments} WHERE job_id = ?",  # noqa: S608
            (*fields.values(), job_id),
        )


def get_run(job_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_run(row) if row else None


def list_runs() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    return [_row_to_run(row) for row in rows]


def delete_run(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))


def mark_stale_runs_interrupted() -> None:
    """Mark runs left 'running'/'pending' by a dead process as interrupted."""
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status = 'interrupted' WHERE status IN ('running', 'pending')"
        )


def save_dataset(record: Dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO datasets (id, name, source, file_path, rows, columns_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                record["id"],
                record.get("name"),
                record.get("source"),
                record.get("file_path"),
                record.get("rows"),
                json.dumps(record.get("columns")),
            ),
        )


def all_datasets() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM datasets").fetchall()
    records = []
    for row in rows:
        record = dict(row)
        record["columns"] = json.loads(record.pop("columns_json") or "[]")
        records.append(record)
    return records
