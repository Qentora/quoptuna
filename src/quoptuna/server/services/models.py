"""SQLModel tables owned by QuOptuna."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    __tablename__ = "quoptuna_runs"
    job_id: str = Field(primary_key=True)
    study_name: Optional[str] = None
    db_name: Optional[str] = None
    status: Optional[str] = None
    request_json: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    best_value: Optional[float] = None
    best_params_json: Optional[str] = None
    total_trials: Optional[int] = None
    current_trial: Optional[int] = None
    session_id: Optional[str] = Field(default=None, index=True)
    user_id: Optional[str] = Field(default=None, index=True)
    dataset_id: Optional[str] = None
    source_file_id: Optional[str] = None
    source_file_path: Optional[str] = None
    source_object_key: Optional[str] = None
    study_storage_key: Optional[str] = None
    artifact_storage: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Dataset(SQLModel, table=True):
    __tablename__ = "quoptuna_datasets"
    id: str = Field(primary_key=True)
    name: Optional[str] = None
    source: Optional[str] = None
    file_path: Optional[str] = None
    object_key: Optional[str] = None
    checksum: Optional[str] = None
    rows: Optional[int] = None
    columns_json: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisSnapshot(SQLModel, table=True):
    __tablename__ = "quoptuna_analysis_snapshots"
    id: str = Field(primary_key=True)
    optimization_id: str = Field(index=True)
    config_key: str
    config_json: str
    revision: int = 0
    payload_json: Optional[str] = None
    artifact_dir: Optional[str] = None
    storage_backend: Optional[str] = None
    artifact_prefix: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None


class AnalysisJob(SQLModel, table=True):
    __tablename__ = "quoptuna_analysis_jobs"
    id: str = Field(primary_key=True)
    snapshot_id: str = Field(index=True)
    status: str
    current_section: Optional[str] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None


class AnalysisReport(SQLModel, table=True):
    __tablename__ = "quoptuna_analysis_reports"
    id: str = Field(primary_key=True)
    optimization_id: str
    snapshot_id: str = Field(index=True)
    snapshot_revision: int
    status: str
    provider: str
    model_name: str
    dataset_description: Optional[str] = None
    markdown: Optional[str] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None


class AnalysisArtifact(SQLModel, table=True):
    __tablename__ = "quoptuna_analysis_artifacts"
    id: str = Field(primary_key=True)
    optimization_id: str = Field(index=True)
    snapshot_id: str = Field(index=True)
    revision: int
    filename: str
    object_key: Optional[str] = None
    storage_backend: str = "local"
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
