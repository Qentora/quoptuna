"""SQLModel database engine and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from quoptuna.server.core.config import settings
from quoptuna.server.services.models import (
    AnalysisArtifact,
    AnalysisJob,
    AnalysisReport,
    AnalysisSnapshot,
    Dataset,
    Run,
)

APPLICATION_MODELS = (
    Run,
    Dataset,
    AnalysisSnapshot,
    AnalysisJob,
    AnalysisReport,
    AnalysisArtifact,
)


def _database_url() -> str:
    # Keep the historical local location unless an explicit URL is configured.
    try:
        from quoptuna.server.services import run_store

        if run_store.APP_DB_PATH != "db/quoptuna_app.db":
            return f"sqlite:///{run_store.APP_DB_PATH}"
    except ImportError:
        pass
    url = settings.DATABASE_URL or "sqlite:///./db/quoptuna_app.db"
    return (
        url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("postgresql://")
        else url
    )


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True, "pool_recycle": 1800}


@lru_cache(maxsize=8)
def get_engine(url: str | None = None):
    url = url or _database_url()
    return create_engine(url, **_engine_kwargs(url))


def init_db() -> None:
    """Create application tables; production schema changes use migrations."""
    SQLModel.metadata.create_all(
        get_engine(),
        tables=[model.__table__ for model in APPLICATION_MODELS],  # type: ignore[attr-defined, union-attr]
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    init_db()
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
