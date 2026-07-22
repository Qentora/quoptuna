"""SQLModel database engine and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from quoptuna.server.core.config import settings


def _database_url() -> str:
    # Keep the historical local location unless an explicit URL is configured.
    try:
        from quoptuna.server.services import run_store  # noqa: PLC0415
        if run_store.APP_DB_PATH != "db/quoptuna_app.db":
            return f"sqlite:///{run_store.APP_DB_PATH}"
    except ImportError:
        pass
    url = settings.DATABASE_URL or "sqlite:///./db/quoptuna_app.db"
    return url.replace("postgresql://", "postgresql+psycopg://", 1) if url.startswith("postgresql://") else url


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True, "pool_recycle": 1800}


engine = None


def get_engine():
    global engine
    url = _database_url()
    if engine is None or str(engine.url) != url:
        engine = create_engine(url, **_engine_kwargs(url))
    return engine


def init_db() -> None:
    """Create application tables; production schema changes use migrations."""
    from quoptuna.server.services.models import (  # noqa: F401, PLC0415
        AnalysisArtifact,
        AnalysisJob,
        AnalysisReport,
        AnalysisSnapshot,
        Dataset,
        Run,
    )

    SQLModel.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    init_db()
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
