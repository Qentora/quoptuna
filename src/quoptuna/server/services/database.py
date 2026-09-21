"""SQLModel database engine and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from quoptuna.server.core.config import settings
from quoptuna.server.services.models import (
    AnalysisArtifact,
    AnalysisJob,
    AnalysisReport,
    AnalysisRevision,
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
    AnalysisRevision,
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


def _ensure_sqlite_parent(url: str) -> None:
    """Create the directory holding a SQLite file.

    ``db/`` is gitignored, so a fresh clone has no such directory and SQLite
    reports "unable to open database file" instead of creating the database.
    """
    if not url.startswith("sqlite"):
        return
    _, separator, path = url.partition(":///")
    if not separator:
        return
    path = path.split("?", 1)[0]
    if not path or path.startswith(":memory:"):
        return
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=8)
def get_engine(url: str | None = None):
    url = url or _database_url()
    _ensure_sqlite_parent(url)
    return create_engine(url, **_engine_kwargs(url))


#: Columns added to existing tables after they shipped. ``create_all`` only
#: creates missing tables, so an established local database would otherwise
#: keep the old shape. Each entry is additive and nullable; anything beyond
#: that belongs in a real migration.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("quoptuna_analysis_jobs", "partial_json", "VARCHAR"),
)


def _add_missing_columns(engine) -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    for table, column, column_type in _ADDED_COLUMNS:
        if table not in existing:
            continue  # create_all already built it with the column
        if column in {c["name"] for c in inspector.get_columns(table)}:
            continue
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"))


def init_db() -> None:
    """Create application tables; production schema changes use migrations."""
    engine = get_engine()
    SQLModel.metadata.create_all(
        engine,
        tables=[model.__table__ for model in APPLICATION_MODELS],  # type: ignore[attr-defined, union-attr]
    )
    _add_missing_columns(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    init_db()
    with Session(get_engine(), expire_on_commit=False) as session:
        yield session
