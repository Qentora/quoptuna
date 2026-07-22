"""
Canonical location and naming for Optuna study databases.

Historically the storage path was built as ``db/{db_name}.db`` without
normalizing, so a ``db_name`` that already ended in ``.db`` produced files
like ``results.db.db``. New databases always get a single ``.db`` suffix;
loading falls back to the legacy double-suffixed file when that is the one
that exists on disk.
"""

from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse
from quoptuna.server.core.config import settings

DB_DIR = Path("db")

DEFAULT_DB_NAME = "results"


def ensure_db_dir() -> Path:
    DB_DIR.mkdir(exist_ok=True)
    return DB_DIR


def normalize_db_name(db_name: str) -> str:
    """Strip any trailing ``.db`` suffixes from a database name."""
    name = (db_name or "").strip()
    while name.lower().endswith(".db"):
        name = name[:-3]
    return name or DEFAULT_DB_NAME


def optuna_db_path(db_name: str) -> Path:
    """Path of the Optuna SQLite database for ``db_name``.

    Prefers the canonical single-suffix file, but keeps loading a legacy
    double-suffixed file (e.g. ``results.db.db``) when only that one exists.
    """
    canonical = DB_DIR / f"{normalize_db_name(db_name)}.db"
    legacy = DB_DIR / f"{db_name}.db"
    if legacy != canonical and legacy.exists() and not canonical.exists():
        return legacy
    return canonical


def optuna_storage_url(db_name: str) -> str:
    url = ""
    if settings.OPTUNA_DATABASE_URL:
        url = settings.OPTUNA_DATABASE_URL
    elif settings.DATABASE_URL.startswith(("postgresql://", "postgresql+")):
        url = settings.DATABASE_URL
    else:
        return f"sqlite:///{optuna_db_path(db_name)}"
    ensure_optuna_schema()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    parsed = urlparse(url)
    options = quote(f"-csearch_path={settings.OPTUNA_DB_SCHEMA}", safe="")
    separator = "&" if parsed.query else ""
    return urlunparse(parsed._replace(query=f"{parsed.query}{separator}options={options}"))


def ensure_optuna_schema() -> None:
    """Create the isolated PostgreSQL schema used by Optuna."""
    url = settings.OPTUNA_DATABASE_URL or settings.DATABASE_URL
    if not url.startswith(("postgresql://", "postgresql+")):
        return
    from sqlalchemy import create_engine, text  # noqa: PLC0415
    engine = create_engine(url)
    with engine.begin() as connection:
        schema = settings.OPTUNA_DB_SCHEMA.replace('"', '""')
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))  # noqa: S608
