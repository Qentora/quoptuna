"""Real-time file logging alongside the terminal.

``attach_file_logging()`` adds a rotating file handler to the root logger so
every backend/server/optuna log record is also written (and flushed per
record) to ``db/logs/quoptuna.log``, where agents and humans can read it
after the fact. Terminal handlers are untouched. Idempotent — safe to call
from every entrypoint (server, CLI).

Override the destination with the ``QUOPTUNA_LOG_FILE`` env var.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from quoptuna.backend.utils.storage import ensure_db_dir

_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_attached: Path | None = None


def default_log_path() -> Path:
    override = os.environ.get("QUOPTUNA_LOG_FILE", "").strip()
    if override:
        return Path(override)
    return ensure_db_dir() / "logs" / "quoptuna.log"


def attach_file_logging() -> Path:
    """Attach the rotating file handler to the root logger (once)."""
    global _attached  # noqa: PLW0603 - idempotency guard across entrypoints
    if _attached is not None:
        return _attached
    path = default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    # Root default is WARNING; INFO records must reach the file handler.
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    # Optuna logs through its own non-propagating handler; propagate so its
    # trial/study lines land in the file too (its stderr handler still runs).
    try:
        import optuna  # noqa: PLC0415

        optuna.logging.enable_propagation()
    except ImportError:  # pragma: no cover
        pass
    _attached = path
    logging.getLogger(__name__).info("Writing logs to %s", path)
    return path
