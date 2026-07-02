"""
Dataset registry.

Maps a dataset id (uuid for uploads, str(uci_id) for UCI datasets) to the
persisted CSV file path and basic metadata. This lets the optimize/preview
endpoints resolve a dataset by id regardless of its source. Records are kept
in memory for speed and mirrored to the app SQLite store so they survive
backend restarts.
"""

from typing import Dict, List, Optional, TypedDict

from quoptuna.server.services import run_store


class DatasetRecord(TypedDict, total=False):
    id: str
    name: str
    source: str  # 'upload' | 'uci'
    file_path: str
    rows: int
    columns: List[str]


_registry: Dict[str, DatasetRecord] = {}


def register(record: DatasetRecord) -> DatasetRecord:
    """Register (or overwrite) a dataset record by its id."""
    _registry[record["id"]] = record
    run_store.save_dataset(dict(record))
    return record


def _load_persisted(dataset_id: str) -> Optional[DatasetRecord]:
    for record in run_store.all_datasets():
        if record["id"] == dataset_id:
            _registry[dataset_id] = record  # type: ignore[assignment]
            return _registry[dataset_id]
    return None


def get(dataset_id: str) -> Optional[DatasetRecord]:
    """Look up a dataset record by id, falling back to the persisted store."""
    record = _registry.get(dataset_id)
    if record is None:
        record = _load_persisted(dataset_id)
    return record


def get_file_path(dataset_id: str) -> Optional[str]:
    """Convenience accessor for a registered dataset's file path."""
    record = get(dataset_id)
    if record is None:
        return None
    return record.get("file_path")


def all_records() -> List[DatasetRecord]:
    """Return all dataset records (persisted store merged with in-memory)."""
    merged: Dict[str, DatasetRecord] = {r["id"]: r for r in run_store.all_datasets()}  # type: ignore[misc]
    merged.update(_registry)
    return list(merged.values())
