"""
In-memory dataset registry.

Maps a dataset id (uuid for uploads, str(uci_id) for UCI datasets) to the
persisted CSV file path and basic metadata. This lets the optimize/preview
endpoints resolve a dataset by id regardless of its source.
"""

from typing import Dict, List, Optional, TypedDict


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
    return record


def get(dataset_id: str) -> Optional[DatasetRecord]:
    """Look up a dataset record by id."""
    return _registry.get(dataset_id)


def get_file_path(dataset_id: str) -> Optional[str]:
    """Convenience accessor for a registered dataset's file path."""
    record = _registry.get(dataset_id)
    if record is None:
        return None
    return record.get("file_path")


def all_records() -> List[DatasetRecord]:
    """Return all registered dataset records."""
    return list(_registry.values())
