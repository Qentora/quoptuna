"""Datasets shipped with quoptuna.

These are small, permissively redistributable UCI datasets committed to the
repository so the dataset picker works without network access to the UCI
archive. They are keyed by their UCI numeric id so they can be listed and
loaded through the same code path as the remotely fetched datasets.

Regenerate a CSV with ``scripts/prepare_<name>.py``; do not hand-edit them.
"""

from __future__ import annotations

from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent

# UCI id -> catalog entry. ``filename`` is relative to ``DATASETS_DIR``.
BUNDLED_DATASETS: dict[int, dict] = {
    422: {
        "id": 422,
        "name": "Wireless Indoor Localization",
        "description": "2000 samples, 7 features - room (1-4) from WiFi signal strengths",
        "num_instances": 2000,
        "num_features": 7,
        "filename": "wireless_indoor_localization.csv",
        "bundled": True,
    },
}


def bundled_dataset_path(dataset_id: int) -> Path | None:
    """Return the on-disk CSV for a bundled dataset, or ``None`` if not bundled."""
    entry = BUNDLED_DATASETS.get(dataset_id)
    if entry is None:
        return None
    path = DATASETS_DIR / entry["filename"]
    return path if path.exists() else None


def bundled_catalog() -> list[dict]:
    """Catalog entries for every bundled dataset whose CSV is actually present."""
    return [
        {k: v for k, v in entry.items() if k != "filename"}
        for dataset_id, entry in BUNDLED_DATASETS.items()
        if bundled_dataset_path(dataset_id) is not None
    ]


__all__ = ["BUNDLED_DATASETS", "DATASETS_DIR", "bundled_catalog", "bundled_dataset_path"]
