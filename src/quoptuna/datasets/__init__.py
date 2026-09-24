"""Datasets shipped with quoptuna.

These are small, permissively redistributable datasets committed to the
repository so the dataset picker works without network access. They are keyed
by numeric id so they can be listed and loaded through the same code path as
the remotely fetched UCI datasets.

Ids below 9000 are real UCI archive ids. Ids from 9000 up are reserved for
bundled datasets that have no UCI entry; nothing ever requests them from the
archive, because ``bundled_dataset_path`` resolves them first.

Regenerate a CSV with ``scripts/prepare_<name>.py``; do not hand-edit them.
"""

from __future__ import annotations

from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent

# dataset id -> catalog entry. ``filename`` is relative to ``DATASETS_DIR`` and
# is gzipped; ``pandas`` infers the codec from the suffix, so readers are
# unchanged by the compression.
BUNDLED_DATASETS: dict[int, dict] = {
    422: {
        "id": 422,
        "name": "Wireless Indoor Localization",
        "description": "2000 samples, 7 features - room (1-4) from WiFi signal strengths",
        "num_instances": 2000,
        "num_features": 7,
        "filename": "wireless_indoor_localization.csv.gz",
        "bundled": True,
    },
    9020: {
        "id": 9020,
        "name": "RECS 2020 Housing Tenure (subset)",
        "description": (
            "2000 samples, 12 features - owned vs rented home, from the EIA "
            "Residential Energy Consumption Survey 2020. Seeded stratified "
            "sample of the full survey; sized for quantum models"
        ),
        "num_instances": 2000,
        "num_features": 12,
        "filename": "recs2020_housing_tenure.csv.gz",
        "bundled": True,
    },
    9021: {
        "id": 9021,
        "name": "RECS 2020 Housing Tenure (full)",
        "description": (
            "18314 samples, 12 features - owned vs rented home, from the EIA "
            "Residential Energy Consumption Survey 2020. Every eligible "
            "household; kernel-based quantum models are impractical at this size"
        ),
        "num_instances": 18314,
        "num_features": 12,
        "filename": "recs2020_housing_tenure_full.csv.gz",
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
