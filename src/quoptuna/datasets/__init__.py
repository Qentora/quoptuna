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
    15: {
        "id": 15,
        "name": "Breast Cancer Wisconsin (Original)",
        "description": "699 samples, 9 features - benign vs malignant cells; source missing values mode-imputed",
        "num_instances": 699,
        "num_features": 9,
        "filename": "uci_15.csv.gz",
        "bundled": True,
    },
    30: {
        "id": 30,
        "name": "Contraceptive Method Choice",
        "description": "1473 samples, 9 features - three contraceptive choices; historical demographic survey",
        "num_instances": 1473,
        "num_features": 9,
        "filename": "uci_30.csv.gz",
        "bundled": True,
    },
    53: {
        "id": 53,
        "name": "Iris",
        "description": "150 samples, 4 features - balanced three-species flower classification",
        "num_instances": 150,
        "num_features": 4,
        "filename": "uci_53.csv.gz",
        "bundled": True,
    },
    161: {
        "id": 161,
        "name": "Mammographic Mass",
        "description": "961 samples, 4 predictive features - benign vs malignant mass; BI-RADS excluded",
        "num_instances": 961,
        "num_features": 4,
        "filename": "uci_161.csv.gz",
        "bundled": True,
    },
    236: {
        "id": 236,
        "name": "Seeds",
        "description": "210 samples, 7 features - balanced three-class wheat kernel classification",
        "num_instances": 210,
        "num_features": 7,
        "filename": "uci_236.csv.gz",
        "bundled": True,
    },
    257: {
        "id": 257,
        "name": "User Knowledge Modeling",
        "description": "403 samples, 5 features - four-level learner knowledge classification",
        "num_instances": 403,
        "num_features": 5,
        "filename": "uci_257.csv.gz",
        "bundled": True,
    },
    267: {
        "id": 267,
        "name": "Banknote Authentication",
        "description": "1372 samples, 4 features - authentic vs forged banknotes",
        "num_instances": 1372,
        "num_features": 4,
        "filename": "uci_267.csv.gz",
        "bundled": True,
    },
    357: {
        "id": 357,
        "name": "Occupancy Detection",
        "description": "20560 ordered samples, 5 features - room occupancy; preserve source chronology when splitting",
        "num_instances": 20560,
        "num_features": 5,
        "filename": "uci_357.csv.gz",
        "bundled": True,
    },
    523: {
        "id": 523,
        "name": "Exasens",
        "description": "399 samples, 6 features - four respiratory diagnoses; source missing values mode-imputed",
        "num_instances": 399,
        "num_features": 6,
        "filename": "uci_523.csv.gz",
        "bundled": True,
    },
    545: {
        "id": 545,
        "name": "Rice Cammeo and Osmancik",
        "description": "3810 samples, 7 features - rice variety classification; kernel quantum models are expensive",
        "num_instances": 3810,
        "num_features": 7,
        "filename": "uci_545.csv.gz",
        "bundled": True,
    },
    850: {
        "id": 850,
        "name": "Raisin",
        "description": "900 samples, 7 features - balanced Kecimen vs Besni raisin classification",
        "num_instances": 900,
        "num_features": 7,
        "filename": "uci_850.csv.gz",
        "bundled": True,
    },
    9022: {
        "id": 9022,
        "name": "NTIA Internet Use Survey 2023 Wearable Use",
        "description": "2000 samples, 7 features - seeded stratified adult wearable-use survey extract",
        "num_instances": 2000,
        "num_features": 7,
        "filename": "ntia_2023_wearable_use.csv.gz",
        "bundled": True,
    },
    9023: {
        "id": 9023,
        "name": "ACS PUMS 2023 Internet Access",
        "description": "2000 samples, 6 features - seeded stratified household internet-access survey extract",
        "num_instances": 2000,
        "num_features": 6,
        "filename": "acs_2023_internet_access.csv.gz",
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
