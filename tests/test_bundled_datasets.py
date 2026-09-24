"""Bundled datasets must be loadable and usable as classification tasks.

They exist so the picker works offline, which means a missing file, a renamed
target column, or a regenerated CSV that silently changed shape would only be
noticed by a user mid-run.
"""

import pandas as pd
import pytest

from quoptuna.backend.task_type import TaskSpec
from quoptuna.datasets import BUNDLED_DATASETS, bundled_catalog, bundled_dataset_path
from quoptuna.server.api.v1.data import POPULAR_UCI_DATASETS

# id -> the target column a user is expected to select.
EXPECTED_TARGETS = {422: "room", 9020: "tenure", 9021: "tenure"}
MAX_CLASSES = 20
MIN_CLASSES = 2


@pytest.mark.parametrize("dataset_id", sorted(BUNDLED_DATASETS))
def test_bundled_csv_matches_its_catalog_entry(dataset_id):
    entry = BUNDLED_DATASETS[dataset_id]
    path = bundled_dataset_path(dataset_id)
    assert path is not None, f"{entry['name']}: {entry['filename']} is missing from the package"

    frame = pd.read_csv(path)
    target = EXPECTED_TARGETS[dataset_id]

    assert target in frame.columns
    assert len(frame) == entry["num_instances"]
    # Every column except the target is a feature.
    assert len(frame.columns) - 1 == entry["num_features"]


@pytest.mark.parametrize("dataset_id", sorted(BUNDLED_DATASETS))
def test_bundled_dataset_is_a_usable_classification_task(dataset_id):
    """Guards against a regenerated CSV whose target became continuous or degenerate."""
    frame = pd.read_csv(bundled_dataset_path(dataset_id))
    target = frame[EXPECTED_TARGETS[dataset_id]]

    assert not frame.isna().any().any(), "bundled data must not need imputation"
    assert MIN_CLASSES <= target.nunique() <= MAX_CLASSES

    # The split node derives this; failing here means the run cannot start.
    spec = TaskSpec.from_target(target)
    assert spec.n_classes == target.nunique()


def test_catalog_exposes_every_present_dataset():
    catalog = {entry["id"]: entry for entry in bundled_catalog()}

    assert set(catalog) == set(BUNDLED_DATASETS)
    for entry in catalog.values():
        # The picker renders these; a filename leaking into the API is a bug.
        assert "filename" not in entry
        assert entry["bundled"] is True


def test_bundled_ids_never_collide_with_uci_ids():
    """Ids >= 9000 are reserved for datasets with no UCI entry.

    A bundled dataset given a real UCI id would shadow that archive dataset.
    """

    ids = [entry["id"] for entry in POPULAR_UCI_DATASETS]
    assert len(ids) == len(set(ids)), f"duplicate dataset ids in the picker: {ids}"
