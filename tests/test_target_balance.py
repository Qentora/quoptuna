"""Target distribution labels must be consistent across picker and preview."""

import pandas as pd

from quoptuna.datasets import bundled_catalog_with_balance
from quoptuna.datasets.balance import target_balance_profile


def test_binary_balance_thresholds():
    assert target_balance_profile(pd.Series(["a"] * 5 + ["b"] * 5))["label"] == "balanced"
    assert target_balance_profile(pd.Series(["a"] * 8 + ["b"] * 2))["label"] == "moderate_imbalance"
    assert target_balance_profile(pd.Series(["a"] * 9 + ["b"]))["label"] == "imbalanced"


def test_multiclass_profile_keeps_each_class_count():
    profile = target_balance_profile(pd.Series(["a", "a", "b", "c"]))

    assert profile == {
        "kind": "multiclass",
        "label": "multiclass",
        "classes": [
            {"label": "a", "count": 2},
            {"label": "b", "count": 1},
            {"label": "c", "count": 1},
        ],
        "minority_fraction": 0.25,
    }


def test_bundled_catalog_reports_adult_target_balance():
    catalog = {entry["id"]: entry for entry in bundled_catalog_with_balance()}

    assert catalog[2]["target_column"] == "income"
    assert catalog[2]["target_balance"]["label"] == "moderate_imbalance"
    assert catalog[9024]["target_balance"]["label"] == "balanced"
