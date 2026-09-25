"""Regression coverage for UCI source-specific target normalization."""

import pandas as pd

from quoptuna.datasets import normalize_uci_targets


def test_adult_income_test_file_suffixes_do_not_create_extra_classes():
    raw = pd.DataFrame(
        {
            "income": [" <=50K", " >50K", " <=50K.", " >50K."],
        }
    )

    normalized = normalize_uci_targets(2, raw)

    assert normalized["income"].tolist() == ["<=50K", ">50K", "<=50K", ">50K"]
    assert normalized["income"].nunique() == 2
    assert raw["income"].nunique() == 4


def test_other_uci_targets_are_not_rewritten():
    raw = pd.DataFrame({"target": ["class.one.", "class.two."]})

    assert normalize_uci_targets(850, raw) is raw
