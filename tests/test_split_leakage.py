"""The train/validation boundary must survive resampling.

``RandomOverSampler`` duplicates minority rows verbatim. When the validation
split was carved *after* resampling, a row and its own copy landed on opposite
sides of that boundary, so the objective scored memorisation: on ILPD, 84% of
the positive-class validation rows were copies of training rows and the
reported F1 read 0.90 against a true 0.27.

These tests pin the ordering that prevents it.
"""

import numpy as np
import pandas as pd
import pytest

from quoptuna.server.services.workflow_service import WorkflowExecutor

MINORITY_ROWS = 20
MAJORITY_ROWS = 80


@pytest.fixture
def imbalanced_frame():
    """A frame whose rows are unique, so any duplicate is proof of a leak."""
    rng = np.random.default_rng(0)
    n = MINORITY_ROWS + MAJORITY_ROWS
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "target": np.array(["neg"] * MAJORITY_ROWS + ["pos"] * MINORITY_ROWS),
        }
    )


def _split(frame, resampling):
    executor = WorkflowExecutor({"nodes": [], "edges": []})
    return executor._execute_train_test_split(
        {"resampling": resampling},
        {
            "selected": {
                "x": frame[["f1", "f2"]],
                "y": frame["target"],
                "x_columns": ["f1", "f2"],
                "y_column": "target",
            }
        },
    )


def _row_set(frame):
    return {tuple(row) for row in np.asarray(frame)}


@pytest.mark.parametrize("resampling", ["none", "oversample", "undersample"])
def test_no_training_row_appears_in_validation(imbalanced_frame, resampling):
    split = _split(imbalanced_frame, resampling)

    leaked = _row_set(split["x_train"]) & _row_set(split["x_val"])
    assert not leaked, f"{len(leaked)} training rows leaked into validation ({resampling})"


@pytest.mark.parametrize("resampling", ["none", "oversample", "undersample"])
def test_no_training_row_appears_in_test(imbalanced_frame, resampling):
    split = _split(imbalanced_frame, resampling)

    leaked = _row_set(split["x_train"]) & _row_set(split["x_test"])
    assert not leaked, f"{len(leaked)} training rows leaked into test ({resampling})"


def test_only_the_training_portion_is_balanced(imbalanced_frame):
    """Validation and test keep the real class distribution.

    Balancing them would measure the model on a population that does not
    exist, and the reported metrics would not describe deployment.
    """
    split = _split(imbalanced_frame, "oversample")

    train_counts = set(pd.Series(np.asarray(split["y_train"]).ravel()).value_counts())
    assert len(train_counts) == 1, "training split should be balanced by oversampling"

    for key in ("y_val", "y_test"):
        counts = set(pd.Series(np.asarray(split[key]).ravel()).value_counts())
        assert len(counts) > 1, f"{key} must keep the original imbalance"


def test_sensitive_column_is_not_carried_when_unconfigured(imbalanced_frame):
    split = _split(imbalanced_frame, "oversample")
    assert split["sensitive_train"] is None
    assert split["sensitive_val"] is None
