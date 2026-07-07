"""Tests for categorical feature encoding and failed-trial handling."""

import numpy as np
import pandas as pd
import pytest
from optuna.trial import TrialState

from quoptuna import DataPreparation
from quoptuna.backend.tuners.optimizer import Optimizer
from quoptuna.backend.utils.data_utils.prepare import (
    MAX_ONEHOT_CATEGORIES,
    MAX_ORDINAL_CATEGORIES,
    encode_features,
    encoded_passthrough_columns,
)
from quoptuna.server.services.workflow_service import WorkflowExecutionError, WorkflowExecutor


def test_encode_features_one_hot_and_impute():
    df = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0],
            "cat": ["a", "b", None, "a"],
            "flag": [True, False, True, False],
        }
    )
    encoded, meta = encode_features(df, method="onehot")

    assert list(encoded.index) == list(df.index)
    assert len(encoded) == len(df)
    # Numeric imputed with median.
    expected_median = 3.0
    assert encoded["num"].iloc[1] == expected_median
    assert meta["num"] == {"kind": "numeric", "imputed": 1}
    # Categorical one-hot with NaN as its own "missing" category.
    assert {"cat_a", "cat_b", "cat_missing"} <= set(encoded.columns)
    assert encoded["cat_missing"].iloc[2] == 1.0
    assert meta["cat"]["kind"] == "onehot"
    assert meta["cat"]["imputed"] == 1
    # Bool cast to float.
    assert encoded["flag"].dtype == float
    # Everything numeric — StandardScaler-safe.
    assert all(pd.api.types.is_numeric_dtype(encoded[c]) for c in encoded.columns)
    assert not encoded.isna().any().any()


def test_encode_features_rejects_high_cardinality():
    df = pd.DataFrame({"id": [f"user_{i}" for i in range(MAX_ONEHOT_CATEGORIES + 5)]})
    with pytest.raises(ValueError, match="id"):
        encode_features(df, method="onehot")
    # Same column is fine under ordinal (single bounded column).
    encoded, _meta = encode_features(df, method="ordinal")
    assert list(encoded.columns) == ["id"]
    df_huge = pd.DataFrame({"id": [f"u{i}" for i in range(MAX_ORDINAL_CATEGORIES + 1)]})
    with pytest.raises(ValueError, match="id"):
        encode_features(df_huge, method="ordinal")


def test_encode_features_ordinal():
    df = pd.DataFrame({"cat": ["b", "a", None, "c", "a"], "num": [1.0, 2.0, 3.0, 4.0, 5.0]})
    encoded, meta = encode_features(df, method="ordinal")

    assert list(encoded.columns) == ["cat", "num"]
    assert meta["cat"]["kind"] == "ordinal"
    assert meta["cat"]["categories"] == ["a", "b", "c", "missing"]
    # Codes scaled to [0, 1]; sorted-category order is deterministic.
    assert encoded["cat"].between(0, 1).all()
    assert encoded["cat"].iloc[1] == 0.0  # "a" -> code 0
    assert encoded["cat"].iloc[3] == pytest.approx(2 / 3)  # "c"
    assert encoded["cat"].iloc[2] == 1.0  # "missing" (last)
    assert encoded_passthrough_columns(meta) == ["cat"]


def test_data_preparation_passthrough_columns_skip_scaler():
    rng = np.random.default_rng(4)
    n = 60
    x = pd.DataFrame({"num": rng.normal(loc=50, scale=10, size=n), "cat": rng.integers(0, 2, n)})
    x["cat"] = x["cat"].astype(float)  # emulate an encoded 0/1 column
    y = pd.Series(rng.choice([0, 1], size=n))

    prep = DataPreparation(
        dataset={"x": x, "y": y},
        x_cols=["num", "cat"],
        y_col="t",
        passthrough_columns=["cat"],
    )
    combined = pd.concat([prep.x_train, prep.x_test]).sort_index()
    # Passthrough column untouched; numeric column z-scored (mean ~0).
    zscored_mean_tolerance = 0.2
    assert set(np.unique(combined["cat"])) <= {0.0, 1.0}
    assert abs(combined["num"].mean()) < zscored_mean_tolerance
    assert list(combined.columns) == ["num", "cat"]


def test_feature_selection_encodes_categoricals():
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "group": rng.choice(["x", "y", "z"], size=n),
            "target": rng.choice(["yes", "no"], size=n),
        }
    )
    executor = WorkflowExecutor({"nodes": [], "edges": []})
    result = executor._execute_feature_selection(
        {"x_columns": ["f1", "group"], "y_column": "target", "categorical_encoding": "onehot"},
        {"in": {"type": "dataset", "dataframe": df}},
    )
    assert {"group_x", "group_y", "group_z"} <= set(result["x"].columns)
    assert result["x_columns"] == list(result["x"].columns)
    assert result["encoding"]["group"]["kind"] == "onehot"
    assert set(result["passthrough_columns"]) == {"group_x", "group_y", "group_z"}
    # Row positions preserved for downstream raw-index alignment.
    assert list(result["x"].index) == list(df.index)

    # Default method is ordinal: one column per feature.
    ordinal = executor._execute_feature_selection(
        {"x_columns": ["f1", "group"], "y_column": "target"},
        {"in": {"type": "dataset", "dataframe": df}},
    )
    assert list(ordinal["x"].columns) == ["f1", "group"]
    assert ordinal["encoding"]["group"]["kind"] == "ordinal"
    assert ordinal["passthrough_columns"] == ["group"]


def test_feature_selection_high_cardinality_is_workflow_error():
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(50)],
            "target": ["yes", "no"] * 25,
        }
    )
    executor = WorkflowExecutor({"nodes": [], "edges": []})
    with pytest.raises(WorkflowExecutionError, match="id"):
        executor._execute_feature_selection(
            {"x_columns": ["id"], "y_column": "target", "categorical_encoding": "onehot"},
            {"in": {"type": "dataset", "dataframe": df}},
        )


def test_failed_trials_are_marked_failed_with_reason(tmp_path):
    """A raising objective must produce FAILED trials with an error attr, not F1=0."""
    # String features guarantee model.fit raises inside the objective.
    data = {
        "train_x": np.array([["a"], ["b"]], dtype=object),
        "train_y": np.array([-1, 1]),
        "test_x": np.array([["a"]], dtype=object),
        "test_y": np.array([1]),
    }
    optimizer = Optimizer(
        db_name=str(tmp_path / "fail_test.db"),
        data=data,
        study_name="fail-test",
        model_types=["MLPClassifier"],
    )
    n_trials = 2
    study, _ = optimizer.optimize(n_trials=n_trials)

    assert len(study.trials) == n_trials
    for trial in study.trials:
        assert trial.state == TrialState.FAIL
        assert "error" in trial.user_attrs
        assert trial.value is None
