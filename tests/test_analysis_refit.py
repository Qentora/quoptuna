"""The analysed model must be the model the search selected.

Two failures this pins, both observed on `new_ilpd_no_faireness_100trial`
(trial 65, val F1 0.64) whose analysis reported F1 0.000 with a confusion
matrix that predicted no positives at all:

1. `build_xai` fitted on a `(n, 1)` target because the label-encoding node
   stores y as a DataFrame and `.values` does not ravel. Training does not
   raise on that shape — it collapses the model's probabilities into a narrow
   band around 0.5, which leaves argmax predictions plausible while making any
   probability threshold meaningless.
2. The trial's `decision_threshold` is chosen against the trial's fit and
   applied to a later refit. When it falls outside that fit's probability
   range, every row lands on one side and every label metric reads 0 for a
   model that is not degenerate.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from quoptuna import XAI, XAIConfig
from quoptuna.server.services import workflow_service

N_ROWS = 40


class _RecordingModel:
    """Records the target shape it was fitted with; predicts by threshold."""

    classes_ = np.array([-1, 1])

    def __init__(self, **kwargs):
        self.fitted_y_shape = None
        self.kwargs = kwargs

    def fit(self, x, y):
        self.fitted_y_shape = np.asarray(y).shape
        return self

    def predict(self, x):
        return np.where(np.asarray(x)[:, 0] > 0, 1, -1)

    def predict_proba(self, x):
        # Compressed around 0.5, exactly like a model trained on a 2-D target.
        p = 0.5 + 0.02 * np.sign(np.asarray(x)[:, 0])
        return np.column_stack([1 - p, p])


@pytest.fixture
def opt_result(tmp_path, monkeypatch):
    """A completed one-trial study plus a split whose y is a DataFrame."""
    storage_url = f"sqlite:///{tmp_path}/refit.db"
    # build_xai resolves the storage lazily, inside the call.
    monkeypatch.setattr(
        "quoptuna.server.services.storage.optuna_storage_url", lambda name: storage_url
    )
    study = optuna.create_study(
        storage=storage_url, study_name="refit", direction="maximize"
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"model_type": "SVC", "C": 1.0},
            distributions={
                "model_type": optuna.distributions.CategoricalDistribution(["SVC"]),
                "C": optuna.distributions.CategoricalDistribution([1.0]),
            },
            value=0.64,
            user_attrs={"decision_threshold": 0.55},
        )
    )

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(N_ROWS, 3)), columns=["a", "b", "c"])
    y = np.where(x["a"] > 0, 1, -1)
    return {
        "db_name": "refit.db",
        "study_name": "refit",
        # The label-encoding node emits a DataFrame, which is the shape trap.
        "x_train": x,
        "y_train": pd.DataFrame({"target": y}),
        "x_test": x,
        "y_test": pd.DataFrame({"target": y}),
    }


def test_refit_target_is_one_dimensional(opt_result, monkeypatch):
    """A (n, 1) target trains silently and ruins the probability scale."""
    created = {}

    def _create_model(model_type, **kwargs):
        created["model"] = _RecordingModel(**kwargs)
        return created["model"]

    monkeypatch.setattr("quoptuna.backend.models.create_model", _create_model)

    xai = workflow_service.build_xai(opt_result)

    assert created["model"].fitted_y_shape == (N_ROWS,)
    assert xai is not None


def test_threshold_outside_the_probability_range_is_discarded():
    """Never report an all-one-class score for a model that predicts both."""
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(N_ROWS, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.where(x["a"] > 0, 1, -1))
    model = _RecordingModel()
    data = {"x_train": x, "y_train": y, "x_test": x, "y_test": y}

    # 0.55 is above every probability this model produces (max 0.52).
    xai = XAI(model=model, data=data, config=XAIConfig(decision_threshold=0.55))
    predictions = np.asarray(xai.predictions)

    assert len(np.unique(predictions)) > 1, "fell back to a degenerate single-class rule"
    assert xai.decision_threshold is None
    assert "outside this fit's probability range" in xai.threshold_discarded


def test_threshold_inside_the_range_is_applied():
    usable_threshold = 0.5
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(N_ROWS, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.where(x["a"] > 0, 1, -1))
    model = _RecordingModel()
    data = {"x_train": x, "y_train": y, "x_test": x, "y_test": y}

    # 0.5 sits inside [0.48, 0.52], so the cutoff is meaningful and kept.
    xai = XAI(model=model, data=data, config=XAIConfig(decision_threshold=usable_threshold))

    assert xai.threshold_discarded is None
    assert xai.decision_threshold == usable_threshold
