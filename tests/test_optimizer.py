"""Fast unit tests for ``quoptuna.backend.tuners.optimizer``.

These avoid real quantum/classical fits by stubbing ``create_model`` with a
tiny fake estimator, so they exercise the ``Optimizer`` control flow (sampling,
scoring, user-attribute logging, study creation/loading and ``optimize``)
without the cost of training. ``FixedTrial`` lets us drive ``objective``
deterministically through both the classical and quantum logging branches.
"""

import numpy as np
import pytest
from optuna.trial import FixedTrial

from quoptuna.backend.tuners import optimizer as optimizer_module
from quoptuna.backend.tuners.optimizer import (
    DEFAULT_MODEL_TYPES,
    DEFAULT_SEARCH_SPACE,
    Optimizer,
)

TINY_SEARCH_SPACE = {"C": [1.0]}
EXPECTED_TRIALS = 2


class _FakeModel:
    """Minimal sklearn-like estimator that never actually trains."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        # Alternate labels so both classes appear and f1_score is well-defined.
        return np.resize(np.array([0, 1]), len(x))

    def score(self, _x, _y):
        return 0.5


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 4))
    labels = np.array([0, 1] * 10)
    return {
        "train_x": features,
        "test_x": features,
        "train_y": labels,
        "test_y": labels,
    }


@pytest.fixture(autouse=True)
def _in_tmp_cwd(monkeypatch, tmp_path):
    # Optimizer.__init__ creates a ./db directory; keep it inside tmp_path.
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_create_model(monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeModel(model_type=model_type, **kwargs),
    )


def test_defaults_are_consistent():
    assert "MLPClassifier" in DEFAULT_MODEL_TYPES
    # Regression guard for the MLP literal fix (must be a valid tuple literal).
    assert "(100,)" in DEFAULT_SEARCH_SPACE["hidden_layer_sizes"]


def test_init_with_dataset_name_sets_data_path():
    opt = Optimizer(db_name="unit", dataset_name="banknote")
    assert opt.db_name == "unit"
    assert opt.data_path.endswith("unit.db")
    assert opt.model_types == DEFAULT_MODEL_TYPES
    assert opt.search_space == DEFAULT_SEARCH_SPACE


def test_init_with_overrides():
    opt = Optimizer(
        db_name="unit",
        data={"train_x": [1]},
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    assert opt.model_types == ["SVC"]
    assert opt.search_space == TINY_SEARCH_SPACE
    assert opt.train_x == [1]


@pytest.mark.parametrize(
    ("model_type", "quantum"),
    [("SVC", False), ("DataReuploadingClassifier", True)],
)
def test_objective_scores_and_logs(tiny_data, fake_create_model, model_type, quantum):
    opt = Optimizer(
        db_name="unit",
        data=tiny_data,
        model_types=[model_type],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": model_type})
    value = opt.objective(trial)

    assert 0.0 <= value <= 1.0
    attrs = trial.user_attrs
    if quantum:
        assert attrs["Classical_accuracy"] == 0
        assert "Quantum_accuracy" in attrs
    else:
        assert attrs["Quantum_accuracy"] == 0
        assert "Classical_accuracy" in attrs


def test_objective_reraises_on_error(tiny_data, monkeypatch):
    """Errors propagate (study.optimize's `catch` marks the trial FAILED) and
    the reason is recorded, instead of silently scoring 0.0."""

    def _boom(*_args, **_kwargs):
        msg = "model construction failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(optimizer_module, "create_model", _boom)
    opt = Optimizer(
        db_name="unit",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": "SVC"})
    with pytest.raises(RuntimeError, match="model construction failed"):
        opt.objective(trial)
    assert "model construction failed" in trial.user_attrs["error"]


def test_optimize_then_load_study(tiny_data, fake_create_model):
    opt = Optimizer(
        db_name="unit_opt",
        study_name="unit_study",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    study, best_trials = opt.optimize(n_trials=EXPECTED_TRIALS)

    assert opt.study is study
    assert len(study.trials) == EXPECTED_TRIALS
    assert best_trials is not None

    # Re-loading the persisted study should not raise and points at the same run.
    opt.study = None
    opt.load_study()
    assert opt.study is not None
    assert opt.study.study_name == "unit_study"


def test_create_or_load_study_is_idempotent(fake_create_model):
    opt = Optimizer(db_name="unit_idem", study_name="reused")
    first = opt._create_or_load_study()
    # ``load_if_exists`` means a second call reuses the same study instead of raising.
    second = opt._create_or_load_study()
    assert first.study_name == second.study_name == "reused"
