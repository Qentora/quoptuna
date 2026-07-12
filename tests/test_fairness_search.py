"""Tests for the fairness-aware search (constrained + multi-objective modes).

Follows the ``test_optimizer.py`` pattern: ``create_model`` is stubbed with a
fake estimator so these exercise the ``Optimizer`` fairness control flow
(validation, constraint attrs, multi-objective directions) without training.
"""

import numpy as np
import optuna
import pytest
from pydantic import ValidationError

from quoptuna.backend.tuners import optimizer as optimizer_module
from quoptuna.backend.tuners.optimizer import Optimizer
from quoptuna.backend.xai import fairness as fm
from quoptuna.server.api.v1.optimize import OptimizationRequest

TINY_SEARCH_SPACE = {"C": [1.0]}
N_TRIALS = 3
DI_THRESHOLD_DEFAULT = 0.8
EXPECTED_DISPARITY = 0.5
N_OBJECTIVES = 2


class _BiasedModel:
    """Predicts +1 for the first half of rows and -1 for the rest, so a
    sensitive column split the same way yields a deterministic disparity."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        n = len(x)
        return np.array([1] * (n // 2) + [-1] * (n - n // 2))

    def score(self, _x, _y):
        return 0.5


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 4))
    labels = np.array([1, -1] * 10)
    return {
        "train_x": features,
        "test_x": features,
        "train_y": labels,
        "test_y": labels,
    }


@pytest.fixture
def sensitive_test():
    # First half group "a" (all predicted +1), second half "b" (all -1):
    # maximal demographic-parity disparity.
    return np.array(["a"] * 10 + ["b"] * 10)


@pytest.fixture(autouse=True)
def _in_tmp_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_create_model(monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _BiasedModel(model_type=model_type, **kwargs),
    )


# --- compute_disparity -------------------------------------------------------


def test_compute_disparity_all_metrics():
    y_true = np.array([1, -1, 1, -1, 1, 1, -1, -1])
    y_pred = np.array([1, -1, 1, 1, 1, -1, -1, -1])
    sens = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    # Selection rates: a=0.75, b=0.25 -> DPD 0.5, DI ratio 1/3 -> disparity 2/3.
    assert (
        fm.compute_disparity(y_true, y_pred, sens, "demographic_parity_difference")
        == EXPECTED_DISPARITY
    )
    assert fm.compute_disparity(y_true, y_pred, sens, "disparate_impact") == pytest.approx(2 / 3)
    # TPRs: a=2/2=1.0, b: positives predicted 1 of 2 -> 0.5 -> EOD 0.5.
    assert (
        fm.compute_disparity(y_true, y_pred, sens, "equal_opportunity_difference")
        == EXPECTED_DISPARITY
    )


def test_compute_disparity_zero_when_parity():
    y = np.array([1, -1, 1, -1])
    sens = np.array(["a", "a", "b", "b"])
    for metric in fm.FAIRNESS_METRICS:
        assert fm.compute_disparity(y, y, sens, metric) == pytest.approx(0.0)


def test_compute_disparity_unknown_metric_raises():
    y = np.array([1, -1])
    with pytest.raises(ValueError, match="Unknown fairness metric"):
        fm.compute_disparity(y, y, np.array(["a", "b"]), "accuracy")


def test_compute_fairness_includes_di_and_eod():
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([1, 1, 1, -1])
    sens = np.array(["a", "a", "b", "b"])
    disparities = fm.compute_fairness(y_true, y_pred, sens)["disparities"]
    assert "disparate_impact" in disparities
    assert "equal_opportunity_difference" in disparities
    assert disparities["disparate_impact"] == disparities["demographic_parity_ratio"]


# --- Optimizer validation ----------------------------------------------------


def test_fairness_mode_requires_sensitive_test():
    with pytest.raises(ValueError, match="requires sensitive_test"):
        Optimizer(db_name="unit", fairness_mode="constrained")


def test_constrained_requires_tpe(sensitive_test):
    with pytest.raises(ValueError, match="requires sampler='tpe'"):
        Optimizer(
            db_name="unit",
            fairness_mode="constrained",
            sampler="random",
            sensitive_test=sensitive_test,
        )


def test_multi_objective_rejects_pruner(sensitive_test):
    with pytest.raises(ValueError, match="does not support pruning"):
        Optimizer(
            db_name="unit",
            fairness_mode="multi_objective",
            pruner="asha",
            sensitive_test=sensitive_test,
        )


def test_unknown_fairness_metric_rejected(sensitive_test):
    with pytest.raises(ValueError, match="Unknown fairness_metric"):
        Optimizer(
            db_name="unit",
            fairness_mode="constrained",
            fairness_metric="accuracy",
            sensitive_test=sensitive_test,
        )


def test_threshold_defaults_are_metric_specific(sensitive_test):
    diff = Optimizer(db_name="u1", fairness_mode="constrained", sensitive_test=sensitive_test)
    assert diff._disparity_threshold == pytest.approx(0.1)
    di = Optimizer(
        db_name="u2",
        fairness_mode="constrained",
        fairness_metric="disparate_impact",
        sensitive_test=sensitive_test,
    )
    # DI ratio threshold 0.8 -> disparity-space threshold 0.2.
    assert di._disparity_threshold == pytest.approx(1 - DI_THRESHOLD_DEFAULT)


# --- constrained mode --------------------------------------------------------


def test_constrained_study_records_constraint_attrs(tiny_data, sensitive_test, fake_create_model):
    opt = Optimizer(
        db_name="unit_constrained",
        study_name="constrained_study",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
        fairness_mode="constrained",
        fairness_metric="demographic_parity_difference",
        fairness_threshold=0.1,
        sensitive_test=sensitive_test,
    )
    sampler = opt._build_sampler()
    assert sampler._constraints_func is not None

    study, _ = opt.optimize(n_trials=N_TRIALS)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == N_TRIALS
    for t in completed:
        # _BiasedModel selects all of group "a", none of "b" -> disparity 1.0,
        # violation 1.0 - 0.1 = 0.9 (> 0 means infeasible).
        assert t.user_attrs["fairness_disparity"] == pytest.approx(1.0)
        assert t.user_attrs["fairness_constraint"][0] == pytest.approx(0.9)
        assert t.user_attrs["fairness_metric"] == "demographic_parity_difference"


def test_constraints_func_defaults_to_worst_for_missing_attr(sensitive_test):
    class _Frozen:
        def __init__(self):
            self.user_attrs = {}

    assert Optimizer._fairness_constraints(_Frozen()) == (fm.WORST_DISPARITY,)


# --- multi-objective mode ----------------------------------------------------


def test_multi_objective_study(tiny_data, sensitive_test, fake_create_model):
    opt = Optimizer(
        db_name="unit_mo",
        study_name="mo_study",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
        fairness_mode="multi_objective",
        sensitive_test=sensitive_test,
    )
    study, best_trials = opt.optimize(n_trials=N_TRIALS)
    assert len(study.directions) == N_OBJECTIVES
    assert best_trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for t in completed:
        assert len(t.values) == N_OBJECTIVES
        assert 0.0 <= t.values[0] <= 1.0
        assert t.values[1] == t.user_attrs["fairness_disparity"]


def test_failed_disparity_scores_worst(tiny_data, sensitive_test, fake_create_model, monkeypatch):
    def _boom(*_args, **_kwargs):
        msg = "degenerate group"
        raise ValueError(msg)

    monkeypatch.setattr(optimizer_module, "compute_disparity", _boom)
    opt = Optimizer(
        db_name="unit_worst",
        study_name="worst_study",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
        fairness_mode="multi_objective",
        sensitive_test=sensitive_test,
    )
    study, _ = opt.optimize(n_trials=1)
    (trial,) = study.trials
    assert trial.user_attrs["fairness_disparity"] == fm.WORST_DISPARITY
    assert trial.values[1] == fm.WORST_DISPARITY


def test_off_mode_unchanged(tiny_data, fake_create_model):
    opt = Optimizer(
        db_name="unit_off",
        study_name="off_study",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    study, _ = opt.optimize(n_trials=1)
    assert len(study.directions) == 1
    (trial,) = study.trials
    assert "fairness_disparity" not in trial.user_attrs


# --- request-level validation ------------------------------------------------


def test_optimization_request_fairness_validation():
    base = {
        "dataset_id": "d",
        "dataset_source": "upload",
        "selected_features": ["x"],
        "target_column": "y",
        "study_name": "s",
        "num_trials": 1,
    }
    with pytest.raises(ValidationError, match="requires sensitive_feature"):
        OptimizationRequest(**base, fairness_mode="constrained")
    with pytest.raises(ValidationError, match="requires sampler='tpe'"):
        OptimizationRequest(
            **base, fairness_mode="constrained", sensitive_feature="g", sampler="grid"
        )
    with pytest.raises(ValidationError, match="does not support pruning"):
        OptimizationRequest(
            **base, fairness_mode="multi_objective", sensitive_feature="g", pruner="hyperband"
        )
    ok = OptimizationRequest(**base, fairness_mode="multi_objective", sensitive_feature="g")
    assert ok.fairness_metric == "equal_opportunity_difference"
