"""Integration tests for the Optimization Configuration case (UCI Banknote).

These exercise the real optimization engine end to end with a low trial count
so that any per-trial model failure (jax-jit ``legacy_vectorized``, the MLP
``[100,)`` literal, or the column-vector label/chex issue) surfaces as a test
failure instead of being silently swallowed as ``value: 0.0``.
"""

import math

import pytest
from optuna import load_study
from optuna.trial import TrialState

from quoptuna.backend.tuners.optimizer import Optimizer
from quoptuna.server.api.v1.optimize import (
    OptimizationRequest,
    optimization_jobs,
    run_optimization_background,
)
from quoptuna.server.services import dataset_registry

from .conftest import TEST_MODEL_TYPES, TEST_SEARCH_SPACE

STUDY_NAME = "my-optimization-study-test-1"
DATABASE_NAME = "results-test1"
SELECTED_FEATURES = ["entropy", "curtosis", "skewness", "variance"]


@pytest.mark.slow
def test_optimization_configuration_full_pipeline(
    banknote_csv, fast_optimizer_training, monkeypatch, tmp_path
):
    """5a: drive run_optimization_background for the exact UI configuration."""
    monkeypatch.chdir(tmp_path)

    dataset_id = "banknote-int-test"
    dataset_registry.register(
        {
            "id": dataset_id,
            "name": "Banknote Authentication (test)",
            "source": "upload",
            "file_path": str(banknote_csv),
            "rows": 120,
            "columns": [*SELECTED_FEATURES, "class"],
        }
    )

    request = OptimizationRequest(
        dataset_id=dataset_id,
        dataset_source="upload",
        selected_features=SELECTED_FEATURES,
        target_column="class",
        study_name=STUDY_NAME,
        database_name=DATABASE_NAME,
        num_trials=5,
        model_types=TEST_MODEL_TYPES,
        search_space=TEST_SEARCH_SPACE,
    )

    job_id = "opt_inttest"
    optimization_jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "current_trial": 0,
        "total_trials": request.num_trials,
        "best_value": None,
        "best_params": None,
        "trials": None,
        "started_at": "",
        "completed_at": None,
        "error": None,
        "request": request.dict(),
    }

    run_optimization_background(job_id, request)

    job = optimization_jobs[job_id]
    assert job["status"] == "completed", job.get("error")
    assert job["error"] is None
    assert isinstance(job["best_value"], float)
    assert 0.0 < job["best_value"] <= 1.0
    assert "model_type" in job["best_params"]
    assert job["trials"]
    assert any(t["value"] > 0 for t in job["trials"])

    # Regression (classical models vanishing on completion): the stored trials
    # must be the real per-trial history serialized from the Optuna study, not
    # rows synthesized from best_params — real rows carry state/user_attrs and
    # unique trial numbers, and each keeps its own model_type.
    assert all("state" in t for t in job["trials"])
    trial_numbers = [t["trial"] for t in job["trials"]]
    assert len(trial_numbers) == len(set(trial_numbers))
    tried_models = {t["params"].get("model_type") for t in job["trials"]}
    study_models = {
        t.params.get("model_type")
        for t in load_study(
            storage=f"sqlite:///db/{DATABASE_NAME}.db", study_name=STUDY_NAME
        ).trials
        if t.state == TrialState.COMPLETE
    }
    assert study_models <= tried_models

    # The Optuna study must have been persisted to db/<database_name>.db.
    study = load_study(
        storage=f"sqlite:///db/{DATABASE_NAME}.db",
        study_name=STUDY_NAME,
    )
    assert len(study.trials) == request.num_trials


@pytest.mark.slow
def test_optimizer_ten_trials_on_banknote(
    preprocessed_banknote, fast_optimizer_training, monkeypatch, tmp_path
):
    """5a-bis: run the Optimizer directly for 10 trials on preprocessed data."""
    monkeypatch.chdir(tmp_path)

    optimizer = Optimizer(
        db_name="results-test1-10trials",
        data=preprocessed_banknote,
        study_name="my-optimization-study-test-10trials",
        model_types=TEST_MODEL_TYPES,
        search_space=TEST_SEARCH_SPACE,
    )

    study, _best = optimizer.optimize(n_trials=10)

    assert len(study.trials) == 10  # noqa: PLR2004
    # With a pruner configured trials may also legitimately end PRUNED.
    assert all(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in study.trials)

    assert study.best_value is not None
    assert math.isfinite(study.best_value)
    assert 0.0 < study.best_value <= 1.0
    assert "model_type" in study.best_trial.params

    # Per-model logging (quantum/classical attrs) should be exercised.
    assert any(t.user_attrs for t in study.trials)


@pytest.mark.slow
def test_optimizer_with_asha_pruner_on_banknote(
    preprocessed_banknote, fast_optimizer_training, monkeypatch, tmp_path
):
    """ASHA + random sampler end to end: trials finish COMPLETE or PRUNED,
    pruned trials carry resource attrs, and best_trial is a completed one."""
    monkeypatch.chdir(tmp_path)

    optimizer = Optimizer(
        db_name="results-test-asha",
        data=preprocessed_banknote,
        study_name="my-optimization-study-test-asha",
        model_types=["DataReuploadingClassifier", "SVC"],
        search_space=TEST_SEARCH_SPACE,
        sampler="random",
        sampler_seed=7,
        pruner="asha",
        pruner_min_resource=1,
        pruner_reduction_factor=2,
        intermediate_metric="accuracy",
    )

    study, _best = optimizer.optimize(n_trials=8)

    states = {t.state for t in study.trials}
    assert states <= {TrialState.COMPLETE, TrialState.PRUNED}
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    assert completed, "ASHA pruned every trial"
    assert study.best_trial.state == TrialState.COMPLETE
    for t in study.trials:
        if t.state == TrialState.PRUNED:
            assert t.user_attrs.get("pruned") is True
            assert "n_steps" in t.user_attrs
