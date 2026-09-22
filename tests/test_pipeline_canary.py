"""End-to-end canary on a dataset whose answer is known.

Banknote Authentication is linearly separable: any correct pipeline scores
~1.0. That makes it a detector for the whole class of bugs that do not raise
and do not fail a unit test, but quietly make the analysed model different
from the model the search selected:

- a validation split contaminated by resampled duplicates,
- an analysis refit on the wrong frame,
- a training target with the wrong shape,
- a decision threshold replayed onto an incompatible probability scale,
- a model with no usable ``predict_proba`` silently nulling every
  probability metric.

Each of those was a real defect here. Every one of them moves a number below.
"""

import numpy as np
import pytest

from quoptuna.server.services.database import get_engine
from quoptuna.server.services.headless import run_headless_optimization

BANKNOTE_CSV = "data/Banknote Authentication.csv"
# Separable data: anything below this means the pipeline lost information the
# model had. Not a tuning target — a correctness floor.
PERFECT = 0.99


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Keep the canary out of the developer's real databases (SPEC-005)."""
    monkeypatch.setattr(
        "quoptuna.backend.utils.storage.optuna_db_path", lambda name: tmp_path / name
    )
    monkeypatch.setattr("quoptuna.server.services.run_store.APP_DB_PATH", tmp_path / "app.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/app.db")
    get_engine.cache_clear()
    yield tmp_path
    get_engine.cache_clear()


@pytest.fixture
def banknote_run(isolated_storage):
    return run_headless_optimization(
        csv_path=BANKNOTE_CSV,
        target="class",
        n_trials=3,
        model_types=["SVC"],
        search_space={"C": [1.0], "gamma": [0.1], "class_weight": [None]},
        sampler="random",
        sampler_seed=0,
        db_name="canary",
        study_name="banknote_canary",
        subset_size=10,
    )


def test_separable_data_scores_near_perfectly(banknote_run):
    metrics = banknote_run["analysis"]["metrics"]

    assert metrics["f1_score"] >= PERFECT
    assert metrics["accuracy"] >= PERFECT


def test_probability_metrics_are_available(banknote_run):
    """A null ROC-AUC means the analysed model exposed no probabilities.

    SVC ships ``probability=False``; without the fixed constructor kwargs that
    override it, every probability metric is null whenever an SVC wins.
    """
    assert banknote_run["analysis"]["metrics"]["roc_auc_score"] is not None
    assert banknote_run["analysis"]["metrics"]["roc_auc_score"] >= PERFECT


def test_analysis_reproduces_the_selected_trial(banknote_run):
    """The search and the analysis measure the same thing; they must agree.

    This is the assertion that catches a refit which silently differs from the
    trial — wrong data, wrong target shape, wrong decision rule. Each of those
    leaves the other metrics looking superficially plausible.
    """
    consistency = banknote_run["analysis"]["refit_consistency"]

    assert consistency is not None, "trial recorded no test F1 to compare against"
    assert consistency["within_tolerance"], (
        f"analysis F1 {consistency['analysis_test_f1']:.3f} does not reproduce the trial's "
        f"{consistency['trial_test_f1']:.3f} (drift {consistency['drift']:.3f})"
    )


def test_confusion_matrix_uses_both_classes(banknote_run):
    """A single populated column is the signature of a degenerate decision rule."""
    matrix = np.asarray(banknote_run["analysis"]["confusion_matrix"]["matrix"])

    predicted_per_class = matrix.sum(axis=0)
    assert (predicted_per_class > 0).all(), f"predicted only one class: {matrix.tolist()}"


def _run(db_name, study_name):
    return run_headless_optimization(
        csv_path=BANKNOTE_CSV,
        target="class",
        n_trials=2,
        # The only shot-based model in the catalogue: its feature map samples
        # from a simulator device, so an unseeded device makes every run —
        # and every re-analysis — return different numbers.
        model_types=["QuantumKitchenSinks"],
        search_space={"max_vmap": [32], "n_qfeatures": ["full"], "n_episodes": [10]},
        sampler="random",
        sampler_seed=0,
        db_name=db_name,
        study_name=study_name,
        subset_size=5,
    )


def test_identical_runs_produce_identical_scores(isolated_storage):
    """The reported symptom: scores moved on every re-analysis.

    Two independent causes, both fixed. The shared-estimator leak between
    model instances is pinned by ``tests/test_model_isolation.py``; this
    covers the other one — sampling from an unseeded simulator device, which
    made even identical inputs produce different predictions.
    """
    first = _run("determinism_a", "banknote_det_a")["analysis"]["metrics"]
    second = _run("determinism_b", "banknote_det_b")["analysis"]["metrics"]

    assert first == second
