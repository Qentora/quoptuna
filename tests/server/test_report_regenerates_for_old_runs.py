"""Regenerating a report for a pre-existing run must recover its full context.

The realistic case for an older run is that the in-memory job is gone: only
``run_store`` (which holds the optimization request) and the Optuna study on disk
survive. The report path must rebuild the configuration, the trial history and
the Pareto front from those two sources rather than quietly reporting a
fairness-aware multi-objective search as an unconfigured single-objective one.
"""

from __future__ import annotations

import asyncio

import optuna
import pytest

from quoptuna.backend.utils import storage as storage_mod
from quoptuna.backend.xai import report_agent
from quoptuna.server.api.v1 import analysis, optimize
from quoptuna.server.services import analysis_store, run_store

STUDY_NAME = "adult-fair"
# One shared distribution per parameter: Optuna rejects a categorical whose
# choices differ between trials of the same study.
MODEL_DIST = optuna.distributions.CategoricalDistribution(["DataReuploadingClassifier", "SVC"])
LAYER_DIST = optuna.distributions.IntDistribution(1, 3)

# (F1, disparity, model). Only the third trial is feasible at threshold 0.1.
TRIALS = [
    (0.81, 0.31, "DataReuploadingClassifier"),
    (0.87, 0.27, "DataReuploadingClassifier"),
    (0.62, 0.08, "SVC"),
]

REQUEST = {
    "dataset_id": "2",
    "dataset_source": "uci",
    "selected_features": ["age"],
    "target_column": "salary",
    "study_name": STUDY_NAME,
    "database_name": "results",
    "num_trials": 3,
    "sampler": "tpe",
    "pruner": "none",
    "resampling": "oversample",
    "categorical_encoding": "onehot",
    "dev_type": "lightning.qubit",
    "max_vmap": 32,
    "sensitive_feature": "sex",
    "fairness_mode": "multi_objective",
    "fairness_metric": "equal_opportunity_difference",
    "fairness_threshold": 0.1,
}


@pytest.fixture
def existing_run(tmp_path, monkeypatch):
    """A finished multi-objective run that survives only on disk."""
    monkeypatch.setattr(run_store, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(analysis_store, "ARTIFACT_ROOT", tmp_path / "analysis")
    monkeypatch.setattr(storage_mod, "DB_DIR", tmp_path / "db")
    storage_mod.DB_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        storage=storage_mod.optuna_storage_url("results"),
        study_name=STUDY_NAME,
        directions=["maximize", "minimize"],
    )
    for f1, disparity, model in TRIALS:
        study.add_trial(
            optuna.trial.create_trial(
                params={"model_type": model, "n_layers": 2},
                distributions={"model_type": MODEL_DIST, "n_layers": LAYER_DIST},
                values=[f1, disparity],
                user_attrs={"fairness_disparity": disparity, "training_time": 4.0},
            )
        )

    run_store.save_run(
        {
            "id": "opt_old",
            "status": "completed",
            "started_at": "2026-09-01T09:00:00",
            "completed_at": "2026-09-01T09:30:00",
            "best_value": 0.87,
            "best_params": {"model_type": "DataReuploadingClassifier", "n_layers": 2},
            "request": REQUEST,
        }
    )
    # A restarted backend has no hot job cache and no in-memory "result" blob.
    monkeypatch.setattr(optimize, "optimization_jobs", {})

    job = analysis_store.create_job("opt_old", {"trial_number": None})
    analysis_store.complete_job(
        job["id"], {"metrics": {"f1_score": 0.87, "accuracy": 0.9}, "plots": {}, "warnings": {}}
    )
    return analysis_store.get_snapshot(job["snapshot_id"])


def regenerate(snapshot, monkeypatch) -> dict:
    captured = {}

    async def fake_generate_report(**kwargs):
        captured["context"] = kwargs["context"]
        return {"markdown": "# Regenerated\n", "lint": [], "reviewed": True}

    monkeypatch.setattr(report_agent, "generate_report", fake_generate_report)
    asyncio.run(
        analysis.generate_ai_report(
            analysis.ReportRequest(
                optimization_id="opt_old",
                analysis_snapshot_id=snapshot["id"],
                analysis_revision=snapshot["revision"],
                api_key="k",
                llm_provider="openai",
                model_name="m",
            )
        )
    )
    return captured["context"]


def test_training_options_are_recovered_from_the_persisted_request(existing_run, monkeypatch):
    configuration = regenerate(existing_run, monkeypatch)["configuration"]
    assert configuration["data"]["resampling"] == "oversample"
    assert configuration["data"]["categorical_encoding"] == "onehot"
    assert configuration["search"]["sampler"] == "tpe"
    assert configuration["search"]["num_trials"] == len(TRIALS)
    assert configuration["training"]["dev_type"] == "lightning.qubit"
    assert configuration["training"]["max_vmap"] == 32  # noqa: PLR2004


def test_trial_history_is_reread_from_the_study_on_disk(existing_run, monkeypatch):
    optimization = regenerate(existing_run, monkeypatch)["optimization"]
    assert optimization["n_trials_recorded"] == len(TRIALS)
    assert optimization["state_counts"] == {"COMPLETE": len(TRIALS)}
    assert {family["model_type"] for family in optimization["model_families"]} == {
        "DataReuploadingClassifier",
        "SVC",
    }


def test_fairness_search_and_pareto_front_are_recovered_not_lost(existing_run, monkeypatch):
    context = regenerate(existing_run, monkeypatch)
    search = context["fairness"]["search"]
    assert search["mode"] == "multi_objective"
    assert search["metric"] == "equal_opportunity_difference"
    assert search["trial_disparities"]["n_trials_scored"] == len(TRIALS)
    assert search["trial_disparities"]["n_feasible"] == 1

    # pareto_trials was never persisted on the run; it comes back from
    # study.best_trials so an old multi-objective run still reports its front.
    pareto = context["pareto_front"]
    assert pareto["present"] is True
    assert pareto["n_points"] == 2  # noqa: PLR2004
    assert pareto["knee_trial"] is not None


def test_a_missing_fairness_audit_is_declared_rather_than_implied(existing_run, monkeypatch):
    """An old snapshot predating the audit must not read as "no disparity found"."""
    context = regenerate(existing_run, monkeypatch)
    assert context["fairness"]["audit_available"] is False
    assert any(
        "protected attribute" in note and "no audit is stored" in note
        for note in context["omissions"]
    )
