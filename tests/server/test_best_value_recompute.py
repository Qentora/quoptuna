"""The /detail endpoint must recompute best_value from the Optuna study.

The stored best_value can go stale when the background task's final write is
lost (e.g. a dev-server reload mid-run). Rehydration paths read /detail, so a
stale value there surfaced as a "Best F1" chip lower than the trial table's
best. The study on disk is the source of truth.
"""

import asyncio

import optuna
import pytest

from quoptuna.server.api.v1.optimize import get_optimization_detail, optimization_jobs
from quoptuna.server.services import run_store
from quoptuna.server.services.storage import optuna_storage_url

BEST_VALUE = 0.9
STALE_VALUE = 0.2  # simulates a lost final write: last-seen, not the max


@pytest.fixture
def study_with_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "db").mkdir(exist_ok=True)
    db_name = "bv-test"
    study_name = "bv-study"
    study = optuna.create_study(
        storage=optuna_storage_url(db_name), study_name=study_name, direction="maximize"
    )
    values = iter([0.5, BEST_VALUE, STALE_VALUE])  # best is NOT the last trial
    study.optimize(lambda t: t.suggest_categorical("C", [1.0]) and next(values), n_trials=3)

    job_id = "opt_bvtest"
    job = {
        "id": job_id,
        "status": "completed",
        "current_trial": 3,
        "total_trials": 3,
        "best_value": STALE_VALUE,  # stale stored value
        "best_params": {"C": 1.0},
        "trials": None,
        "started_at": "2026-07-12T00:00:00",
        "completed_at": "2026-07-12T00:01:00",
        "error": None,
        "request": {
            "dataset_id": "bv-ds",
            "database_name": db_name,
            "study_name": study_name,
        },
    }
    optimization_jobs[job_id] = job
    run_store.save_run(job)
    yield job_id
    optimization_jobs.pop(job_id, None)
    run_store.delete_run(job_id)


def test_detail_recomputes_stale_best_value(study_with_run):
    detail = asyncio.run(get_optimization_detail(study_with_run))
    assert detail["best_value"] == BEST_VALUE
    # The corrected value is written back so subsequent reads agree too.
    assert optimization_jobs[study_with_run]["best_value"] == BEST_VALUE
    assert run_store.get_run(study_with_run)["best_value"] == BEST_VALUE
