"""Unit tests for serialize_study_trials (completion-time trial history)."""

import optuna

from quoptuna.server.api.v1.optimize import serialize_study_trials
from quoptuna.server.services.storage import optuna_storage_url


def test_missing_study_returns_empty_list(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert serialize_study_trials("no-such-db", "no-such-study") == []


def test_serializes_finished_trials_with_their_own_params(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "db").mkdir(exist_ok=True)
    db_name = "ser-test"
    study_name = "ser-study"
    study = optuna.create_study(
        storage=optuna_storage_url(db_name), study_name=study_name, direction="maximize"
    )

    def objective(trial):
        model = trial.suggest_categorical("model_type", ["SVC", "IQPKernelClassifier"])
        return 0.9 if model == "IQPKernelClassifier" else 0.5

    study.optimize(objective, n_trials=6)

    trials = serialize_study_trials(db_name, study_name)
    assert len(trials) == 6
    assert all(t["state"] == "COMPLETE" for t in trials)
    # Each row keeps its own params — both model families stay visible.
    models = {t["params"]["model_type"] for t in trials}
    assert models == {"SVC", "IQPKernelClassifier"}
    assert [t["trial"] for t in trials] == sorted({t["trial"] for t in trials})
