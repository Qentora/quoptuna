"""Integration tests for the headless CLI optimization runner.

Runs the exact UI pipeline (OptimizationRequest -> build_workflow ->
WorkflowExecutor) end-to-end on a local Iris CSV: multiclass (3 classes,
macro-F1, OvR quantum model) and a binary regression run. No network — the
UCI path is exercised only for option plumbing via the Typer runner.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from typer.testing import CliRunner

from quoptuna import cli
from quoptuna.server.services.headless import run_headless_optimization

runner = CliRunner()

N_CLASSES = 3
MIN_GOOD_F1 = 0.5


@pytest.fixture
def iris_csv(tmp_path):
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(iris.target_names)))
    # Short feature names keep the CLI/CSV round-trip simple.
    df.columns = ["sl", "sw", "pl", "pw", "species"]
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    # DB_DIR is read at call time by optuna_db_path, so patching it isolates
    # every consumer (Optimizer, build_xai, analysis) in one place.
    monkeypatch.setattr("quoptuna.backend.utils.storage.DB_DIR", tmp_path / "db")


@pytest.mark.slow
def test_iris_multiclass_end_to_end(iris_csv):
    summary = run_headless_optimization(
        csv_path=iris_csv,
        target="species",
        n_trials=2,
        model_types=["SVC", "DataReuploadingClassifier"],
        search_space={
            "max_vmap": [4],
            "batch_size": [8],
            "learning_rate": [0.05],
            "n_layers": [1],
            "observable_type": ["single"],
            "C": [1.0],
            "gamma": [0.1],
        },
        favorable_class="setosa",
        max_steps=10,
        convergence_interval=5,
        db_name="cli_it_mc.db",
        subset_size=20,
        analyze=True,
    )

    spec = summary["task_spec"]
    assert spec["kind"] == "multiclass"
    assert spec["n_classes"] == N_CLASSES
    assert spec["class_labels"] == ["setosa", "versicolor", "virginica"]
    assert spec["favorable_code"] == 0
    # best_value is a macro F1 in [0, 1]
    assert 0.0 <= summary["best_value"] <= 1.0

    analysis = summary["analysis"]
    assert set(analysis["confusion_matrix"]["labels"]) == {"setosa", "versicolor", "virginica"}
    cm = np.asarray(analysis["confusion_matrix"]["matrix"])
    assert cm.shape == (N_CLASSES, N_CLASSES)
    assert analysis["metrics"]["f1_score"] is not None
    if analysis["curves"]:
        assert len(analysis["curves"]["per_class_auc"]) == N_CLASSES


@pytest.mark.slow
def test_iris_multiclass_without_favorable_class(iris_csv):
    """Training must not require a favorable class (it's fairness-only)."""
    summary = run_headless_optimization(
        csv_path=iris_csv,
        target="species",
        n_trials=1,
        model_types=["SVC"],
        search_space={"C": [1.0], "gamma": [0.1]},
        db_name="cli_it_nofav.db",
        analyze=False,
    )
    assert summary["task_spec"]["kind"] == "multiclass"
    assert summary["task_spec"]["favorable_code"] is None
    assert summary["best_value"] > MIN_GOOD_F1


@pytest.mark.slow
def test_binary_regression_run(iris_csv, tmp_path):
    """Binary targets keep the {-1,+1} path (regression guard)."""
    df = pd.read_csv(iris_csv)
    df = df[df["species"] != "virginica"]
    path = tmp_path / "iris_binary.csv"
    df.to_csv(path, index=False)

    summary = run_headless_optimization(
        csv_path=str(path),
        target="species",
        n_trials=1,
        model_types=["SVC"],
        search_space={"C": [1.0], "gamma": [0.1]},
        label_neg="setosa",
        label_pos="versicolor",
        db_name="cli_it_bin.db",
        analyze=True,
    )
    assert summary["task_spec"]["kind"] == "binary"
    assert summary["task_spec"]["class_labels"] == ["setosa", "versicolor"]
    assert summary["analysis"]["curves"] is None  # binary: no per-class curve set
    assert summary["best_value"] > MIN_GOOD_F1


def test_cli_optimize_requires_dataset():
    result = runner.invoke(cli.app, ["optimize", "--trials", "1"])
    assert result.exit_code == 2  # noqa: PLR2004
    assert "exactly one" in result.output


def test_cli_optimize_dispatches(monkeypatch, iris_csv):
    """Option parsing: the Typer command forwards UI-equivalent options."""
    seen = {}

    def _fake_run(**kwargs):
        seen.update(kwargs)
        return {"best_value": 1.0, "task_spec": None}

    monkeypatch.setattr(
        "quoptuna.server.services.headless.run_headless_optimization", _fake_run
    )
    result = runner.invoke(
        cli.app,
        [
            "optimize",
            "--csv", iris_csv,
            "--target", "species",
            "--trials", "2",
            "--models", "SVC,IQPKernelClassifier",
            "--favorable-class", "setosa",
            "--sensitive-feature", "sw",
            "--seed", "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["n_trials"] == 2  # noqa: PLR2004
    assert seen["model_types"] == ["SVC", "IQPKernelClassifier"]
    assert seen["favorable_class"] == "setosa"
    assert seen["sensitive_feature"] == "sw"
    assert seen["sampler_seed"] == 7  # noqa: PLR2004
