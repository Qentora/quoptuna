"""Unit tests for the Optimizer class."""

import os
from pathlib import Path

from quoptuna.backend.tuners.optimizer import Optimizer


def test_optimizer_initialization_with_data(train_test_split_data, tmp_path):
    """Test Optimizer initialization with provided data."""
    db_path = tmp_path / "test_db"
    db_name = str(db_path)

    optimizer = Optimizer(
        db_name=db_name,
        data=train_test_split_data,
        study_name="test_study",
    )

    assert optimizer.db_name == db_name
    assert optimizer.train_x is not None
    assert optimizer.test_x is not None
    assert optimizer.train_y is not None
    assert optimizer.test_y is not None
    assert optimizer.study_name == "test_study"
    assert optimizer.storage_location == f"sqlite:///{optimizer.data_path}"


def test_optimizer_initialization_without_data(tmp_path):
    """Test Optimizer initialization without data."""
    db_path = tmp_path / "test_db"
    db_name = str(db_path)

    optimizer = Optimizer(
        db_name=db_name,
        dataset_name="",
        study_name="test_study",
    )

    assert optimizer.db_name == db_name
    assert optimizer.train_x is None
    assert optimizer.test_x is None
    assert optimizer.train_y is None
    assert optimizer.test_y is None
    assert optimizer.study_name == "test_study"


def test_optimizer_creates_db_directory(tmp_path, train_test_split_data):
    """Test that Optimizer creates db directory if it doesn't exist."""
    # Change to tmp directory to avoid creating db in project root
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        db_name = "test_optimizer_db"
        optimizer = Optimizer(
            db_name=db_name,
            data=train_test_split_data,
            study_name="test_study",
        )

        # Check that db directory was created
        db_dir = Path("db")
        assert db_dir.exists()
        assert db_dir.is_dir()

        # Verify the database path is set correctly
        expected_db_path = db_dir / f"{db_name}.db"
        assert optimizer.data_path == str(expected_db_path)

    finally:
        os.chdir(original_cwd)


def test_optimizer_with_dataset_name(tmp_path):
    """Test Optimizer initialization with dataset_name."""
    db_path = tmp_path / "test_db"
    db_name = str(db_path)

    optimizer = Optimizer(
        db_name=db_name,
        dataset_name="test_dataset",
        study_name="test_study",
    )

    assert optimizer.dataset_name == "test_dataset"
    assert optimizer.db_name == db_name


def test_optimizer_optimize_with_data(train_test_split_data, tmp_path):
    """Test the optimize method with provided data."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        db_name = "test_optimize_db"
        optimizer = Optimizer(
            db_name=db_name,
            data=train_test_split_data,
            study_name="test_optimize_study",
        )

        # Run optimization with a small number of trials
        study, best_trials = optimizer.optimize(n_trials=2)

        # Verify study was created
        assert study is not None
        assert optimizer.study is not None

        # Verify trials were run
        assert len(study.trials) == 2

        # Verify best trials exist
        assert best_trials is not None

    finally:
        os.chdir(original_cwd)


def test_optimizer_load_study(train_test_split_data, tmp_path):
    """Test loading a study after optimization."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        db_name = "test_load_study_db"
        study_name = "test_load_study"

        # Create and run optimizer
        optimizer = Optimizer(
            db_name=db_name,
            data=train_test_split_data,
            study_name=study_name,
        )

        # Run optimization
        optimizer.optimize(n_trials=2)

        # Create new optimizer instance and load study
        optimizer2 = Optimizer(
            db_name=db_name,
            data=train_test_split_data,
            study_name=study_name,
        )

        optimizer2.load_study()

        # Verify study was loaded
        assert optimizer2.study is not None
        assert optimizer2.study.study_name == study_name
        assert len(optimizer2.study.trials) == 2

    finally:
        os.chdir(original_cwd)


def test_optimizer_log_user_attributes(train_test_split_data, tmp_path):
    """Test the log_user_attributes method."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        db_name = "test_log_attrs_db"
        optimizer = Optimizer(
            db_name=db_name,
            data=train_test_split_data,
            study_name="test_log_attrs_study",
        )

        # Run optimization to get a trial
        study, _ = optimizer.optimize(n_trials=1)
        trial = study.trials[0]

        # Test logging quantum model attributes
        eval_scores = {"accuracy": 0.85, "f1_score": 0.80, "score": 0.82}
        optimizer.log_user_attributes("CircuitCentricClassifier", eval_scores, trial)

        assert trial.user_attrs["Quantum_accuracy"] == 0.85
        assert trial.user_attrs["Quantum_f1_score"] == 0.80
        assert trial.user_attrs["Quantum_score"] == 0.82
        assert trial.user_attrs["Classical_accuracy"] == 0
        assert trial.user_attrs["Classical_f1_score"] == 0
        assert trial.user_attrs["Classical_score"] == 0

        # Test logging classical model attributes
        eval_scores2 = {"accuracy": 0.90, "f1_score": 0.88, "score": 0.89}
        optimizer.log_user_attributes("SVC", eval_scores2, trial)

        assert trial.user_attrs["Classical_accuracy"] == 0.90
        assert trial.user_attrs["Classical_f1_score"] == 0.88
        assert trial.user_attrs["Classical_score"] == 0.89
        assert trial.user_attrs["Quantum_accuracy"] == 0
        assert trial.user_attrs["Quantum_f1_score"] == 0
        assert trial.user_attrs["Quantum_score"] == 0

    finally:
        os.chdir(original_cwd)


def test_optimizer_objective_handles_errors(train_test_split_data, tmp_path):
    """Test that the objective function handles errors gracefully."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        db_name = "test_error_handling_db"
        optimizer = Optimizer(
            db_name=db_name,
            data=train_test_split_data,
            study_name="test_error_handling_study",
        )

        # Run optimization - some trials may fail but should return 0
        study, _ = optimizer.optimize(n_trials=3)

        # Verify that trials were attempted even if some failed
        assert len(study.trials) == 3

        # All trials should have a value (0 if failed)
        for trial in study.trials:
            assert trial.value is not None

    finally:
        os.chdir(original_cwd)
