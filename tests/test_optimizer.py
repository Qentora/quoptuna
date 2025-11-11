"""Tests for the Optimizer class."""

import os
from pathlib import Path

import numpy as np

from quoptuna.backend.tuners.optimizer import Optimizer


def test_optimizer_initialization_with_data(sample_split_data, tmp_path):
    """Test Optimizer initialization with provided data."""
    db_name = "test_optimizer"
    optimizer = Optimizer(
        db_name=db_name,
        data=sample_split_data,
        study_name="test_study",
    )

    assert optimizer.db_name == db_name
    assert optimizer.study_name == "test_study"
    assert optimizer.train_x is not None
    assert optimizer.test_x is not None
    assert optimizer.train_y is not None
    assert optimizer.test_y is not None
    assert optimizer.study is None  # Study not created yet


def test_optimizer_initialization_with_dataset_name(tmp_path):
    """Test Optimizer initialization with dataset name."""
    db_name = "test_db"
    dataset_name = "sample_dataset"

    optimizer = Optimizer(
        db_name=db_name,
        dataset_name=dataset_name,
        study_name="test_study",
    )

    assert optimizer.db_name == db_name
    assert optimizer.dataset_name == dataset_name
    assert optimizer.study_name == "test_study"


def test_optimizer_creates_db_directory(tmp_path):
    """Test that Optimizer creates db directory if it doesn't exist."""
    # Change to temp directory to avoid polluting the repo
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        optimizer = Optimizer(db_name="test_db", study_name="test_study")

        # Check that db directory was created
        db_dir = Path("db")
        assert db_dir.exists()
        assert db_dir.is_dir()
    finally:
        os.chdir(original_cwd)


def test_optimizer_storage_location(tmp_path):
    """Test that storage location is set correctly."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        db_name = "test_storage"
        optimizer = Optimizer(db_name=db_name, study_name="test_study")

        expected_path = f"db/{db_name}.db"
        expected_storage = f"sqlite:///{expected_path}"

        assert optimizer.storage_location == expected_storage
    finally:
        os.chdir(original_cwd)


def test_optimizer_log_user_attributes_classical(sample_split_data, tmp_path):
    """Test logging user attributes for classical models."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        optimizer = Optimizer(
            db_name="test_log",
            data=sample_split_data,
            study_name="test_study",
        )

        # Mock trial object
        from optuna import create_study

        study = create_study()
        trial = study.ask()

        eval_scores = {"accuracy": 0.85, "f1_score": 0.82, "score": 0.83}
        optimizer.log_user_attributes("SVC", eval_scores, trial)

        # Check that classical attributes were set
        assert trial.user_attrs["Classical_accuracy"] == 0.85
        assert trial.user_attrs["Classical_f1_score"] == 0.82
        assert trial.user_attrs["Classical_score"] == 0.83
        assert trial.user_attrs["Quantum_accuracy"] == 0
        assert trial.user_attrs["Quantum_f1_score"] == 0
        assert trial.user_attrs["Quantum_score"] == 0
    finally:
        os.chdir(original_cwd)


def test_optimizer_log_user_attributes_quantum(sample_split_data, tmp_path):
    """Test logging user attributes for quantum models."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        optimizer = Optimizer(
            db_name="test_log",
            data=sample_split_data,
            study_name="test_study",
        )

        from optuna import create_study

        study = create_study()
        trial = study.ask()

        eval_scores = {"accuracy": 0.88, "f1_score": 0.86, "score": 0.87}
        optimizer.log_user_attributes("CircuitCentricClassifier", eval_scores, trial)

        # Check that quantum attributes were set
        assert trial.user_attrs["Quantum_accuracy"] == 0.88
        assert trial.user_attrs["Quantum_f1_score"] == 0.86
        assert trial.user_attrs["Quantum_score"] == 0.87
        assert trial.user_attrs["Classical_accuracy"] == 0
        assert trial.user_attrs["Classical_f1_score"] == 0
        assert trial.user_attrs["Classical_score"] == 0
    finally:
        os.chdir(original_cwd)


def test_optimizer_with_empty_data_dict(tmp_path):
    """Test Optimizer initialization with empty data dictionary."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        optimizer = Optimizer(db_name="test_empty", data={}, study_name="test_study")

        assert optimizer.train_x is None
        assert optimizer.test_x is None
        assert optimizer.train_y is None
        assert optimizer.test_y is None
    finally:
        os.chdir(original_cwd)


def test_optimizer_with_no_data(tmp_path):
    """Test Optimizer initialization with no data provided."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        optimizer = Optimizer(db_name="test_no_data", study_name="test_study")

        assert optimizer.data == {}
        assert optimizer.train_x is None
        assert optimizer.test_x is None
        assert optimizer.train_y is None
        assert optimizer.test_y is None
    finally:
        os.chdir(original_cwd)


def test_optimizer_partial_data(tmp_path):
    """Test Optimizer with partial data provided."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        partial_data = {
            "train_x": np.array([[1, 2], [3, 4]]),
            "train_y": np.array([1, -1]),
        }

        optimizer = Optimizer(
            db_name="test_partial",
            data=partial_data,
            study_name="test_study",
        )

        assert optimizer.train_x is not None
        assert optimizer.train_y is not None
        assert optimizer.test_x is None
        assert optimizer.test_y is None
    finally:
        os.chdir(original_cwd)


def test_optimizer_all_classical_model_types(sample_split_data, tmp_path):
    """Test that all classical model types are correctly identified."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        optimizer = Optimizer(
            db_name="test_classical",
            data=sample_split_data,
            study_name="test_study",
        )

        from optuna import create_study

        study = create_study()
        eval_scores = {"accuracy": 0.9, "f1_score": 0.85, "score": 0.88}

        classical_models = ["SVC", "SVClinear", "MLPClassifier", "Perceptron"]

        for model_type in classical_models:
            trial = study.ask()
            optimizer.log_user_attributes(model_type, eval_scores, trial)

            # Verify classical attributes are set
            assert trial.user_attrs["Classical_accuracy"] == 0.9
            assert trial.user_attrs["Quantum_accuracy"] == 0
    finally:
        os.chdir(original_cwd)
