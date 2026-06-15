"""Shared fixtures for the backend integration/regression tests.

The fixtures here reproduce the optimizer's own preprocessing on a small,
committed subset of the UCI Banknote Authentication dataset (id 267) so the
tests are deterministic and run offline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BANKNOTE_CSV = FIXTURES_DIR / "banknote_sample.csv"

# Feature/target selection matching the "Optimization Configuration" UI case.
SELECTED_FEATURES = ["entropy", "curtosis", "skewness", "variance"]
TARGET_COLUMN = "class"

# The 15 model types the optimizer actually samples (src/quoptuna/backend/
# tuners/optimizer.py:111-130). Image/grid models (Conv/Quanvolutional/WeiNet)
# and QuantumBoltzmann(+Separable) are intentionally excluded: they expect
# image-shaped inputs and are not part of the optimizer's search space.
OPTIMIZER_MODEL_TYPES = [
    "CircuitCentricClassifier",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierSeparable",
    "ProjectedQuantumKernel",
    "QuantumKitchenSinks",
    "QuantumMetricLearner",
    "TreeTensorClassifier",
    "SeparableVariationalClassifier",
    "SeparableKernelClassifier",
    "SVC",
    "SVClinear",
    "MLPClassifier",
    "Perceptron",
]

# A single params dict mirroring the search space in optimizer.py:75-109 with
# valid representative values. This is exactly how objective() feeds one params
# dict to every candidate model. Note hidden_layer_sizes uses the corrected
# "(100,)" literal.
BASE_PARAMS = {
    "max_vmap": 1,
    "batch_size": 32,
    "learning_rate": 0.01,
    "n_input_copies": 2,
    "n_layers": 1,
    "observable_type": "single",
    "repeats": 5,
    "C": 1,
    "gamma_factor": 1,
    "trotter_steps": 3,
    "t": 0.1,
    "n_qfeatures": "half",
    "n_episodes": 10,
    "visible_qubits": "full",
    "temperature": 1,
    "encoding_layers": 3,
    "degree": 2,
    "n_qchannels": 5,
    "qkernel_shape": 3,
    "kernel_shape": 3,
    "filter_name": "sharpen",
    "gamma": 0.1,
    "alpha": 0.01,
    "hidden_layer_sizes": "(100,)",
    "eta0": 0.1,
}

# A deliberately tiny search space for the optimizer integration tests: a single
# value per hyperparameter (so params are deterministic) and a small, mixed set
# of model types. Including the easily-separable classical models guarantees a
# best_value > 0 on Banknote. Wrapping every BASE_PARAMS value in a one-element
# list guarantees every candidate model's required keys are present.
TEST_SEARCH_SPACE = {key: [value] for key, value in BASE_PARAMS.items()}
TEST_MODEL_TYPES = [
    "DataReuploadingClassifier",
    "ProjectedQuantumKernel",
    "SVC",
    "MLPClassifier",
    "Perceptron",
]


# Variational qml_benchmarks models train to convergence with a floor of
# 2 * convergence_interval (default 200 -> >=400 steps) and max_steps=10000.
# create_model does not expose these knobs, so the tests cap them after
# construction to keep fits fast while still exercising the real fit/predict
# path. Accuracy is irrelevant here -- we only assert the run succeeds.
FAST_MAX_STEPS = 30
FAST_CONVERGENCE_INTERVAL = 5


def cap_training(
    model,
    max_steps: int = FAST_MAX_STEPS,
    convergence_interval: int = FAST_CONVERGENCE_INTERVAL,
):
    """Shrink training iterations for any model that exposes the knobs."""
    if hasattr(model, "max_steps"):
        model.max_steps = max_steps
    if hasattr(model, "convergence_interval"):
        model.convergence_interval = convergence_interval
    return model


@pytest.fixture
def fast_optimizer_training(monkeypatch):
    """Patch the optimizer's create_model so every built model trains briefly.

    The optimizer imports ``create_model`` into its own module namespace, so we
    patch it there. This keeps the real create_model -> fit -> score path intact
    (still exercising all three fixes) while bounding wall-clock time.
    """
    import quoptuna.backend.tuners.optimizer as optimizer_module

    real_create_model = optimizer_module.create_model

    def fast_create_model(model_type, **kwargs):
        return cap_training(real_create_model(model_type, **kwargs))

    monkeypatch.setattr(optimizer_module, "create_model", fast_create_model)
    return fast_create_model


@pytest.fixture(scope="session")
def banknote_csv() -> Path:
    """Path to the committed Banknote sample CSV."""
    assert BANKNOTE_CSV.exists(), f"Missing fixture: {BANKNOTE_CSV}"
    return BANKNOTE_CSV


@pytest.fixture
def base_params() -> dict:
    """A fresh copy of the representative search-space params for one trial."""
    return dict(BASE_PARAMS)


@pytest.fixture
def preprocessed_banknote(banknote_csv: Path) -> dict:
    """Reproduce the optimizer's preprocessing and return a data dict.

    Mirrors ``quoptuna.backend.utils.data_utils.data.preprocess_data``:
    StandardScaler the features, map labels to ``{-1, 1}``, then
    ``train_test_split(random_state=42)``. Labels are returned as 1-D arrays,
    matching the section-2 ravel fix.
    """
    df = pd.read_csv(banknote_csv)
    x = df[SELECTED_FEATURES].to_numpy()
    y_raw = df[TARGET_COLUMN].to_numpy()

    x = StandardScaler().fit_transform(x)
    classes = np.unique(y_raw)
    y = np.where(y_raw == classes[0], 1, -1)

    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)
    return {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y,
        "test_y": test_y,
    }
