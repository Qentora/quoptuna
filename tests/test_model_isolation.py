"""Two models of the same type must not share fitted state.

The kernel-head models took their inner estimator as a mutable default
argument (``svm=SVC(...)``, ``linear_model=LogisticRegression(...)``). Python
evaluates a default once, at import, so every instance of the class shared one
estimator object — and each ``fit`` refitted it in place.

Consequences in the analysis pipeline, where several models are built in one
process: the job's model is fitted, then the fairness section builds its own,
and that second fit rewrote the first model's classifier head. Predictions
changed underneath the metrics that were already being computed, so every
re-analysis of the same run returned different scores.
"""

import numpy as np
import pytest

from quoptuna.backend.models import MODEL_PARAM_KEYS, create_model
from quoptuna.backend.tuners.optimizer import DEFAULT_SEARCH_SPACE

# Kernel/feature-map models whose final classifier is an inner sklearn
# estimator — the ones that carried the shared default.
SHARED_ESTIMATOR_MODELS = [
    ("ProjectedQuantumKernel", "svm"),
    ("IQPKernelClassifier", "svm"),
    ("SeparableKernelClassifier", "svm"),
    ("QuantumKitchenSinks", "linear_model"),
]
N_ROWS = 24


def _build(model_type):
    """Construct via the real factory; every listed hyperparameter is required."""
    params = {
        key: DEFAULT_SEARCH_SPACE[key][0]
        for key in MODEL_PARAM_KEYS[model_type]
        if key in DEFAULT_SEARCH_SPACE
    }
    params["max_vmap"] = 8
    return create_model(model_type, **params)


@pytest.fixture
def two_tasks():
    """Two datasets with different decision boundaries.

    Different boundaries matter: a leaked fit only changes predictions when
    the second model learned something else.
    """
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=(N_ROWS, 2))
    x2 = rng.normal(size=(N_ROWS, 2))
    return (x1, np.where(x1[:, 0] > 0, 1, -1)), (x2, np.where(x2[:, 1] > 0, 1, -1))


@pytest.mark.parametrize(("model_type", "attribute"), SHARED_ESTIMATOR_MODELS)
def test_instances_do_not_share_their_inner_estimator(model_type, attribute):
    first = _build(model_type)
    second = _build(model_type)

    assert getattr(first, attribute) is not getattr(second, attribute)


@pytest.mark.parametrize(("model_type", "attribute"), SHARED_ESTIMATOR_MODELS)
def test_fitting_another_model_does_not_change_predictions(model_type, attribute, two_tasks):
    (x1, y1), (x2, y2) = two_tasks
    first = _build(model_type)
    first.fit(x1, y1)
    before = np.asarray(first.predict(x1)).copy()

    # The second fit is what the analysis does when the fairness section
    # builds its own XAI while the job's model is still in use.
    _build(model_type).fit(x2, y2)

    after = np.asarray(first.predict(x1))
    assert np.array_equal(before, after), (
        f"{model_type}: {(before != after).sum()}/{len(before)} predictions changed "
        "because another instance was fitted"
    )


@pytest.mark.parametrize(("model_type", "attribute"), SHARED_ESTIMATOR_MODELS)
def test_refitting_the_same_configuration_is_reproducible(model_type, attribute, two_tasks):
    """Analysis reruns retrain the trial; identical inputs must give identical output."""
    (x1, y1), _ = two_tasks

    first = _build(model_type).fit(x1, y1)
    second = _build(model_type).fit(x1, y1)

    assert np.array_equal(np.asarray(first.predict(x1)), np.asarray(second.predict(x1)))
