"""Per-model integration + regression tests on preprocessed Banknote data.

``test_model_fits_on_banknote`` guarantees at least one real fit/predict for
every model the optimizer can sample, which directly exercises all three fixes
(jax-jit ``legacy_vectorized``, 1-D label/chex, MLP ``(100,)`` literal).
"""

import ast

import pytest
from sklearn.metrics import f1_score

from quoptuna.backend.models import create_model

from .conftest import OPTIMIZER_MODEL_TYPES, cap_training


@pytest.mark.slow
@pytest.mark.parametrize("model_type", OPTIMIZER_MODEL_TYPES)
def test_model_fits_on_banknote(model_type, base_params, preprocessed_banknote):
    """5b: each optimizer model fits + predicts once on preprocessed Banknote.

    Image/grid models (ConvolutionalNeuralNetwork, QuanvolutionalNeuralNetwork,
    WeiNet) and QuantumBoltzmannMachine(+Separable) are out of scope here: they
    are not in the optimizer's model list and expect image-shaped inputs. Their
    construction is covered by tests/test_create_model.py.
    """
    train_x = preprocessed_banknote["train_x"]
    train_y = preprocessed_banknote["train_y"]
    test_x = preprocessed_banknote["test_x"]
    test_y = preprocessed_banknote["test_y"]

    model = cap_training(create_model(model_type, **base_params))
    model.fit(train_x, train_y)

    preds = model.predict(test_x)
    assert len(preds) == len(test_y)

    score = f1_score(test_y, preds, average="weighted")
    assert 0.0 <= score <= 1.0


def test_mlp_literal_is_valid_and_bad_literal_raises():
    """5c: the corrected literal parses; the old "[100,)" literal must raise."""
    model = create_model(
        "MLPClassifier",
        hidden_layer_sizes="(100,)",
        learning_rate=0.01,
        alpha=0.0001,
    )
    assert model.hidden_layer_sizes == (100,)

    with pytest.raises(SyntaxError):
        ast.literal_eval("[100,)")
