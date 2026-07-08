"""Tests for the training callback hook in ``model_utils.train``.

Uses a trivial quadratic loss instead of a quantum circuit so the JAX loop
runs in milliseconds while exercising the real ``train()`` code path.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from sklearn.exceptions import ConvergenceWarning

from quoptuna.backend.base.pennylane_models.qml_benchmarks.model_utils import train

MAX_STEPS = 12
INTERVAL = 3
ABORT_STEP = 5
EXPECTED_ABORT_HISTORY_LEN = 6


class _TinyModel:
    """Bare-minimum surface required by ``train``."""

    max_steps = MAX_STEPS
    batch_size = 4
    max_vmap = 4
    learning_rate = 0.1
    jit = False

    def __init__(self):
        self.params_ = {"w": jnp.array([1.0, 1.0])}


def _loss_fn(params, x, _y):
    return jnp.mean((x @ params["w"]) ** 2)


def _make_data():
    rng = np.random.default_rng(0)
    x = jnp.array(rng.normal(size=(16, 2)))
    y = jnp.array(rng.choice([-1.0, 1.0], size=16))
    return x, y


def _key_generator():
    keys = iter(jax.random.split(jax.random.PRNGKey(0), MAX_STEPS + 1))
    return lambda: next(keys)


def test_callback_fires_every_interval():
    model = _TinyModel()
    x, y = _make_data()
    calls = []
    model.training_callback = lambda step, hist: calls.append((step, len(hist)))

    # The loop may converge and break early (that path is fine); a short run
    # that hits max_steps raises ConvergenceWarning instead.
    with contextlib.suppress(ConvergenceWarning):
        train(model, _loss_fn, optax.adam, x, y, _key_generator(), convergence_interval=INTERVAL)

    # Callback fires exactly every INTERVAL steps, starting at step INTERVAL-1,
    # and receives the loss history accumulated so far.
    assert calls, "callback never fired"
    assert [step for step, _ in calls] == list(range(INTERVAL - 1, calls[-1][0] + 1, INTERVAL))
    assert all(n == step + 1 for step, n in calls)


def test_raising_callback_finalizes_bookkeeping():
    model = _TinyModel()
    x, y = _make_data()

    class _AbortError(Exception):
        pass

    def _cb(step, _hist):
        if step >= ABORT_STEP:
            raise _AbortError

    model.training_callback = _cb
    with pytest.raises(_AbortError):
        train(model, _loss_fn, optax.adam, x, y, _key_generator(), convergence_interval=INTERVAL)

    # Aborted training still records its consumed resources.
    assert len(model.loss_history_) == EXPECTED_ABORT_HISTORY_LEN  # aborted at step index 5
    assert model.training_time_ >= 0
    # Params are exposed mid-training for callbacks that need to predict.
    assert "w" in model.params_
