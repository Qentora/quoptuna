"""Platform-precision regression tests.

Both cases below failed only on Windows, where NumPy's default integer is
int32 while Linux/macOS use int64 -- so they are written to assert the
behaviour that must hold on every platform, not to skip on any of them.
"""

import os
from unittest import mock

import numpy as np
import pytest

from quoptuna.backend.base.pennylane_models.qml_benchmarks import jax_config
from quoptuna.backend.base.pennylane_models.qml_benchmarks.models.quantum_kitchen_sinks import (
    QuantumKitchenSinks,
)
from quoptuna.backend.base.pennylane_models.qml_benchmarks.models.separable import (
    SeparableKernelClassifier,
)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 4))
    return x, np.where(x[:, 0] > 0, 1, -1)


class TestSamplingPrecision:
    def test_probe_is_a_noop_when_numpy_default_int_is_64bit(self):
        """On Linux/macOS the workaround must not engage and cost precision."""
        with mock.patch.object(jax_config, "_numpy_default_int_is_64bit", return_value=True):
            assert jax_config.sampling_needs_float32() is False

    def test_probe_engages_only_when_x64_and_narrow_numpy_int_coincide(self):
        with mock.patch.object(jax_config, "_numpy_default_int_is_64bit", return_value=False):
            import jax

            assert jax_config.sampling_needs_float32() is bool(jax.config.jax_enable_x64)

    @pytest.mark.parametrize(("value", "expected"), [("1", True), ("0", False)])
    def test_env_override_wins_over_the_probe(self, value, expected):
        with mock.patch.dict(os.environ, {"QUOPTUNA_SAMPLING_FLOAT32": value}):
            assert jax_config.sampling_needs_float32() is expected

    def test_context_manager_restores_the_global_setting(self):
        import jax

        before = jax.config.jax_enable_x64
        with jax_config.sampling_precision():
            pass
        assert jax.config.jax_enable_x64 == before

    def test_shot_based_model_fits_under_the_default_precision(self, toy_data):
        """Regression: QKS raised XlaRuntimeError on Windows under x64."""
        x, y = toy_data
        model = QuantumKitchenSinks(n_episodes=4, random_state=0)
        model.fit(x, y)
        assert model.predict(x).shape == y.shape


class TestSeparableKernelMatrix:
    def _fitted_circuit(self, n_features, max_vmap):
        model = SeparableKernelClassifier(
            encoding_layers=2, max_vmap=max_vmap, random_state=0
        )
        model.n_qubits_ = n_features
        model.construct_circuit()
        return model

    def test_matches_pairwise_reference(self):
        """The vectorised Gram matrix must equal the per-pair computation."""
        rng = np.random.default_rng(1)
        x1, x2 = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
        model = self._fitted_circuit(4, max_vmap=64)

        expected = np.array(
            [[float(model.forward(np.concatenate((a, b)))) for b in x2] for a in x1]
        )
        np.testing.assert_allclose(model.precompute_kernel(x1, x2), expected, atol=1e-10)

    @pytest.mark.parametrize("max_vmap", [1, 3, 64, 10_000])
    def test_chunking_width_does_not_change_the_result(self, max_vmap):
        """max_vmap bounds memory only; it must never alter the kernel."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=(9, 4))
        reference = self._fitted_circuit(4, max_vmap=10_000).precompute_kernel(x, x)
        actual = self._fitted_circuit(4, max_vmap=max_vmap).precompute_kernel(x, x)
        np.testing.assert_allclose(actual, reference, atol=1e-10)

    def test_kernel_matrix_is_symmetric_and_unit_diagonal(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(6, 4))
        gram = self._fitted_circuit(4, max_vmap=8).precompute_kernel(x, x)
        np.testing.assert_allclose(gram, gram.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(gram), np.ones(len(x)), atol=1e-10)

    def test_fit_predict_round_trip(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(30, 4))
        y = np.where(x[:, 0] > 0, 1, -1)
        model = SeparableKernelClassifier(encoding_layers=1, max_vmap=32, random_state=0)
        model.fit(x, y)
        assert model.predict(x).shape == y.shape
