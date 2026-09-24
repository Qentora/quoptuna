import numpy as np

from quoptuna.backend.base.pennylane_models.qml_benchmarks.models import (
    quantum_metric_learning as metric_module,
)


def test_fit_adjusts_vmap_when_smallest_class_shrinks_batch(monkeypatch):
    def initialize_without_circuit(self, _n_features, classes):
        self.classes_ = classes
        self.params_ = {}

    def skip_training(model, *args, **kwargs):
        return model.params_

    monkeypatch.setattr(
        metric_module.QuantumMetricLearner, "initialize", initialize_without_circuit
    )
    monkeypatch.setattr(metric_module, "train", skip_training)

    batch_size = 32
    smallest_class_size = 3
    model = metric_module.QuantumMetricLearner(
        batch_size=batch_size, max_vmap=batch_size, jit=False
    )
    features = np.arange(7, dtype=float).reshape(-1, 1)
    labels = np.array([-1, -1, -1, 1, 1, 1, 1])

    model.fit(features, labels)

    assert model.batch_size == smallest_class_size
    assert model.max_vmap == smallest_class_size
