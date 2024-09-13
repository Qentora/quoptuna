# import CircuitCentricClassifier
from quoptuna.backend.models.pennylane_models.circuit_models import (
    CircuitCentricClassifier,
)


def test_circuit_centric_classifier():
    clf = CircuitCentricClassifier()
    assert clf is not None
