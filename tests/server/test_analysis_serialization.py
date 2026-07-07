"""Unit tests for the raw-data (JSON) serialization helpers in analysis.py.

These helpers back the /api/v1/analysis/{curves,confusion-matrix,
feature-importance}/data endpoints and must produce plain, JSON-safe Python
structures with the documented shapes.
"""

import json

import numpy as np
import pytest

from quoptuna.server.api.v1.analysis import (
    MAX_CURVE_POINTS,
    _confusion_matrix_payload,
    _downsample_indices,
    _positive_proba,
    _pr_payload,
    _roc_payload,
)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    # Informative but noisy scores so curves have many thresholds.
    proba = np.clip(y * 0.6 + rng.normal(0.2, 0.25, size=200), 0, 1)
    return y, proba


class TestDownsampleIndices:
    def test_small_input_untouched(self):
        idx = _downsample_indices(10)
        assert idx.tolist() == list(range(10))

    def test_large_input_capped_and_keeps_endpoints(self):
        n = 10_000
        idx = _downsample_indices(n)
        assert len(idx) <= MAX_CURVE_POINTS
        assert idx[0] == 0
        assert idx[-1] == n - 1
        assert np.all(np.diff(idx) > 0)  # strictly increasing, no duplicates


class TestRocPayload:
    def test_schema_and_values(self, binary_data):
        y, proba = binary_data
        payload = _roc_payload(y, proba)
        assert set(payload) == {"fpr", "tpr", "auc"}
        assert isinstance(payload["fpr"], list)
        assert isinstance(payload["tpr"], list)
        assert len(payload["fpr"]) == len(payload["tpr"])
        assert len(payload["fpr"]) <= MAX_CURVE_POINTS
        assert all(0 <= v <= 1 for v in payload["fpr"] + payload["tpr"])
        assert isinstance(payload["auc"], float)
        assert 0 <= payload["auc"] <= 1
        json.dumps(payload)  # must be JSON-serializable

    def test_matches_sklearn(self, binary_data):
        from sklearn.metrics import roc_auc_score

        y, proba = binary_data
        payload = _roc_payload(y, proba)
        assert payload["auc"] == pytest.approx(roc_auc_score(y, proba))


class TestPrPayload:
    def test_schema_and_values(self, binary_data):
        y, proba = binary_data
        payload = _pr_payload(y, proba)
        assert set(payload) == {"precision", "recall", "average_precision"}
        assert len(payload["precision"]) == len(payload["recall"])
        assert len(payload["precision"]) <= MAX_CURVE_POINTS
        assert all(0 <= v <= 1 for v in payload["precision"] + payload["recall"])
        assert isinstance(payload["average_precision"], float)
        json.dumps(payload)

    def test_matches_sklearn(self, binary_data):
        from sklearn.metrics import average_precision_score

        y, proba = binary_data
        payload = _pr_payload(y, proba)
        assert payload["average_precision"] == pytest.approx(average_precision_score(y, proba))


class TestConfusionMatrixPayload:
    def test_schema_and_normalization(self):
        cm = np.array([[8, 2], [1, 9]])
        payload = _confusion_matrix_payload(cm, [0, 1])
        assert payload["labels"] == ["0", "1"]
        assert payload["matrix"] == [[8, 2], [1, 9]]
        assert payload["normalized"][0] == pytest.approx([0.8, 0.2])
        assert payload["normalized"][1] == pytest.approx([0.1, 0.9])
        json.dumps(payload)

    def test_zero_row_safe(self):
        payload = _confusion_matrix_payload([[0, 0], [3, 7]], ["neg", "pos"])
        assert payload["normalized"][0] == [0.0, 0.0]
        assert payload["normalized"][1] == pytest.approx([0.3, 0.7])


class TestPositiveProba:
    def test_two_column_reduced_to_positive_class(self):
        proba = np.array([[0.9, 0.1], [0.2, 0.8]])
        assert _positive_proba(proba).tolist() == [0.1, 0.8]

    def test_one_dimensional_passthrough(self):
        assert _positive_proba([0.3, 0.7]).tolist() == [0.3, 0.7]
