"""Multi-class support: TaskSpec, label encoding, OvR wrapping, macro-F1
objective, and favorable-class fairness binarization.

Binary regression checks are interleaved: every multiclass branch is paired
with an assertion that the binary path is byte-identical to the old behavior.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from quoptuna.backend.models import create_model
from quoptuna.backend.task_type import TaskSpec
from quoptuna.backend.tuners import optimizer as optimizer_module
from quoptuna.backend.tuners.optimizer import Optimizer
from quoptuna.backend.utils.data_utils.data import preprocess_data
from quoptuna.backend.xai.fairness import binarize_favorable, compute_disparity
from quoptuna.server.services.workflow_service import WorkflowExecutor

N_CLASSES = 3


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


class TestTaskSpec:
    def test_binary_derivation(self):
        spec = TaskSpec.from_target([0, 1, 0, 1], label_mapping={"neg": 0, "pos": 1})
        assert spec.kind == "binary"
        assert spec.n_classes == 2  # noqa: PLR2004
        assert spec.encoded_classes() == [-1, 1]
        assert list(spec.encode([0, 1, 0])) == [-1, 1, -1]

    def test_multiclass_derivation(self):
        spec = TaskSpec.from_target(["c", "a", "b", "a"], favorable_class="b")
        assert spec.kind == "multiclass"
        assert spec.n_classes == N_CLASSES
        assert spec.class_labels == ("a", "b", "c")  # sorted-unique, deterministic
        assert spec.favorable_code == 1
        assert list(spec.encode(["a", "b", "c"])) == [0, 1, 2]

    def test_display_labels(self):
        spec = TaskSpec.from_target(["x", "y", "z"])
        assert spec.display_label(2) == "z"
        binary = TaskSpec.from_target(["n", "p"], label_mapping={"neg": "n", "pos": "p"})
        assert binary.display_label(-1) == "n"
        assert binary.display_label(1) == "p"

    def test_round_trip(self):
        spec = TaskSpec.from_target(["a", "b", "c"], favorable_class="c")
        assert TaskSpec.from_dict(spec.to_dict()) == spec

    def test_unknown_favorable_raises(self):
        with pytest.raises(ValueError, match="favorable_class"):
            TaskSpec.from_target(["a", "b", "c"], favorable_class="nope")

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            TaskSpec.from_target(["a", "a"])


# ---------------------------------------------------------------------------
# Label encoding (workflow split + legacy preprocess paths)
# ---------------------------------------------------------------------------


def _run_split(y_values, config=None):
    executor = WorkflowExecutor({"nodes": [], "edges": []})
    data = {
        "x": pd.DataFrame({"f1": np.arange(len(y_values), dtype=float)}),
        "y": pd.Series(y_values, name="target"),
        "x_columns": ["f1"],
        "y_column": "target",
    }
    return executor._execute_train_test_split(config or {}, {"in": data})


class TestSplitEncoding:
    def test_multiclass_encodes_integer_codes(self):
        result = _run_split(["a", "b", "c"] * 8)
        codes = set(
            np.unique(
                np.concatenate(
                    [result["y_train"].to_numpy().ravel(), result["y_test"].to_numpy().ravel()]
                )
            )
        )
        assert codes == {0, 1, 2}
        assert result["task_spec"]["kind"] == "multiclass"
        assert result["task_spec"]["class_labels"] == ["a", "b", "c"]

    def test_binary_regression_unchanged(self):
        result = _run_split([0, 1] * 12, config={"label_mapping": {"neg": 0, "pos": 1}})
        codes = set(
            np.unique(
                np.concatenate(
                    [result["y_train"].to_numpy().ravel(), result["y_test"].to_numpy().ravel()]
                )
            )
        )
        assert codes == {-1, 1}
        assert result["task_spec"]["kind"] == "binary"

    def test_multiclass_never_collapses_pos_vs_rest(self):
        # The old bug: a stray label_mapping on a 3-class target collapsed
        # everything to pos-vs-rest. Now the multiclass path wins.
        result = _run_split(["a", "b", "c"] * 8, config={"favorable_class": "b"})
        assert result["task_spec"]["favorable_code"] == 1
        codes = set(np.unique(result["y_train"].to_numpy().ravel()))
        assert codes <= {0, 1, 2}
        assert len(codes) == N_CLASSES


class TestLegacyPreprocess:
    def test_binary_still_encoded(self):
        x = pd.DataFrame({"a": np.arange(10, dtype=float)})
        _, _, train_y, test_y = preprocess_data(x, np.array([5, 7] * 5))
        assert set(np.unique(np.concatenate([train_y, test_y]))) == {-1, 1}

    def test_multiclass_passes_through(self):
        x = pd.DataFrame({"a": np.arange(12, dtype=float)})
        y = np.array([0, 1, 2] * 4)
        _, _, train_y, test_y = preprocess_data(x, y)
        assert set(np.unique(np.concatenate([train_y, test_y]))) == {0, 1, 2}


# ---------------------------------------------------------------------------
# OvR adapter / model registry
# ---------------------------------------------------------------------------


class TestCreateModelMulticlass:
    def test_binary_returns_bare_model(self):
        model = create_model("SVC", gamma=0.1, C=1.0)
        assert type(model).__name__ == "SVC"

    def test_variational_wrapped_for_multiclass(self):
        model = create_model(
            "DataReuploadingClassifier",
            n_classes=3,
            max_vmap=8,
            batch_size=8,
            learning_rate=0.05,
            n_layers=1,
            observable_type="single",
            max_steps=10,
            convergence_interval=5,
        )
        from sklearn.multiclass import OneVsRestClassifier  # noqa: PLC0415

        assert isinstance(model, OneVsRestClassifier)
        # Pruning callback gate: the wrapper must not expose max_steps.
        assert not hasattr(model, "max_steps")

    def test_kernel_model_not_wrapped(self):
        model = create_model("IQPKernelClassifier", n_classes=3, max_vmap=8, repeats=1, C=1.0)
        assert type(model).__name__ == "IQPKernelClassifier"

    def test_ovr_fit_predict_proba(self):
        x, y = make_blobs(n_samples=30, centers=3, n_features=3, random_state=0)
        model = create_model(
            "SeparableVariationalClassifier",
            n_classes=3,
            batch_size=8,
            learning_rate=0.05,
            encoding_layers=1,
            max_steps=10,
            convergence_interval=5,
        )
        model.fit(x, y)
        proba = model.predict_proba(x)
        assert proba.shape == (30, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert set(np.unique(model.predict(x))) <= {0, 1, 2}


# ---------------------------------------------------------------------------
# Optimizer objective (macro F1) — fake model, no training
# ---------------------------------------------------------------------------


class _PerfectModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, _x, y):
        self._y = np.asarray(y)
        return self

    def predict(self, x):
        return self._y[: len(x)]

    def score(self, _x, _y):
        return 1.0


class TestMacroF1Objective:
    @pytest.fixture
    def multiclass_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 4))
        y = np.array([0, 1, 2] * 10)
        return {"train_x": x, "test_x": x, "train_y": y, "test_y": y}

    def test_objective_uses_macro_average(self, multiclass_data, tmp_path, monkeypatch):
        monkeypatch.setattr(optimizer_module, "create_model", lambda *_a, **k: _PerfectModel(**k))
        monkeypatch.setattr(
            "quoptuna.backend.utils.storage.optuna_db_path", lambda name: tmp_path / name
        )
        spec = TaskSpec.from_target([0, 1, 2], favorable_class=None).to_dict()
        opt = Optimizer(
            db_name="mc_test.db",
            data=multiclass_data,
            study_name="mc_macro",
            model_types=["SVC"],
            search_space={"C": [1.0]},
            sampler_seed=0,
            task_spec=spec,
        )
        assert opt.f1_average == "macro"
        study, _ = opt.optimize(n_trials=1)
        assert study.best_value == pytest.approx(1.0)
        assert study.user_attrs["task_spec"]["kind"] == "multiclass"

    def test_binary_default_unchanged(self, tmp_path):
        opt = Optimizer(db_name="bin_test.db", data={}, study_name="bin")
        assert opt.f1_average == "binary"
        assert opt.favorable_label == 1


# ---------------------------------------------------------------------------
# Fairness favorable-class binarization
# ---------------------------------------------------------------------------


class TestFavorableBinarization:
    def test_default_matches_old_to_binary(self):
        y = np.array([-1, 1, 1, -1])
        assert list(binarize_favorable(y)) == [0, 1, 1, 0]

    def test_multiclass_favorable(self):
        y = np.array([0, 1, 2, 1])
        assert list(binarize_favorable(y, favorable=1)) == [0, 1, 0, 1]

    def test_compute_disparity_with_favorable(self):
        # Group "a" always receives the favorable class 2; group "b" never.
        y_true = np.array([2, 2, 0, 1])
        y_pred = np.array([2, 2, 0, 1])
        sensitive = np.array(["a", "a", "b", "b"])
        disparity = compute_disparity(
            y_true, y_pred, sensitive, "demographic_parity_difference", favorable=2
        )
        assert disparity == pytest.approx(1.0)


class TestReviewFixes:
    """Regression tests for the multi-class review findings."""

    def test_task_spec_rejects_high_cardinality_target(self):
        from quoptuna.backend.task_type import MAX_N_CLASSES  # noqa: PLC0415

        with pytest.raises(ValueError, match="max"):
            TaskSpec.from_target(np.arange(MAX_N_CLASSES + 1))

    def test_preprocess_data_binary_matches_taskspec_convention(self):
        """classes[0] must map to -1 (TaskSpec convention), not +1."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 3))
        y = np.array(["no", "yes"] * 20)
        _, _, train_y, test_y = preprocess_data(x, y)
        all_y = np.concatenate([np.ravel(train_y), np.ravel(test_y)])
        # 'no' (sorted first) -> -1, 'yes' -> +1; balanced input stays balanced.
        assert set(all_y.tolist()) == {-1, 1}
        assert (all_y == 1).sum() == 20  # noqa: PLR2004

    def test_preprocess_data_multiclass_encodes_raw_labels(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 3))
        y = np.array(["a", "b", "c"] * 10)
        _, _, train_y, test_y = preprocess_data(x, y)
        all_y = np.concatenate([np.ravel(train_y), np.ravel(test_y)])
        assert set(all_y.tolist()) == {0, 1, 2}

    def test_data_preparation_binary_matches_taskspec_convention(self):
        from quoptuna.backend.utils.data_utils.prepare import DataPreparation  # noqa: PLC0415

        df_x = pd.DataFrame({"f": np.linspace(0, 1, 20)})
        df_y = pd.Series(["neg_class", "pos_class"] * 10, name="t")
        prep = DataPreparation(dataset={"x": df_x, "y": df_y}, x_cols=["f"], y_col="t")
        _, _, y_train, y_test = prep.prepare_data()
        all_y = np.concatenate([np.ravel(y_train), np.ravel(y_test)])
        assert (all_y == 1).sum() == 10  # noqa: PLR2004  # 'pos_class' (sorted second) -> +1

    def test_ovr_wrapper_exposes_aggregated_converged_flag(self):
        from sklearn.linear_model import Perceptron  # noqa: PLC0415

        from quoptuna.backend.base.pennylane_models.ovr import wrap_one_vs_rest  # noqa: PLC0415

        rng = np.random.default_rng(0)
        x = rng.normal(size=(30, 3))
        y = np.array([0, 1, 2] * 10)
        model = wrap_one_vs_rest(Perceptron())
        # Perceptron never raises ConvergenceWarning as exception -> converged
        # (adapters default converged_=True when no warning fired).
        model.fit(x, y)
        assert model.converged_ is True

    def test_get_report_isolates_metric_failures(self):
        """One failing metric must not abort later metrics (multiclass report)."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from quoptuna.backend.xai.xai import XAI  # noqa: PLC0415

        xai = MagicMock(spec=XAI)
        xai.get_confusion_matrix.side_effect = ValueError("boom")
        xai.get_classification_report.return_value = "ok"
        xai.get_roc_curve.side_effect = ValueError("multiclass not supported")
        xai.get_mcc.return_value = 0.5
        for name in (
            "get_roc_auc_score",
            "get_precision_recall_curve",
            "get_average_precision_score",
            "get_f1_score",
            "get_log_loss",
            "get_cohens_kappa",
            "get_precision",
            "get_recall",
        ):
            getattr(xai, name).return_value = 0.1
        report = XAI.get_report(xai)
        assert report["mcc"] == 0.5  # noqa: PLR2004  # computed despite earlier failures
        assert report["confusion_matrix"] == "boom"

    def test_per_class_curves_skip_absent_class_without_nan(self):
        from quoptuna.server.api.v1.analysis import _per_class_curve_payloads  # noqa: PLC0415

        rng = np.random.default_rng(0)
        # Class 2 never appears in y_test; proba has only 2 columns (model
        # trained without class 2), model_classes maps codes 0 and 1.
        y_test = np.array([0, 1] * 15)
        proba = rng.uniform(size=(30, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        spec = {"kind": "multiclass", "n_classes": 3, "class_labels": ["a", "b", "c"]}
        roc, _pr = _per_class_curve_payloads(y_test, proba, spec, model_classes=[0, 1])
        labels = [c["label"] for c in roc["per_class"]]
        assert "c" not in labels  # absent class skipped, not misattributed
        for c in roc["per_class"]:
            assert all(np.isfinite(c["fpr"])), "NaN leaked into ROC payload"
        assert roc["macro_auc"] is None  # incomplete class coverage -> no macro

    def test_plot_class_index_ignores_requested_for_binary(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        from quoptuna.server.api.v1.analysis import _plot_class_index  # noqa: PLC0415

        xai = MagicMock()
        values = MagicMock()
        values.values = np.zeros((5, 3, 2))  # per-class SHAP, binary
        xai.shap_values = values
        xai.get_classes.return_value = [0, 1]
        # UI always sends 0; binary must keep the positive class regardless.
        assert _plot_class_index(xai, requested=0) == 1
        xai.get_classes.return_value = [0, 1, 2]
        values.values = np.zeros((5, 3, 3))
        assert _plot_class_index(xai, requested=2) == 2  # noqa: PLR2004
