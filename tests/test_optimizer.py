"""Fast unit tests for ``quoptuna.backend.tuners.optimizer``.

These avoid real quantum/classical fits by stubbing ``create_model`` with a
tiny fake estimator, so they exercise the ``Optimizer`` control flow (sampling,
scoring, user-attribute logging, study creation/loading and ``optimize``)
without the cost of training. ``FixedTrial`` lets us drive ``objective``
deterministically through both the classical and quantum logging branches.
"""

import numpy as np
import optuna
import pytest
from optuna.trial import FixedTrial
from sklearn.exceptions import ConvergenceWarning

from quoptuna.backend.tuners import optimizer as optimizer_module
from quoptuna.backend.tuners.optimizer import (
    DEFAULT_MODEL_TYPES,
    DEFAULT_SEARCH_SPACE,
    Optimizer,
)

TINY_SEARCH_SPACE = {"C": [1.0]}
EXPECTED_TRIALS = 2
PRUNER_REDUCTION_FACTOR = 2
ITERATIVE_MAX_STEPS = 12
ITERATIVE_BATCH_SIZE = 4
FIRST_PRUNE_CALLBACK_STEP = 2
PRUNED_TRIAL_COUNT = 3
UNCONVERGED_MAX_STEPS = 12
CONVERGENCE_TEST_MAX_STEPS = 500
CONVERGENCE_TEST_INTERVAL = 50


class _FakeModel:
    """Minimal sklearn-like estimator that never actually trains."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, _x, _y):
        return self

    def predict(self, x):
        # Alternate labels so both classes appear and f1_score is well-defined.
        return np.resize(np.array([0, 1]), len(x))

    def score(self, _x, _y):
        return 0.5


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 4))
    labels = np.array([0, 1] * 10)
    return {
        "train_x": features,
        "test_x": features,
        "train_y": labels,
        "test_y": labels,
    }


@pytest.fixture(autouse=True)
def _in_tmp_cwd(monkeypatch, tmp_path):
    # Optimizer.__init__ creates a ./db directory; keep it inside tmp_path.
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_create_model(monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeModel(model_type=model_type, **kwargs),
    )


def test_defaults_are_consistent():
    assert "MLPClassifier" in DEFAULT_MODEL_TYPES
    # Regression guard for the MLP literal fix (must be a valid tuple literal).
    assert "(100,)" in DEFAULT_SEARCH_SPACE["hidden_layer_sizes"]


def test_init_with_dataset_name_sets_data_path():
    opt = Optimizer(db_name="unit", dataset_name="banknote")
    assert opt.db_name == "unit"
    assert opt.data_path.endswith("unit.db")
    assert opt.model_types == DEFAULT_MODEL_TYPES
    assert opt.search_space == DEFAULT_SEARCH_SPACE


def test_init_with_overrides():
    opt = Optimizer(
        db_name="unit",
        data={"train_x": [1]},
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    assert opt.model_types == ["SVC"]
    assert opt.search_space == TINY_SEARCH_SPACE
    assert opt.train_x == [1]


@pytest.mark.parametrize(
    ("model_type", "quantum"),
    [("SVC", False), ("DataReuploadingClassifier", True)],
)
def test_objective_scores_and_logs(tiny_data, fake_create_model, model_type, quantum):
    opt = Optimizer(
        db_name="unit",
        data=tiny_data,
        model_types=[model_type],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": model_type})
    value = opt.objective(trial)

    assert 0.0 <= value <= 1.0
    attrs = trial.user_attrs
    if quantum:
        assert attrs["Classical_accuracy"] == 0
        assert "Quantum_accuracy" in attrs
    else:
        assert attrs["Quantum_accuracy"] == 0
        assert "Classical_accuracy" in attrs


def test_objective_reraises_on_error(tiny_data, monkeypatch):
    """Errors propagate (study.optimize's `catch` marks the trial FAILED) and
    the reason is recorded, instead of silently scoring 0.0."""

    def _boom(*_args, **_kwargs):
        msg = "model construction failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(optimizer_module, "create_model", _boom)
    opt = Optimizer(
        db_name="unit",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": "SVC"})
    with pytest.raises(RuntimeError, match="model construction failed"):
        opt.objective(trial)
    assert "model construction failed" in trial.user_attrs["error"]


def test_optimize_then_load_study(tiny_data, fake_create_model):
    opt = Optimizer(
        db_name="unit_opt",
        study_name="unit_study",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    study, best_trials = opt.optimize(n_trials=EXPECTED_TRIALS)

    assert opt.study is study
    assert len(study.trials) == EXPECTED_TRIALS
    assert best_trials is not None

    # Re-loading the persisted study should not raise and points at the same run.
    opt.study = None
    opt.load_study()
    assert opt.study is not None
    assert opt.study.study_name == "unit_study"


def test_create_or_load_study_is_idempotent(fake_create_model):
    opt = Optimizer(db_name="unit_idem", study_name="reused")
    first = opt._create_or_load_study()
    # ``load_if_exists`` means a second call reuses the same study instead of raising.
    second = opt._create_or_load_study()
    assert first.study_name == second.study_name == "reused"


def test_build_sampler_variants():
    for name, cls_name in [
        ("tpe", "TPESampler"),
        ("random", "RandomSampler"),
        ("grid", "GridSampler"),
    ]:
        opt = Optimizer(
            db_name="unit",
            sampler=name,
            model_types=["SVC"],
            search_space=TINY_SEARCH_SPACE,
        )
        assert type(opt._build_sampler()).__name__ == cls_name
    with pytest.raises(ValueError, match="Unknown sampler"):
        Optimizer(db_name="unit", sampler="bogus")._build_sampler()


def test_build_pruner_variants():
    for name, cls_name in [
        ("none", "NopPruner"),
        ("asha", "SuccessiveHalvingPruner"),
        ("hyperband", "HyperbandPruner"),
    ]:
        opt = Optimizer(db_name="unit", pruner=name)
        assert type(opt._build_pruner()).__name__ == cls_name
    with pytest.raises(ValueError, match="Unknown pruner"):
        Optimizer(db_name="unit", pruner="bogus")._build_pruner()


def test_default_pruner_is_asha():
    opt = Optimizer(db_name="unit_default_pruner")
    assert opt.pruner == "asha"
    assert type(opt._build_pruner()).__name__ == "SuccessiveHalvingPruner"


class _FakeIterativeModel(_FakeModel):
    """Fake with the iterative-training surface (max_steps + callback use)."""

    max_steps = ITERATIVE_MAX_STEPS
    convergence_interval = 3
    batch_size = ITERATIVE_BATCH_SIZE

    def fit(self, x, _y):
        callback = getattr(self, "training_callback", None)
        loss_history = []
        for step in range(self.max_steps):
            loss_history.append(1.0 / (step + 1))
            if callback is not None and (step + 1) % self.convergence_interval == 0:
                callback(step, loss_history)
        self.loss_history_ = np.array(loss_history)
        self.training_time_ = 0.1
        return self


def test_optimize_with_asha_prunes_trials(tiny_data, monkeypatch):
    """With an aggressively-pruning study, iterative models get a callback,
    pruned trials end PRUNED (not FAILED) and still carry resource attrs."""
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeIterativeModel(model_type=model_type, **kwargs),
    )

    opt = Optimizer(
        db_name="unit_asha",
        study_name="asha_study",
        data=tiny_data,
        model_types=["DataReuploadingClassifier"],
        search_space={"C": [0.1, 1.0, 10.0]},
        sampler="random",
        sampler_seed=0,
        pruner="asha",
        pruner_min_resource=1,
        pruner_reduction_factor=PRUNER_REDUCTION_FACTOR,
        intermediate_metric="neg_loss",
    )
    # Force pruning deterministically: every should_prune check says yes.
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda _self: True)

    study, _ = opt.optimize(n_trials=PRUNED_TRIAL_COUNT)
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) == PRUNED_TRIAL_COUNT
    for t in pruned:
        assert t.user_attrs["pruned"] is True
        assert t.user_attrs["pruned_at_step"] == FIRST_PRUNE_CALLBACK_STEP
        assert "error" not in t.user_attrs


def test_optimize_without_pruner_attaches_no_callback(tiny_data, monkeypatch):
    created = []

    def _factory(model_type, **kwargs):
        model = _FakeIterativeModel(model_type=model_type, **kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(optimizer_module, "create_model", _factory)
    opt = Optimizer(
        db_name="unit_nopruner",
        study_name="nopruner_study",
        data=tiny_data,
        model_types=["DataReuploadingClassifier"],
        search_space=TINY_SEARCH_SPACE,
        pruner="none",
    )
    study, _ = opt.optimize(n_trials=1)
    assert all(not hasattr(m, "training_callback") for m in created)
    (trial,) = study.trials
    # Resource accounting still recorded for completed trials.
    assert trial.user_attrs["pruned"] is False
    assert trial.user_attrs["n_steps"] == ITERATIVE_MAX_STEPS
    assert trial.user_attrs["batch_size"] == ITERATIVE_BATCH_SIZE


def test_pruner_reports_use_report_index(tiny_data, monkeypatch):
    """ASHA rungs must be fed the report index (0,1,2,...), not raw training
    steps, so min_resource/reduction_factor mean 'number of reports'."""
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeIterativeModel(model_type=model_type, **kwargs),
    )
    opt = Optimizer(
        db_name="unit_rung",
        study_name="rung_study",
        data=tiny_data,
        model_types=["DataReuploadingClassifier"],
        search_space=TINY_SEARCH_SPACE,
        pruner="asha",
        intermediate_metric="neg_loss",
    )
    study, _ = opt.optimize(n_trials=1)
    (trial,) = study.trials
    # max_steps=12, interval=3 -> 4 reports at indices 0..3 (not steps 2,5,8,11).
    assert sorted(trial.intermediate_values.keys()) == [0, 1, 2, 3]


def test_conditional_search_space_suggests_only_model_params(tiny_data, monkeypatch):
    """Each trial's params must be limited to the sampled model's whitelist."""
    from quoptuna.backend.models import MODEL_PARAM_KEYS  # noqa: PLC0415

    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeModel(model_type=model_type),
    )
    opt = Optimizer(
        db_name="unit_cond",
        study_name="cond_study",
        data=tiny_data,
        model_types=["SVC", "Perceptron", "DataReuploadingClassifier"],
        sampler="random",
        sampler_seed=0,
    )
    study, _ = opt.optimize(n_trials=6)
    for trial in study.trials:
        model_type = trial.params["model_type"]
        allowed = set(MODEL_PARAM_KEYS[model_type]) | {"model_type"}
        assert set(trial.params) <= allowed
        # And nothing the model uses (that the space covers) is missing.
        expected = {k for k in MODEL_PARAM_KEYS[model_type] if k in DEFAULT_SEARCH_SPACE}
        assert expected <= set(trial.params)


def test_validation_split_carved_from_train(tiny_data):
    opt = Optimizer(db_name="unit_val", data=tiny_data)
    opt._ensure_validation_split()
    # 20 train rows -> 16 train / 4 val, stratified on the balanced labels.
    assert len(opt.train_x) == 16  # noqa: PLR2004
    assert len(opt.val_x) == 4  # noqa: PLR2004
    assert set(np.asarray(opt.val_y).tolist()) == {0, 1}
    # Test split untouched, and the call is idempotent.
    assert len(opt.test_x) == len(tiny_data["test_x"])
    train_before = opt.train_x
    opt._ensure_validation_split()
    assert opt.train_x is train_before


def test_validation_split_falls_back_to_test_on_tiny_train():
    rng = np.random.default_rng(0)
    data = {
        "train_x": rng.normal(size=(6, 2)),
        "train_y": np.array([0, 1] * 3),
        "test_x": rng.normal(size=(4, 2)),
        "test_y": np.array([0, 1, 0, 1]),
    }
    opt = Optimizer(db_name="unit_val_tiny", data=data)
    opt._ensure_validation_split()
    assert opt.val_x is opt.test_x
    assert opt.val_y is opt.test_y


def test_stratified_split_preserves_minority_class():
    from quoptuna.backend.utils.data_utils.data import (  # noqa: PLC0415
        stratified_train_test_split,
    )

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2))
    y = np.array([1] * 8 + [0] * 32)  # 20% minority
    _, _, y_train, y_test = stratified_train_test_split(x, y, random_state=42)
    assert (np.asarray(y_train) == 1).sum() > 0
    assert (np.asarray(y_test) == 1).sum() > 0
    # Single-member class: falls back to unstratified without raising.
    y_degenerate = np.array([1] + [0] * 39)
    stratified_train_test_split(x, y_degenerate, random_state=42)


def test_objective_returns_validation_f1(tiny_data, monkeypatch):
    from sklearn.metrics import f1_score  # noqa: PLC0415

    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeModel(model_type=model_type),
    )
    opt = Optimizer(
        db_name="unit_valobj",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": "SVC"})
    value = opt.objective(trial)
    fake_pred = np.resize(np.array([0, 1]), len(opt.val_x))
    assert value == pytest.approx(f1_score(opt.val_y, fake_pred, zero_division=0))
    # Test metrics still logged under the existing (prefixed) names.
    attrs = trial.user_attrs
    test_pred = np.resize(np.array([0, 1]), len(opt.test_x))
    assert attrs["Classical_f1_score"] == pytest.approx(f1_score(opt.test_y, test_pred))
    assert attrs["val_f1_score"] == pytest.approx(value)


def test_class_weight_reaches_classical_models():
    from quoptuna.backend.models import create_model  # noqa: PLC0415

    assert "class_weight" in DEFAULT_SEARCH_SPACE
    svc = create_model("SVC", C=1.0, gamma=0.1, class_weight="balanced")
    assert svc.class_weight == "balanced"
    # Default path (key absent -> None) unchanged.
    assert create_model("SVC", C=1.0, gamma=0.1).class_weight is None


def test_decision_threshold_tuning_improves_val_f1(monkeypatch):
    """A model with well-ranked probabilities but a bad 0.5 cutoff must get a
    tuned threshold and a higher objective value."""

    class _ProbaModel(_FakeModel):
        def fit(self, _x, _y):
            return self

        def predict(self, x):
            # Default cutoff predicts all-negative -> F1 = 0.
            return np.zeros(len(x))

        def predict_proba(self, x):
            # Perfectly ranked but compressed probabilities: the first
            # feature encodes the true label (positives ~0.4, negatives ~0.2)
            # so ranking survives any row shuffling by the splits.
            pos = np.where(np.asarray(x)[:, 0] > 0.5, 0.4, 0.2)  # noqa: PLR2004
            return np.column_stack([1 - pos, pos])

    labels = np.array([0, 1] * 10)
    features = np.column_stack([labels.astype(float), np.zeros(20)])
    data = {"train_x": features, "train_y": labels, "test_x": features, "test_y": labels}
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _ProbaModel(model_type=model_type),
    )
    opt = Optimizer(
        db_name="unit_thresh",
        data=data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": "SVC"})
    value = opt.objective(trial)
    attrs = trial.user_attrs
    assert attrs["val_f1_unthresholded"] == 0.0
    assert 0.2 < attrs["decision_threshold"] <= 0.4  # noqa: PLR2004
    assert value == 1.0  # threshold between the two bands separates perfectly
    assert attrs["f1_score_thresholded"] == 1.0


def test_threshold_tuning_skipped_without_predict_proba(tiny_data, monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeModel(model_type=model_type),
    )
    opt = Optimizer(
        db_name="unit_nothresh",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
    )
    trial = FixedTrial({"C": 1.0, "model_type": "SVC"})
    opt.objective(trial)
    assert "decision_threshold" not in trial.user_attrs


def test_default_intermediate_metric_is_f1():
    assert Optimizer(db_name="unit_im").intermediate_metric == "f1"


def test_f1_intermediate_metric_reports_f1_values(tiny_data, monkeypatch):
    """The 'f1' intermediate metric must report validation F1, not accuracy."""
    from sklearn.metrics import f1_score  # noqa: PLC0415

    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeIterativeModel(model_type=model_type, **kwargs),
    )
    opt = Optimizer(
        db_name="unit_f1im",
        study_name="f1im_study",
        data=tiny_data,
        model_types=["DataReuploadingClassifier"],
        search_space=TINY_SEARCH_SPACE,
        pruner="asha",
        intermediate_metric="f1",
    )
    study, _ = opt.optimize(n_trials=1)
    (trial,) = study.trials
    # _FakeIterativeModel.predict alternates [0,1,...]; expected F1 on the
    # 128-row-capped validation split (carved from train).
    val_x = opt.val_x[: Optimizer.VALIDATION_SUBSET_SIZE]
    val_y = opt.val_y[: Optimizer.VALIDATION_SUBSET_SIZE]
    expected = f1_score(val_y, np.resize(np.array([0, 1]), len(val_x)), zero_division=0)
    assert trial.intermediate_values[0] == pytest.approx(expected)


def test_unconverged_trial_is_scored_not_failed(tiny_data, monkeypatch):
    """ConvergenceWarning from the training loop must not waste the trial:
    the partially-trained model is scored and the trial completes."""

    class _UnconvergedModel(_FakeModel):
        max_steps = UNCONVERGED_MAX_STEPS

        def fit(self, _x, _y):
            self.loss_history_ = np.ones(self.max_steps)
            self.training_time_ = 0.05
            msg = "did not converge"
            raise ConvergenceWarning(msg)

    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _UnconvergedModel(model_type=model_type, **kwargs),
    )
    opt = Optimizer(
        db_name="unit_noconv",
        study_name="noconv_study",
        data=tiny_data,
        model_types=["DataReuploadingClassifier"],
        search_space=TINY_SEARCH_SPACE,
    )
    study, _ = opt.optimize(n_trials=1)
    (trial,) = study.trials
    assert trial.state.name == "COMPLETE"
    assert trial.value is not None
    assert trial.user_attrs["converged"] is False
    assert trial.user_attrs["n_steps"] == UNCONVERGED_MAX_STEPS


def test_max_vmap_override_replaces_search_space_entry():
    opt = Optimizer(db_name="unit_vmap", max_vmap=32)
    assert opt.search_space["max_vmap"] == [32]
    # Other entries untouched.
    assert opt.search_space["batch_size"] == DEFAULT_SEARCH_SPACE["batch_size"]
    # No-op when the (custom) space has no max_vmap key.
    opt2 = Optimizer(db_name="unit_vmap2", search_space={"C": [1.0]}, max_vmap=32)
    assert "max_vmap" not in opt2.search_space


def test_default_max_vmap_vectorizes_full_batch():
    # Vectorized default: one vmap call per batch, not 32 size-1 calls.
    assert DEFAULT_SEARCH_SPACE["max_vmap"] == [32]
    for v in DEFAULT_SEARCH_SPACE["max_vmap"]:
        for b in DEFAULT_SEARCH_SPACE["batch_size"]:
            assert b % v == 0


def test_resume_with_conflicting_choices_fails_fast(tiny_data, monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "create_model",
        lambda model_type, **kwargs: _FakeModel(model_type=model_type),
    )
    common = {
        "db_name": "unit_resume",
        "study_name": "resume_conflict",
        "data": tiny_data,
        # Must be a model whose whitelist includes max_vmap, so the param is
        # actually suggested (conditional search space) and stored.
        "model_types": ["DataReuploadingClassifier"],
    }
    Optimizer(**common, search_space={"max_vmap": [1], "batch_size": [32]}).optimize(n_trials=1)
    # Same study, changed choices -> one clear error, before any trial runs.
    with pytest.raises(ValueError, match="new study name"):
        Optimizer(**common, search_space={"max_vmap": [32], "batch_size": [32]}).optimize(
            n_trials=1
        )
    # Identical choices resume fine.
    study, _ = Optimizer(**common, search_space={"max_vmap": [1], "batch_size": [32]}).optimize(
        n_trials=1
    )
    assert len(study.trials) == EXPECTED_TRIALS


def test_non_dividing_max_vmap_fails_fast():
    with pytest.raises(ValueError, match="must divide"):
        Optimizer(
            db_name="unit_vmap_bad",
            search_space={"max_vmap": [3], "batch_size": [32]},
        )
    # Override path is validated too.
    with pytest.raises(ValueError, match="must divide"):
        Optimizer(db_name="unit_vmap_bad2", max_vmap=5)


def test_convergence_interval_reaches_create_model(tiny_data, monkeypatch):
    captured = {}

    def _factory(model_type, **kwargs):
        captured.update(kwargs)
        return _FakeModel(model_type=model_type)

    monkeypatch.setattr(optimizer_module, "create_model", _factory)
    opt = Optimizer(
        db_name="unit_ci",
        data=tiny_data,
        model_types=["SVC"],
        search_space=TINY_SEARCH_SPACE,
        max_steps=CONVERGENCE_TEST_MAX_STEPS,
        convergence_interval=CONVERGENCE_TEST_INTERVAL,
    )
    trial = FixedTrial({"C": 1.0, "model_type": "SVC"})
    opt.objective(trial)
    assert captured["max_steps"] == CONVERGENCE_TEST_MAX_STEPS
    assert captured["convergence_interval"] == CONVERGENCE_TEST_INTERVAL
