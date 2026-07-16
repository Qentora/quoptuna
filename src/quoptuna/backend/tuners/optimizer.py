import logging
import time
from typing import Optional

import numpy as np
from optuna import Trial, TrialPruned, create_study, load_study
from optuna.distributions import CategoricalDistribution
from optuna.pruners import HyperbandPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score

from quoptuna.backend.base.pennylane_models.device import resolve_dev_type
from quoptuna.backend.models import MODEL_PARAM_KEYS, create_model
from quoptuna.backend.utils.data_utils.data import (
    load_data,
    preprocess_data,
    stratified_train_test_split,
)
from quoptuna.backend.utils.storage import ensure_db_dir, optuna_db_path
from quoptuna.backend.xai.fairness import FAIRNESS_METRICS, WORST_DISPARITY, compute_disparity

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Default hyperparameter search space. ``objective`` samples one value per key
# via ``trial.suggest_categorical``; insertion order is preserved so the
# sampling sequence is identical to the previous inline definition. Callers
# (e.g. tests) can pass a reduced ``search_space``/``model_types`` to the
# ``Optimizer`` to shrink the search and speed up runs.
DEFAULT_SEARCH_SPACE: dict = {
    # Every max_vmap choice must divide every batch_size choice (enforced at
    # fit time by model_utils.train); 32 vectorizes the whole default batch
    # in a single JAX vmap call instead of 32 size-1 circuit evaluations.
    "max_vmap": [32],
    "batch_size": [32],
    "learning_rate": [0.001, 0.01, 0.1],
    "n_input_copies": [1, 2, 3],
    "n_layers": [1, 5, 10],
    "observable_type": ["single", "half", "full"],
    "repeats": [1, 5, 10],
    "C": [0.1, 1, 10, 100],
    "gamma_factor": [0.1, 1, 10],
    "trotter_steps": [1, 3, 5],
    "t": [0.01, 0.1, 1.0],
    "n_qfeatures": ["full", "half"],
    "n_episodes": [10, 100, 500, 2000],
    "visible_qubits": ["single", "half", "full"],
    "temperature": [1, 10, 100],
    "encoding_layers": [1, 3, 5, 10],
    "degree": [2, 3, 4],
    "n_qchannels": [1, 5, 10],
    "qkernel_shape": [2, 3],
    "kernel_shape": [2, 3, 5],
    "filter_name": ["edge_detect", "smooth", "sharpen"],
    "gamma": [0.001, 0.01, 0.1, 1],
    "alpha": [0.01, 0.001, 0.0001],
    "hidden_layer_sizes": ["(100,)", "(10, 10, 10, 10)", "(50, 10, 5)"],
    "eta0": [0.1, 1, 10],
    # Imbalance handling for the classical models that accept it (only
    # suggested for those, via the conditional per-model search space).
    "class_weight": [None, "balanced"],
}

DEFAULT_MODEL_TYPES: list = [
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


class Optimizer:
    def __init__(  # noqa: PLR0913
        self,
        db_name: str,
        dataset_name: str = "",
        data: Optional[dict] = None,  # noqa: FA100
        study_name: str = "",
        model_types: Optional[list] = None,  # noqa: FA100
        search_space: Optional[dict] = None,  # noqa: FA100
        sampler: str = "tpe",
        sampler_seed: Optional[int] = None,  # noqa: FA100
        pruner: str = "asha",
        pruner_min_resource: int = 1,
        pruner_reduction_factor: int = 3,
        intermediate_metric: str = "f1",
        max_steps: Optional[int] = None,  # noqa: FA100
        convergence_interval: Optional[int] = None,  # noqa: FA100
        max_vmap: Optional[int] = None,  # noqa: FA100
        dev_type: str = "default.qubit",
        fairness_mode: str = "off",
        fairness_metric: str = "equal_opportunity_difference",
        fairness_threshold: Optional[float] = None,  # noqa: FA100
        sensitive_test: Optional[np.ndarray] = None,  # noqa: FA100
        task_spec: Optional[dict] = None,  # noqa: FA100
    ):
        """Initialize the Optimizer class.

        Args:
            db_name: The name of the database to be used for storing optimization results.
            dataset_name: The name of the dataset. If provided, the data will be loaded from a
                CSV file located in the 'notebook' directory. Defaults to an empty string.
            data: A dictionary containing training and testing data. If not provided, an empty
                dictionary will be used. Expected keys are 'train_x', 'test_x', 'train_y', and
                'test_y'.
            study_name: The name of the study for Optuna. Defaults to an empty string.
            sampler: Optuna sampler to use: "tpe" (default), "random", or "grid".
            sampler_seed: Optional seed for the sampler (reproducible searches).
            pruner: Optuna pruner: "asha" (default; asynchronous successive
                halving), "hyperband", or "none". Coerced to "none" when
                ``fairness_mode`` is "multi_objective" (pruning is
                unsupported on multi-objective studies).
            pruner_min_resource: ASHA/Hyperband minimum resource (rung 0), in
                units of intermediate reports.
            pruner_reduction_factor: ASHA/Hyperband reduction factor.
            intermediate_metric: What iterative models report for pruning:
                "f1" (default; validation F1 with the objective's averaging —
                keeps pruning aligned with the objective on imbalanced data),
                "accuracy" (validation accuracy), or "neg_loss" (negated
                recent training loss; cheaper, but only meaningful when a
                single model type is searched).
            max_steps: Optional cap on training steps for iterative models.
            convergence_interval: Optional override of the models' flat-loss
                convergence window (also the cadence of pruning reports).
            max_vmap: Optional override of the ``max_vmap`` search-space entry
                (circuit evaluations vectorized per JAX call). Must divide the
                batch size.
            dev_type: PennyLane simulator device for quantum models:
                "default.qubit" (default) or "lightning.qubit" (C++
                state-vector simulator, usually faster). Falls back to
                "default.qubit" with a warning if unavailable.
            fairness_mode: "off" (default), "constrained" (single-objective F1
                with a TPE feasibility constraint on the disparity), or
                "multi_objective" (maximize F1, minimize disparity; Pareto
                front returned via ``study.best_trials``).
            fairness_metric: Disparity metric driving the search — one of
                ``FAIRNESS_METRICS``.
            fairness_threshold: Constraint threshold. For difference metrics a
                trial is feasible when disparity <= threshold (default 0.1);
                for ``disparate_impact`` feasible when the DI ratio >=
                threshold (default 0.8, the four-fifths rule).
            sensitive_test: Sensitive-attribute values aligned positionally
                with the test split. Required when ``fairness_mode`` != "off".

        Attributes:
            db_name: The name of the database.
            dataset_name: The name of the dataset.
            data_path: The path to the dataset CSV file or an empty string if no dataset name is
                provided.
            data: The data dictionary containing training and testing data.
            train_x: The training features.
            test_x: The testing features.
            train_y: The training labels.
            test_y: The testing labels.
            storage_location: The storage location for the Optuna study.
            study_name: The name of the Optuna study.
            study: The Optuna study object.
        """
        self.db_name = db_name
        self.dataset_name = dataset_name
        if len(self.dataset_name) > 0:
            self.data_path = f"notebook/{self.dataset_name}.csv"
        else:
            self.data_path = ""
        self.data = data or {}  # Use an empty dictionary if no data is provided
        self.train_x, self.train_y = self.data.get("train_x"), self.data.get("train_y")
        self.test_x, self.test_y = self.data.get("test_x"), self.data.get("test_y")
        # Validation split (carved from train at optimize() time); used for
        # objective scoring and pruning reports. Test stays for reporting.
        self.val_x = self.val_y = None
        ensure_db_dir()
        self.data_path = str(optuna_db_path(self.db_name))
        self.storage_location = f"sqlite:///{self.data_path}"
        self.study_name = study_name
        self.study = None
        self.model_types = model_types or DEFAULT_MODEL_TYPES
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.sampler = sampler
        self.sampler_seed = sampler_seed
        self.pruner = pruner
        self.pruner_min_resource = pruner_min_resource
        self.pruner_reduction_factor = pruner_reduction_factor
        self.intermediate_metric = intermediate_metric
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self._disparity_threshold: float | None = None
        self._warned_pruner_noop = False
        if max_vmap is not None and "max_vmap" in self.search_space:
            # Override the vectorization width without touching other entries;
            # objective() keeps sampling it like any other hyperparameter.
            self.search_space = {**self.search_space, "max_vmap": [max_vmap]}
        self._validate_vmap_batch_divisibility()
        # Probe once per run, not per trial.
        self.dev_type = resolve_dev_type(dev_type)
        # Task spec (see backend.task_type.TaskSpec.to_dict): binary targets
        # use the {-1,+1} convention and binary F1; multiclass targets use
        # integer codes 0..K-1, OvR-wrapped variational models and macro-F1.
        self.task_spec = task_spec
        self.n_classes = int(task_spec["n_classes"]) if task_spec else 2
        self.f1_average = "macro" if self.n_classes > 2 else "binary"  # noqa: PLR2004
        self.favorable_label = (
            int(task_spec["favorable_code"])
            if task_spec and task_spec.get("favorable_code") is not None
            else 1
        )
        self.fairness_mode = fairness_mode
        self.fairness_metric = fairness_metric
        self.sensitive_test = None if sensitive_test is None else np.asarray(sensitive_test).ravel()
        self._validate_fairness_config()
        # Normalize the threshold into disparity space once (0 = parity), so
        # both modes share `disparity <= threshold` semantics: DI's ratio
        # threshold r becomes 1 - r; difference thresholds pass through.
        if self.fairness_mode != "off":
            if fairness_threshold is None:
                fairness_threshold = 0.8 if self.fairness_metric == "disparate_impact" else 0.1
            if self.fairness_metric == "disparate_impact":
                self._disparity_threshold = 1.0 - fairness_threshold
            else:
                self._disparity_threshold = fairness_threshold
        else:
            self._disparity_threshold = None

    def _validate_vmap_batch_divisibility(self):
        """Fail fast if any sampled max_vmap could not divide a batch_size.

        model_utils.train raises at fit time for non-dividing pairs, which
        would silently fail every trial of a study; catching it here surfaces
        the misconfiguration before any trial runs.
        """
        vmap_choices = self.search_space.get("max_vmap")
        batch_choices = self.search_space.get("batch_size")
        if not vmap_choices or not batch_choices:
            return
        bad = [
            (v, b)
            for v in vmap_choices
            for b in batch_choices
            if isinstance(v, int) and isinstance(b, int) and b % v != 0
        ]
        if bad:
            msg = (
                "Every max_vmap choice must divide every batch_size choice; "
                f"offending (max_vmap, batch_size) pairs: {bad}"
            )
            raise ValueError(msg)

    def _validate_fairness_config(self):
        if self.fairness_mode not in ("off", "constrained", "multi_objective"):
            msg = (
                f"Unknown fairness_mode: {self.fairness_mode!r} "
                "(expected 'off', 'constrained' or 'multi_objective')"
            )
            raise ValueError(msg)
        if self.fairness_mode == "off":
            return
        if self.fairness_metric not in FAIRNESS_METRICS:
            msg = (
                f"Unknown fairness_metric: {self.fairness_metric!r} "
                f"(expected one of {FAIRNESS_METRICS})"
            )
            raise ValueError(msg)
        if self.sensitive_test is None:
            msg = f"fairness_mode={self.fairness_mode!r} requires sensitive_test"
            raise ValueError(msg)
        if self.fairness_mode == "constrained" and self.sampler != "tpe":
            # Random/Grid samplers silently ignore constraints_func; an
            # experiment must not silently degrade to an unconstrained search.
            msg = "fairness_mode='constrained' requires sampler='tpe'"
            raise ValueError(msg)
        if self.fairness_mode == "multi_objective" and self.pruner != "none":
            # Optuna does not support pruning on multi-objective studies;
            # coerce (instead of rejecting) so the "asha" default still works.
            logger.warning(
                "Pruner %r is unsupported with fairness_mode='multi_objective'; using 'none'.",
                self.pruner,
            )
            self.pruner = "none"

    @staticmethod
    def _fairness_constraints(trial) -> tuple:
        # Consulted by TPE for completed trials only; a trial that failed
        # before recording the attr is treated as maximally infeasible.
        return tuple(trial.user_attrs.get("fairness_constraint", (WORST_DISPARITY,)))

    def _build_sampler(self):
        if self.sampler == "tpe":
            if self.fairness_mode == "constrained":
                return TPESampler(
                    seed=self.sampler_seed, constraints_func=self._fairness_constraints
                )
            return TPESampler(seed=self.sampler_seed)
        if self.sampler == "random":
            return RandomSampler(seed=self.sampler_seed)
        if self.sampler == "grid":
            # All hyperparameters are categorical, so the search space doubles
            # as a grid. Note the grid may be exhausted before n_trials.
            # The grid stays a flat snapshot even though the objective
            # suggests parameters conditionally per model: GridSampler
            # pre-assigns every grid key per trial and conditional suggests
            # simply consume the pre-assigned values (unsuggested keys are
            # ignored), so grid cells differing only in irrelevant params are
            # redundant but valid.
            return GridSampler(
                {**self.search_space, "model_type": self.model_types},
                seed=self.sampler_seed,
            )
        msg = f"Unknown sampler: {self.sampler!r} (expected 'tpe', 'random' or 'grid')"
        raise ValueError(msg)

    def _build_pruner(self):
        if self.pruner == "none":
            return NopPruner()
        if self.pruner == "asha":
            return SuccessiveHalvingPruner(
                min_resource=self.pruner_min_resource,
                reduction_factor=self.pruner_reduction_factor,
            )
        if self.pruner == "hyperband":
            return HyperbandPruner(
                min_resource=self.pruner_min_resource,
                reduction_factor=self.pruner_reduction_factor,
            )
        msg = f"Unknown pruner: {self.pruner!r} (expected 'none', 'asha' or 'hyperband')"
        raise ValueError(msg)

    def load_and_preprocess_data(self):
        self.X, self.y = load_data(self.data_path)
        # Standalone path (no server workflow): derive the task spec from the
        # raw target so multiclass CSVs get OvR wrapping and macro-F1 instead
        # of silently keeping the binary defaults (which fails every trial).
        if self.task_spec is None:
            from quoptuna.backend.task_type import TaskSpec  # noqa: PLC0415

            spec = TaskSpec.from_target(self.y)
            self.task_spec = spec.to_dict()
            self.n_classes = spec.n_classes
            self.f1_average = "macro" if spec.n_classes > 2 else "binary"  # noqa: PLR2004
        self.train_x, self.test_x, self.train_y, self.test_y = preprocess_data(self.X, self.y)

    def objective(self, trial: Trial):
        # Idempotent: covers direct objective() callers (tests, FixedTrial
        # evaluations) that bypass optimize().
        self._ensure_validation_split()
        model = None
        try:
            # Sample one value per hyperparameter from the (possibly reduced)
            # search space. Insertion order is preserved for reproducibility.
            # Conditional search space: suggest the model first, then only the
            # hyperparameters that model actually uses — irrelevant params
            # would otherwise pollute TPE's model of good/bad regions.
            # Iterating self.search_space preserves insertion order and the
            # intersection semantics of caller-reduced spaces. Unknown model
            # types (custom registrations in tests) fall back to the full
            # flat space.
            model_type = trial.suggest_categorical("model_type", self.model_types)
            allowed = MODEL_PARAM_KEYS.get(model_type)
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in self.search_space.items()
                if allowed is None or name in allowed
            }

            model = create_model(
                model_type,
                n_classes=self.n_classes,
                max_steps=self.max_steps,
                convergence_interval=self.convergence_interval,
                dev_type=self.dev_type,
                **params,
            )

            self._attach_pruning_callback(trial, model)
            self._fit_and_record_convergence(trial, model)

            # The objective is scored on the validation split so model
            # selection doesn't tune to the test set; test metrics are still
            # computed (one inference pass each — quantum predicts are
            # expensive) and logged under their existing user-attr names so
            # the Analyze tab and reports stay consistent.
            val_pred = model.predict(self.val_x)
            val_f1 = f1_score(self.val_y, val_pred, average=self.f1_average, zero_division=0)
            val_f1 = self._tune_decision_threshold(trial, model, float(val_f1))
            y_pred = model.predict(self.test_x)
            f_score_ = f1_score(self.test_y, y_pred, average=self.f1_average)
            acc_ = accuracy_score(self.test_y, y_pred)
            score = acc_

            self.log_user_attributes(
                model_type,
                {"accuracy": acc_, "f1_score": f_score_, "score": score},
                trial,
            )
            trial.set_user_attr("val_f1_score", float(val_f1))
            self._log_resource_attributes(trial, model)

            if self.fairness_mode != "off":
                # Fairness stays on the test split: sensitive_test is
                # positionally aligned to it, and no sensitive attribute is
                # available for the train-derived validation split.
                disparity = self._record_fairness(trial, y_pred)
                if self.fairness_mode == "multi_objective":
                    return val_f1, disparity

            return val_f1  # noqa: TRY300
        except TrialPruned:
            # Must be re-raised before the generic handler below: pruning is a
            # normal ASHA outcome, not a failure. Resource attrs are still
            # recorded so pruned trials count toward "quantum calls" totals.
            logger.info("Trial %s pruned", trial.number)
            self._log_resource_attributes(trial, model, pruned=True)
            raise
        except Exception as e:
            # Record why and re-raise so Optuna marks the trial FAILED (the
            # `catch` in study.optimize keeps the study going). Returning 0
            # here would make broken configurations look like real F1=0 runs.
            logger.exception("Trial %s failed", trial.number)
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            self._log_resource_attributes(trial, model)
            raise

    THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)

    def _tune_decision_threshold(self, trial: Trial, model, base_val_f1: float) -> float:
        """Sweep the binary decision threshold on the validation split.

        Unweighted losses on imbalanced data often yield well-ranked
        probabilities with a bad default 0.5 cutoff; tuning the cutoff on
        validation recovers minority-class F1 without touching training.
        Returns the best thresholded validation F1 (or ``base_val_f1``
        unchanged for multiclass / models without predict_proba / failures).
        The threshold is recorded as the ``decision_threshold`` user attr;
        the reported test-side ``f1_score`` attr stays unthresholded (Analyze
        recomputes without it), with ``f1_score_thresholded`` alongside.
        """
        if self.n_classes > 2 or not hasattr(model, "predict_proba"):  # noqa: PLR2004
            return base_val_f1
        try:
            classes = np.sort(np.unique(np.asarray(self.train_y).ravel()))
            neg_label, pos_label = classes[0], classes[-1]
            # predict_proba columns are ordered [P(neg), P(pos)] for the
            # {-1,+1} quantum models and follow classes_ for sklearn.
            val_proba = np.asarray(model.predict_proba(self.val_x))[:, 1]
            scores = [
                f1_score(
                    self.val_y,
                    np.where(val_proba >= t, pos_label, neg_label),
                    average=self.f1_average,
                    zero_division=0,
                )
                for t in self.THRESHOLD_GRID
            ]
            best_idx = int(np.argmax(scores))
            best_f1 = float(scores[best_idx])
            if best_f1 <= base_val_f1:
                return base_val_f1
            threshold = float(self.THRESHOLD_GRID[best_idx])
            test_proba = np.asarray(model.predict_proba(self.test_x))[:, 1]
            test_f1_thresholded = f1_score(
                self.test_y,
                np.where(test_proba >= threshold, pos_label, neg_label),
                average=self.f1_average,
                zero_division=0,
            )
            trial.set_user_attr("decision_threshold", threshold)
            trial.set_user_attr("val_f1_unthresholded", base_val_f1)
            trial.set_user_attr("f1_score_thresholded", float(test_f1_thresholded))
        except Exception as exc:  # noqa: BLE001 - tuning must never fail a trial
            logger.warning("Decision-threshold tuning skipped: %s", exc)
            return base_val_f1
        return best_f1

    def _attach_pruning_callback(self, trial: Trial, model) -> None:
        """Attach the per-step pruning hook where the model supports it.

        Iterative (JAX-trained) models report intermediate values so the
        pruner can stop unpromising trials early. Kernel/sklearn models (and
        OvR wrappers, whose K interleaved loss series are not comparable)
        have no per-step hook and run to completion.
        """
        if self.pruner == "none":
            return
        if hasattr(model, "max_steps"):
            model.training_callback = self._make_pruning_callback(trial, model)
        elif self.n_classes > 2 and not self._warned_pruner_noop:  # noqa: PLR2004
            self._warned_pruner_noop = True
            logger.warning(
                "Pruner %r has no effect on OvR-wrapped multiclass models: "
                "each trial trains all %d sub-models to completion.",
                self.pruner,
                self.n_classes,
            )

    def _fit_and_record_convergence(self, trial: Trial, model) -> None:
        """Fit the model, tolerating non-convergence, and record the outcome."""
        try:
            model.fit(self.train_x, self.train_y)
            # OvR wrappers swallow per-class ConvergenceWarnings and expose an
            # aggregated flag instead; unwrapped models raise directly.
            trial.set_user_attr(key="converged", value=bool(getattr(model, "converged_", True)))
        except ConvergenceWarning:
            # The training loop hit max_steps without meeting the flat-loss
            # criterion. The partially-trained parameters are still valid —
            # score the model instead of discarding max_steps of compute.
            logger.warning("Trial %s did not converge; scoring anyway", trial.number)
            trial.set_user_attr(key="converged", value=False)

    def _record_fairness(self, trial: Trial, y_pred) -> float:
        """Compute and record the trial's fairness disparity; return it."""
        try:
            disparity = compute_disparity(
                self.test_y,
                y_pred,
                self.sensitive_test,
                self.fairness_metric,
                favorable=self.favorable_label,
            )
        except Exception:  # noqa: BLE001 - any metric failure means "maximally unfair"
            # e.g. a group with a single class in this trial's predictions —
            # score it as maximally unfair, not failed.
            logger.warning("Trial %s: fairness computation failed", trial.number)
            disparity = WORST_DISPARITY
        trial.set_user_attr("fairness_disparity", float(disparity))
        trial.set_user_attr("fairness_metric", self.fairness_metric)
        if self.fairness_mode == "constrained":
            threshold = self._require_disparity_threshold()
            trial.set_user_attr("fairness_constraint", (float(disparity - threshold),))
        return float(disparity)

    # Validation subset cap for the "accuracy" intermediate metric. Quantum
    # predicts run in max_vmap-sized chunks, so evaluating the full test set at
    # every report can rival the cost of training itself for some models
    # (e.g. QuantumMetricLearner). A fixed prefix keeps values comparable
    # across trials while bounding the per-report circuit count.
    VALIDATION_SUBSET_SIZE = 128

    def _make_pruning_callback(self, trial: Trial, model):
        """Build the per-interval hook consumed by ``model_utils.train``.

        The pruner is fed the *report index* (0, 1, 2, ...), not the raw
        training step, so ``pruner_min_resource``/``pruner_reduction_factor``
        operate in units of intermediate reports as documented. Raw steps
        would let a single report jump several ASHA rungs at once.
        """
        if self.val_x is None or self.val_y is None:
            msg = "Pruning callback requires validation features and labels"
            raise ValueError(msg)
        val_x = self.val_x[: self.VALIDATION_SUBSET_SIZE]
        val_y = self.val_y[: self.VALIDATION_SUBSET_SIZE]
        report_count = {"n": 0}

        def callback(step, loss_history):
            if self.intermediate_metric == "neg_loss":
                window = getattr(model, "convergence_interval", 200)
                value = -float(np.mean(loss_history[-window:]))
            elif self.intermediate_metric == "f1":
                # Match the objective: on imbalanced data, accuracy would let
                # majority-class predictors survive pruning while the trials
                # actually learning the minority class get killed.
                value = float(
                    f1_score(
                        val_y,
                        model.predict(val_x),
                        average=self.f1_average,
                        zero_division=0,
                    )
                )
            else:
                value = float(accuracy_score(val_y, model.predict(val_x)))
            report_index = report_count["n"]
            report_count["n"] += 1
            trial.report(value, step=report_index)
            if trial.should_prune():
                trial.set_user_attr("pruned_at_step", int(step))
                msg = f"Pruned at step {step} ({self.intermediate_metric}={value:.4f})"
                raise TrialPruned(msg)

        return callback

    def _require_disparity_threshold(self) -> float:
        threshold = self._disparity_threshold
        if threshold is None:
            msg = "Constrained fairness mode requires a disparity threshold"
            raise RuntimeError(msg)
        return threshold

    def _log_resource_attributes(
        self,
        trial: Trial,
        model,
        *,
        pruned: bool = False,
    ):
        """Persist per-trial resource usage for time/shots-to-solution analysis."""
        if model is None:
            return
        trial.set_user_attr("pruned", pruned)
        training_time = getattr(model, "training_time_", None)
        if training_time is not None:
            trial.set_user_attr("training_time", float(training_time))
        loss_history = getattr(model, "loss_history_", None)
        if loss_history is not None:
            trial.set_user_attr("n_steps", len(loss_history))
        batch_size = getattr(model, "batch_size", None)
        if batch_size is not None:
            trial.set_user_attr("batch_size", int(batch_size))

    def log_user_attributes(self, model_type, eval_scores, trial):
        if model_type in ["SVC", "SVClinear", "MLPClassifier", "Perceptron"]:
            trial.set_user_attr("Classical_accuracy", eval_scores["accuracy"])
            trial.set_user_attr("Classical_f1_score", eval_scores["f1_score"])
            trial.set_user_attr("Classical_score", eval_scores["score"])
            trial.set_user_attr("Quantum_accuracy", 0)
            trial.set_user_attr("Quantum_f1_score", 0)
            trial.set_user_attr("Quantum_score", 0)
        else:
            trial.set_user_attr("Quantum_accuracy", eval_scores["accuracy"])
            trial.set_user_attr("Quantum_f1_score", eval_scores["f1_score"])
            trial.set_user_attr("Quantum_score", eval_scores["score"])
            trial.set_user_attr("Classical_accuracy", 0)
            trial.set_user_attr("Classical_f1_score", 0)
            trial.set_user_attr("Classical_score", 0)

    def load_study(self):
        # load the study from the database
        self.study = load_study(
            storage=self.storage_location,
            study_name=self.study_name,
        )

    def _create_or_load_study(self, retries: int = 5, delay: float = 0.3):
        """Create the study, tolerating reuse and concurrent schema creation.

        ``load_if_exists=True`` makes re-running an existing study name append
        trials instead of raising ``DuplicatedStudyError``. The retry loop guards
        against the SQLite race where a concurrent ``load_study`` (e.g. the live
        trials-polling endpoint) initializes the same brand-new database and both
        connections try to ``CREATE TABLE studies`` at once.
        """
        sampler = self._build_sampler()
        pruner = self._build_pruner()
        directions = (
            ["maximize", "minimize"] if self.fairness_mode == "multi_objective" else ["maximize"]
        )
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return create_study(
                    storage=self.storage_location,
                    sampler=sampler,
                    pruner=pruner,
                    directions=directions,
                    study_name=self.study_name,
                    load_if_exists=True,
                )
            except Exception as exc:  # noqa: PERF203 - resilient retry on storage races
                last_exc = exc
                message = str(exc).lower()
                if "already exists" in message or "locked" in message:
                    logger.warning(
                        "Study storage busy (attempt %s/%s): %s", attempt + 1, retries, exc
                    )
                    time.sleep(delay)
                    continue
                raise
        # Final fallback: the study/tables exist, so just load it (with the
        # same sampler/pruner — they are process-level, not persisted).
        try:
            return load_study(
                storage=self.storage_location,
                study_name=self.study_name,
                sampler=sampler,
                pruner=pruner,
            )
        except Exception:
            if last_exc is not None:
                raise last_exc from None
            raise

    def _check_resume_compatibility(self, study) -> None:
        """Fail fast when resuming a study whose stored choices conflict.

        Optuna forbids changing a parameter's categorical choices within a
        study ("CategoricalDistribution does not support dynamic value
        space"). Without this check, a study created under older defaults
        (e.g. max_vmap [1]) fails every new trial with that cryptic error;
        here we raise one clear, actionable error instead.
        """
        stored: dict = {}
        for t in study.trials:
            for name, dist in t.distributions.items():
                if isinstance(dist, CategoricalDistribution) and name not in stored:
                    stored[name] = list(dist.choices)
        if not stored:
            return
        current = {**self.search_space, "model_type": list(self.model_types)}
        conflicts = {
            name: (choices, list(current[name]))
            for name, choices in stored.items()
            if name in current and list(current[name]) != choices
        }
        if conflicts:
            detail = "; ".join(
                f"{name}: stored {old} vs current {new}"
                for name, (old, new) in sorted(conflicts.items())
            )
            msg = (
                f"Study {study.study_name!r} was created with different "
                f"hyperparameter choices ({detail}). An Optuna study's "
                "parameter choices are immutable — use a new study name "
                "(or pass a matching search_space) to continue."
            )
            raise ValueError(msg)

    MIN_TRAIN_ROWS_FOR_VAL_SPLIT = 10

    def _ensure_validation_split(self):
        """Carve a validation split out of TRAIN (idempotent).

        The objective and pruning reports are scored on this split so model
        selection doesn't tune to the test set (which Analyze/reports use for
        final metrics). On tiny datasets where a further split is not viable,
        fall back to validating on the test split (the previous behavior).
        """
        if self.val_x is not None:
            return
        train_y = np.asarray(self.train_y).ravel()
        if len(train_y) >= self.MIN_TRAIN_ROWS_FOR_VAL_SPLIT:
            try:
                self.train_x, self.val_x, self.train_y, self.val_y = stratified_train_test_split(
                    self.train_x, train_y, test_size=0.2, random_state=42
                )
            except ValueError:
                logger.warning("Validation split failed; validating on the test split.")
        else:
            logger.warning(
                "Train set too small (%d rows) for a validation split; "
                "validating on the test split.",
                len(train_y),
            )
        if self.val_x is None:
            self.val_x = self.test_x
            self.val_y = self.test_y

    def optimize(self, n_trials=100):
        """Run the study.

        Note: ``study.best_value`` is the *validation* F1; the corresponding
        test-set metrics live in each trial's user attrs (``f1_score``,
        ``accuracy``).
        """
        if (
            self.train_x is None
            or self.test_x is None
            or self.train_y is None
            or self.test_y is None
        ):
            self.load_and_preprocess_data()
        self._ensure_validation_split()
        # database  stored in a db folder "db"

        # sqllite database

        study = self._create_or_load_study()
        self.study = study
        self._check_resume_compatibility(study)
        if self.task_spec:
            # Persisted so analysis endpoints can recover the class structure
            # (kind, labels, favorable class) after a backend restart.
            study.set_user_attr("task_spec", self.task_spec)
        # `catch` turns objective exceptions into FAILED trials (with the error
        # recorded as a user attr) instead of aborting the whole study.
        study.optimize(self.objective, n_trials=n_trials, catch=(Exception,))
        return study, study.best_trials
