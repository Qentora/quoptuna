import logging
import time
from typing import Optional

import numpy as np
from optuna import Trial, TrialPruned, create_study, load_study
from optuna.pruners import HyperbandPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score

from quoptuna.backend.models import create_model
from quoptuna.backend.utils.data_utils.data import load_data, preprocess_data
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
    "max_vmap": [1],
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
        pruner: str = "none",
        pruner_min_resource: int = 1,
        pruner_reduction_factor: int = 3,
        intermediate_metric: str = "accuracy",
        max_steps: Optional[int] = None,  # noqa: FA100
        convergence_interval: Optional[int] = None,  # noqa: FA100
        max_vmap: Optional[int] = None,  # noqa: FA100
        fairness_mode: str = "off",
        fairness_metric: str = "equal_opportunity_difference",
        fairness_threshold: Optional[float] = None,  # noqa: FA100
        sensitive_test: Optional[np.ndarray] = None,  # noqa: FA100
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
            pruner: Optuna pruner: "none" (default), "asha" (asynchronous
                successive halving), or "hyperband".
            pruner_min_resource: ASHA/Hyperband minimum resource (rung 0), in
                units of intermediate reports.
            pruner_reduction_factor: ASHA/Hyperband reduction factor.
            intermediate_metric: What iterative models report for pruning:
                "accuracy" (validation accuracy, comparable across model types)
                or "neg_loss" (negated recent training loss; cheaper, but only
                meaningful when a single model type is searched).
            max_steps: Optional cap on training steps for iterative models.
            convergence_interval: Optional override of the models' flat-loss
                convergence window (also the cadence of pruning reports).
            max_vmap: Optional override of the ``max_vmap`` search-space entry
                (circuit evaluations vectorized per JAX call). Must divide the
                batch size.
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
        self.train_x = self.data.get("train_x")
        self.test_x = self.data.get("test_x")
        self.train_y = self.data.get("train_y")
        self.test_y = self.data.get("test_y")
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
        if max_vmap is not None and "max_vmap" in self.search_space:
            # Override the vectorization width without touching other entries;
            # objective() keeps sampling it like any other hyperparameter.
            self.search_space = {**self.search_space, "max_vmap": [max_vmap]}
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
            # Optuna raises at trial.report()/should_prune() time for
            # multi-objective studies; fail at config time instead.
            msg = "fairness_mode='multi_objective' does not support pruning; set pruner='none'"
            raise ValueError(msg)

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
        self.train_x, self.test_x, self.train_y, self.test_y = preprocess_data(self.X, self.y)

    def objective(self, trial: Trial):
        model = None
        try:
            # Sample one value per hyperparameter from the (possibly reduced)
            # search space. Insertion order is preserved for reproducibility.
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in self.search_space.items()
            }

            model_type = trial.suggest_categorical("model_type", self.model_types)

            model = create_model(
                model_type,
                max_steps=self.max_steps,
                convergence_interval=self.convergence_interval,
                **params,
            )

            # Iterative (JAX-trained) models report intermediate values so the
            # pruner can stop unpromising trials early. Kernel/sklearn models
            # have no training steps and simply run to completion.
            if self.pruner != "none" and hasattr(model, "max_steps"):
                model.training_callback = self._make_pruning_callback(trial, model)

            try:
                model.fit(self.train_x, self.train_y)
                trial.set_user_attr(key="converged", value=True)
            except ConvergenceWarning:
                # The training loop hit max_steps without meeting the flat-loss
                # criterion. The partially-trained parameters are still valid —
                # score the model instead of discarding max_steps of compute.
                logger.warning("Trial %s did not converge; scoring anyway", trial.number)
                trial.set_user_attr(key="converged", value=False)
            score = model.score(self.test_x, self.test_y)

            y_pred = model.predict(self.test_x)
            f_score_ = f1_score(self.test_y, y_pred)
            acc_ = accuracy_score(self.test_y, y_pred)

            self.log_user_attributes(
                model_type,
                {"accuracy": acc_, "f1_score": f_score_, "score": score},
                trial,
            )
            self._log_resource_attributes(trial, model)

            if self.fairness_mode != "off":
                try:
                    disparity = compute_disparity(
                        self.test_y, y_pred, self.sensitive_test, self.fairness_metric
                    )
                except Exception:  # noqa: BLE001 - any metric failure means "maximally unfair"
                    # e.g. a group with a single class in this trial's
                    # predictions — score it as maximally unfair, not failed.
                    logger.warning("Trial %s: fairness computation failed", trial.number)
                    disparity = WORST_DISPARITY
                trial.set_user_attr("fairness_disparity", float(disparity))
                trial.set_user_attr("fairness_metric", self.fairness_metric)
                if self.fairness_mode == "constrained":
                    trial.set_user_attr(
                        "fairness_constraint",
                        (float(disparity - self._disparity_threshold),),
                    )
                if self.fairness_mode == "multi_objective":
                    return f_score_, float(disparity)

            return f_score_  # noqa: TRY300
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
        test_x = self.test_x
        test_y = self.test_y
        if test_x is None or test_y is None:
            msg = "Pruning callback requires test features and labels"
            raise ValueError(msg)
        val_x = test_x[: self.VALIDATION_SUBSET_SIZE]
        val_y = test_y[: self.VALIDATION_SUBSET_SIZE]
        report_count = {"n": 0}

        def callback(step, loss_history):
            if self.intermediate_metric == "neg_loss":
                window = getattr(model, "convergence_interval", 200)
                value = -float(np.mean(loss_history[-window:]))
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

    def optimize(self, n_trials=100):
        if (
            self.train_x is None
            or self.test_x is None
            or self.train_y is None
            or self.test_y is None
        ):
            self.load_and_preprocess_data()
        # database  stored in a db folder "db"

        # sqllite database

        study = self._create_or_load_study()
        self.study = study
        # `catch` turns objective exceptions into FAILED trials (with the error
        # recorded as a user attr) instead of aborting the whole study.
        study.optimize(self.objective, n_trials=n_trials, catch=(Exception,))
        return study, study.best_trials
