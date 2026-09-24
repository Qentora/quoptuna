"""
Workflow execution service

This service handles the execution of visual workflows created in the frontend.
It integrates with the existing quoptuna services (Optimizer, DataPreparation, XAI).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from quoptuna import XAI, DataPreparation, Optimizer, XAIConfig
from quoptuna.backend.utils.data_utils.data import stratified_train_test_split
from quoptuna.backend.utils.data_utils.resampling import resample_train_split
from quoptuna.backend.utils.storage import DEFAULT_DB_NAME
from quoptuna.server.services.sensitive import resolve_sensitive_series

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Raised when workflow execution fails"""


def study_best_trial(study):
    """Best trial for single- or multi-objective studies.

    ``study.best_trial`` raises for multi-objective studies; there we pick the
    Pareto-front point with the highest F1 (objective 0 in all modes).
    """
    if len(study.directions) > 1:
        return max(study.best_trials, key=lambda t: t.values[0])
    return study.best_trial


def _training_budget(opt_result: Dict[str, Any]) -> Dict[str, Any]:
    """Search-time knobs that are not part of ``trial.params``.

    ``Optimizer.objective`` passes ``max_steps`` / ``convergence_interval`` /
    ``dev_type`` to ``create_model`` alongside the sampled hyperparameters, but
    Optuna only records what it sampled. Retraining a trial without them
    rebuilds the JAX-trained models at their class-default training budget
    rather than the one the search actually used, so replay them here. Omit
    ``None`` so ``create_model`` keeps its own defaults.
    """
    budget = {
        "max_steps": opt_result.get("max_steps"),
        "convergence_interval": opt_result.get("convergence_interval"),
        "dev_type": opt_result.get("dev_type"),
    }
    return {key: value for key, value in budget.items() if value is not None}


#: Fraction of the train split held out for objective scoring, pruning reports
#: and the search-time fairness disparity. Matches the fallback carve in
#: ``Optimizer._ensure_validation_split`` so both paths behave identically.
VALIDATION_FRACTION = 0.2
#: Below this, a three-way split leaves too few validation rows to score; the
#: run keeps the two-way split and Optimizer falls back to validating on test.
MIN_TRAIN_ROWS_FOR_VAL_SPLIT = 10


def _carve_validation_split(x_train, y_train, sensitive_train=None):
    """Split train into (fit, validation), carrying the sensitive column along.

    Called BEFORE any resampling so no duplicated row can span the boundary.
    Returns ``(x_fit, x_val, y_fit, y_val, sensitive_fit, sensitive_val)``;
    ``x_val``/``y_val`` are ``None`` when the train split is too small to
    divide, which leaves the previous two-way behaviour intact.
    """
    y_arr = np.asarray(y_train).ravel()
    if len(y_arr) < MIN_TRAIN_ROWS_FOR_VAL_SPLIT:
        logger.warning(
            "Train split has %d rows — too few to hold out a validation set; "
            "the objective will be scored on the test split.",
            len(y_arr),
        )
        return x_train, None, y_train, None, sensitive_train, None

    # Index positionally so the sensitive series follows the exact same rows
    # regardless of how the frames are indexed.
    positions = np.arange(len(y_arr))
    try:
        fit_pos, val_pos = stratified_train_test_split(
            positions, y_arr, test_size=VALIDATION_FRACTION, random_state=42
        )[:2]
    except ValueError:
        logger.warning(
            "Validation carve failed (a class is too rare to stratify); "
            "the objective will be scored on the test split."
        )
        return x_train, None, y_train, None, sensitive_train, None

    def _take(obj, pos):
        if obj is None:
            return None
        return obj.iloc[pos] if hasattr(obj, "iloc") else np.asarray(obj)[pos]

    return (
        _take(x_train, fit_pos),
        _take(x_train, val_pos),
        _take(y_train, fit_pos),
        _take(y_train, val_pos),
        _take(sensitive_train, fit_pos),
        _take(sensitive_train, val_pos),
    )


def build_xai(
    opt_result: Dict[str, Any],
    trial_number: int | None = None,
    use_proba: bool = True,
    subset_size: int = 50,
    background_size: int | None = None,
    max_evals: int | None = None,
):
    """Retrain a study trial and build an ``XAI`` instance for it.

    Shared by the SHAP / metrics / report analysis endpoints. When
    ``trial_number`` is ``None`` the study's best trial is used.

    The refit uses ``opt_result["x_train"]`` — the same inner, resampled
    training frame the trial fitted on — and the trial's recorded
    ``decision_threshold``, so the analysed classifier is the one the search
    scored rather than a differently-trained model read at a 0.5 cutoff.
    """
    from optuna import load_study

    from quoptuna import XAI, XAIConfig
    from quoptuna.backend.models import create_model
    from quoptuna.server.services.storage import optuna_storage_url

    db_name = str(opt_result.get("db_name") or DEFAULT_DB_NAME)
    study_name = opt_result.get("study_name")
    storage_location = optuna_storage_url(db_name)
    study = load_study(storage=storage_location, study_name=study_name)

    if trial_number is not None:
        trial = next((t for t in study.trials if t.number == trial_number), None)
        if trial is None:
            raise WorkflowExecutionError(f"Trial {trial_number} not found in study")
    else:
        trial = study_best_trial(study)

    task_spec = opt_result.get("task_spec")
    n_classes = int(task_spec["n_classes"]) if task_spec else 2
    params = {k: v for k, v in trial.params.items() if k != "model_type"}
    model = create_model(
        trial.params["model_type"],
        n_classes=n_classes,
        **_training_budget(opt_result),
        **params,
    )

    x_train_df = opt_result["x_train"]
    y_train_df = opt_result["y_train"]
    x_train_np = x_train_df.values if hasattr(x_train_df, "values") else x_train_df
    # MUST be 1-D, exactly as the search fits it (_execute_optimization ravels
    # too). The label-encoding node stores y as a DataFrame, so ``.values``
    # alone yields (n, 1); training on that shape does not raise — it silently
    # collapses the model's probabilities into a narrow band around 0.5, which
    # left argmax predictions plausible while making any probability threshold
    # meaningless (every row on one side of it).
    y_train_np = np.asarray(y_train_df).ravel()
    model.fit(x_train_np, y_train_np)

    data_dict = {
        "x_train": x_train_df,
        "x_test": opt_result["x_test"],
        "y_train": y_train_df,
        "y_test": opt_result["y_test"],
    }

    # Absent for multiclass tasks, models without predict_proba, and trials
    # whose sweep found nothing better than the default cutoff.
    decision_threshold = trial.user_attrs.get("decision_threshold")

    xai_config = XAIConfig(
        use_proba=use_proba,
        onsubset=True,
        subset_size=subset_size,
        max_evals=max_evals,
        decision_threshold=decision_threshold,
        **({} if background_size is None else {"background_size": background_size}),
    )
    try:
        return XAI(model=model, data=data_dict, config=xai_config)
    except TypeError:
        if not use_proba:
            raise
        # Models with no predict_proba at all (LinearSVC, Perceptron) cannot
        # serve the probability mode. Explaining labels instead is the same
        # fallback the UI's "use probabilities" toggle offers, and is strictly
        # better than failing the whole analysis: the label-based metrics,
        # confusion matrix and SHAP values are all still valid.
        logger.warning(
            "%s has no predict_proba; analysing in label mode (probability metrics unavailable)",
            trial.params["model_type"],
        )
        xai_config.use_proba = False
        return XAI(model=model, data=data_dict, config=xai_config)


class WorkflowExecutor:
    """Executes visual workflows by running nodes in topological order"""

    def __init__(self, workflow: Dict[str, Any], upload_dir: str = "./uploads"):
        self.workflow = workflow
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        self.nodes = {node["id"]: node for node in workflow.get("nodes", [])}
        self.edges = workflow.get("edges", [])
        self.results: Dict[str, Any] = {}  # Store results from each node

    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get list of node IDs that this node depends on"""
        dependencies = []
        for edge in self.edges:
            if edge["target"] == node_id:
                dependencies.append(edge["source"])
        return dependencies

    def topological_sort(self) -> List[str]:
        """Sort nodes in execution order using topological sort"""
        # Build dependency graph
        in_degree = dict.fromkeys(self.nodes, 0)
        for edge in self.edges:
            in_degree[edge["target"]] += 1

        # Find nodes with no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        sorted_nodes = []

        while queue:
            node_id = queue.pop(0)
            sorted_nodes.append(node_id)

            # Reduce in-degree for dependent nodes
            for edge in self.edges:
                if edge["source"] == node_id:
                    in_degree[edge["target"]] -= 1
                    if in_degree[edge["target"]] == 0:
                        queue.append(edge["target"])

        if len(sorted_nodes) != len(self.nodes):
            raise WorkflowExecutionError("Workflow contains cycles")

        return sorted_nodes

    def execute_node(self, node_id: str) -> Any:
        """Execute a single node and return its result"""
        node = self.nodes[node_id]
        node_type = node["data"]["type"]
        config = node["data"].get("config", {})

        logger.info(f"Executing node {node_id} of type {node_type}")

        # Get input from dependencies
        dependencies = self.get_node_dependencies(node_id)
        inputs = {dep: self.results[dep] for dep in dependencies}

        # Execute based on node type
        if node_type == "data-upload":
            return self._execute_data_upload(config, inputs)
        if node_type == "data-uci":
            return self._execute_data_uci(config, inputs)
        if node_type == "data-preview":
            return self._execute_data_preview(config, inputs)
        if node_type == "feature-selection":
            return self._execute_feature_selection(config, inputs)
        if node_type == "train-test-split":
            return self._execute_train_test_split(config, inputs)
        if node_type == "scaler":
            return self._execute_scaler(config, inputs)
        if node_type == "label-encoding":
            return self._execute_label_encoding(config, inputs)
        if node_type in ["quantum-model", "classical-model"]:
            return self._execute_model_config(config, inputs, node_type)
        if node_type == "optuna-config":
            return self._execute_optuna_config(config, inputs)
        if node_type == "optimization":
            return self._execute_optimization(config, inputs)
        if node_type == "shap-analysis":
            return self._execute_shap_analysis(config, inputs)
        if node_type == "confusion-matrix":
            return self._execute_confusion_matrix(config, inputs)
        if node_type == "feature-importance":
            return self._execute_feature_importance(config, inputs)
        if node_type == "export-model":
            return self._execute_export_model(config, inputs)
        if node_type == "generate-report":
            return self._execute_generate_report(config, inputs)
        raise WorkflowExecutionError(f"Unknown node type: {node_type}")

    def _execute_data_upload(self, config: Dict, inputs: Dict) -> Dict:
        """Handle CSV file upload"""
        file_path = config.get("file_path")
        if not file_path:
            raise WorkflowExecutionError("No file path provided for data upload")

        df = pd.read_csv(file_path)
        return {
            "type": "dataset",
            "dataframe": df,
            "rows": len(df),
            "columns": list(df.columns),
        }

    def _execute_data_uci(self, config: Dict, inputs: Dict) -> Dict:
        """Fetch dataset from UCI repository"""
        dataset_id = config.get("dataset_id")
        if not dataset_id:
            raise WorkflowExecutionError("No dataset ID provided")

        # Fetch from UCI
        dataset = fetch_ucirepo(id=int(dataset_id))
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

        return {
            "type": "dataset",
            "dataframe": df,
            "rows": len(df),
            "columns": list(df.columns),
            "metadata": dataset.metadata,
        }

    def _execute_data_preview(self, config: Dict, inputs: Dict) -> Dict:
        """Generate dataset preview statistics"""
        if not inputs:
            raise WorkflowExecutionError("No input data for preview")

        dataset = list(inputs.values())[0]
        df = dataset["dataframe"]

        # Convert dtypes to strings for JSON serialization
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Convert describe() output, handling NaN values
        describe_dict = df.describe().fillna(0).to_dict()

        return {
            "type": "preview",
            "shape": list(df.shape),  # Convert tuple to list
            "dtypes": dtypes_dict,
            "describe": describe_dict,
            "head": df.head().to_dict(),
            "columns": list(df.columns),
            "rows": len(df),
        }

    def _execute_feature_selection(self, config: Dict, inputs: Dict) -> Dict:
        """Select features and target column"""
        if not inputs:
            raise WorkflowExecutionError("No input data for feature selection")

        dataset = list(inputs.values())[0]
        df = dataset["dataframe"]

        x_columns = config.get("x_columns", [])
        y_column = config.get("y_column")

        if not x_columns or not y_column:
            raise WorkflowExecutionError("Must specify x_columns and y_column")

        # Categorical features would otherwise reach StandardScaler / the models
        # as strings and fail every trial; encode them (and impute NaN) here so
        # the whole downstream pipeline only ever sees numeric columns.
        from quoptuna.backend.utils.data_utils.prepare import (
            encode_features,
            encoded_passthrough_columns,
        )

        method = config.get("categorical_encoding", "ordinal")
        try:
            x_encoded, encoding = encode_features(df[x_columns], method=method)
        except ValueError as e:
            raise WorkflowExecutionError(str(e)) from e
        encoded_cols = {c: m["kind"] for c, m in encoding.items() if m["kind"] != "numeric"}
        if encoded_cols:
            logger.info(f"Encoded categorical features ({method}): {encoded_cols}")

        return {
            "type": "selected_data",
            "x": x_encoded,
            "y": df[y_column],
            "x_columns": list(x_encoded.columns),
            "y_column": y_column,
            "encoding": encoding,
            # Already in [0, 1] — must bypass StandardScaler at the split step.
            "passthrough_columns": encoded_passthrough_columns(encoding),
        }

    def _execute_train_test_split(self, config: Dict, inputs: Dict) -> Dict:
        """Split data into train and test sets"""
        if not inputs:
            raise WorkflowExecutionError("No input data for train/test split")

        data = list(inputs.values())[0]
        x = data["x"]
        y = data["y"]

        # Derive the task spec (binary vs multiclass) from the ORIGINAL target
        # values and apply the encoding here, before DataPreparation touches y.
        # Doing it downstream (label-encoding node) is too late: the split
        # would already have re-encoded y, so a string comparison against the
        # original values maps every row to one class.
        from quoptuna.backend.task_type import TaskSpec

        label_mapping = config.get("label_mapping")
        task_spec = TaskSpec.from_target(
            y,
            label_mapping=label_mapping,
            favorable_class=config.get("favorable_class"),
        )
        # ALWAYS encode via the spec — binary with or without an explicit
        # mapping, and multiclass. Leaving unmapped binary targets to
        # DataPreparation's legacy encoder (classes[0] -> +1) inverts the
        # sign convention relative to the stored spec (class_labels[0] -> -1),
        # silently swapping class names in every downstream consumer.
        if task_spec.kind == "binary":
            derived = "" if label_mapping else " (derived; no explicit mapping)"
            logger.info(
                f"Encoding binary target at split: {task_spec.class_labels[0]} -> -1, "
                f"{task_spec.class_labels[1]} -> 1{derived}"
            )
        else:
            logger.info(
                f"Multiclass target ({task_spec.n_classes} classes): encoding "
                f"{list(task_spec.class_labels)} -> 0..{task_spec.n_classes - 1}"
            )
        y = pd.Series(task_spec.encode(y), name=data["y_column"])

        # Use DataPreparation class
        data_prep = DataPreparation(
            dataset={"x": x, "y": y},
            x_cols=list(x.columns),
            y_col=data["y_column"] if isinstance(data["y_column"], str) else data["y_column"][0],
            passthrough_columns=data.get("passthrough_columns"),
        )

        # Rebalance the TRAIN split only (test stays representative of the
        # real class distribution); applies uniformly to every model type,
        # quantum included, since it happens before Optimizer ever sees the
        # data rather than via a per-model class_weight constructor arg.
        #
        # ORDER IS LOAD-BEARING. The validation split is carved HERE, before
        # resampling, and only the inner training portion is resampled.
        # Oversampling duplicates minority rows verbatim; carving validation
        # out of an already-oversampled frame (which is what Optimizer used to
        # do) puts exact copies of training rows into validation, so the
        # objective scores memorisation instead of generalisation. On ILPD
        # that leaked 84% of the positive-class validation rows and inflated
        # the reported F1 from ~0.28 to 0.90.
        #
        # A sensitive column, if configured, is resolved from the raw file
        # FIRST (while x_train.index is still positional into it — both the
        # carve and the resampling invalidate that) and then carried through
        # both steps in lockstep, so the fairness audit stays row-aligned to
        # whichever split it is measured on.
        resampling = config.get("resampling", "none")
        sensitive_column = config.get("sensitive_feature")
        sensitive_full_train = None
        if sensitive_column:
            sensitive_full_train, _ = resolve_sensitive_series(
                config.get("dataset_id", ""), sensitive_column, data_prep.x_train, data_prep.x_test
            )

        x_fit, x_val, y_fit, y_val, sensitive_fit, sensitive_val = _carve_validation_split(
            data_prep.x_train, data_prep.y_train, sensitive_full_train
        )

        x_fit, y_fit, sensitive_fit = resample_train_split(
            x_fit,
            y_fit,
            strategy=resampling,
            sensitive_train=sensitive_fit,
        )

        return {
            "type": "split_data",
            # x_train/y_train are the frame models are FITTED on: inner train,
            # resampled. Analyze refits on exactly this, so its metrics are
            # comparable to the trial's.
            "x_train": x_fit,
            "x_test": data_prep.x_test,
            "y_train": y_fit,
            "y_test": data_prep.y_test,
            "x_val": x_val,
            "y_val": y_val,
            "sensitive_train": sensitive_fit,
            "sensitive_val": sensitive_val,
            "x_columns": data["x_columns"],
            "y_column": data["y_column"],
            "task_spec": task_spec.to_dict(),
        }

    def _execute_scaler(self, config: Dict, inputs: Dict) -> Dict:
        """Data is already scaled by DataPreparation, just pass through"""
        if not inputs:
            raise WorkflowExecutionError("No input data for scaler")

        # DataPreparation already handles scaling, just pass through
        return list(inputs.values())[0]

    def _execute_label_encoding(self, config: Dict, inputs: Dict) -> Dict:
        """Encode labels to -1 and 1 for binary classification (as required by quantum models)"""
        if not inputs:
            raise WorkflowExecutionError("No input data for label encoding")

        result = list(inputs.values())[0]

        # Get unique classes from training data
        import numpy as np

        y_train = result["y_train"]
        y_test = result["y_test"]

        def _already_encoded(series) -> bool:
            values = series.values.ravel() if hasattr(series, "values") else np.ravel(series)
            return set(np.unique(values).tolist()) <= {-1, 1}

        # Multiclass targets were already encoded to 0..K-1 at the split node;
        # validate and pass through (binary {-1,+1} re-encoding does not apply).
        task_spec = result.get("task_spec")
        if task_spec and task_spec.get("kind") == "multiclass":
            n_classes = int(task_spec["n_classes"])
            valid_codes = set(range(n_classes))

            def _codes(series):
                values = series.values.ravel() if hasattr(series, "values") else np.ravel(series)
                return set(np.unique(values).tolist())

            observed = _codes(y_train) | _codes(y_test)
            if not observed <= valid_codes:
                raise WorkflowExecutionError(
                    f"Multiclass labels {sorted(observed)} are not valid codes 0..{n_classes - 1}"
                )
            return result

        # Explicit user-provided mapping takes precedence (values map to -1/1) —
        # but it was already applied at the split node on the original values;
        # re-applying it to {-1, 1}-encoded labels would map every row to -1.
        label_mapping = config.get("label_mapping")
        if label_mapping and _already_encoded(y_train) and _already_encoded(y_test):
            return result
        if label_mapping:
            neg = str(label_mapping.get("neg"))
            pos = str(label_mapping.get("pos"))
            logger.info(f"Applying explicit label mapping: {neg} -> -1, {pos} -> 1")

            def _map(series):
                as_str = series.astype(str) if hasattr(series, "astype") else series
                mapped = np.where(np.asarray(as_str) == pos, 1, -1)
                if hasattr(series, "columns"):
                    return pd.DataFrame(mapped, columns=series.columns, index=series.index)
                return pd.DataFrame(mapped, columns=["target"])

            result["y_train"] = _map(y_train)
            result["y_test"] = _map(y_test)
            return result

        # Get unique classes
        unique_classes = np.unique(
            np.concatenate(
                [
                    y_train.values.ravel() if hasattr(y_train, "values") else y_train.ravel(),
                    y_test.values.ravel() if hasattr(y_test, "values") else y_test.ravel(),
                ]
            )
        )

        # For binary classification, map to -1 and 1
        if len(unique_classes) == 2:
            logger.info(
                f"Binary classification detected. Mapping classes {unique_classes} to [-1, 1]"
            )

            # Create mapping: first class -> -1, second class -> 1
            class_mapping = {unique_classes[0]: -1, unique_classes[1]: 1}

            # Apply mapping
            if hasattr(y_train, "replace"):
                # pandas DataFrame/Series
                result["y_train"] = y_train.replace(class_mapping)
                result["y_test"] = y_test.replace(class_mapping)
            else:
                # numpy array
                y_train_mapped = np.where(y_train == unique_classes[0], -1, 1)
                y_test_mapped = np.where(y_test == unique_classes[0], -1, 1)
                result["y_train"] = pd.DataFrame(
                    y_train_mapped,
                    columns=y_train.columns if hasattr(y_train, "columns") else ["target"],
                )
                result["y_test"] = pd.DataFrame(
                    y_test_mapped,
                    columns=y_test.columns if hasattr(y_test, "columns") else ["target"],
                )
        else:
            # K>2 without a task_spec means the split node never derived the
            # multiclass encoding — failing loudly beats silently feeding
            # unencoded labels to binary-asserting models.
            raise WorkflowExecutionError(
                f"Multi-class target ({len(unique_classes)} classes) reached label "
                "encoding without a task_spec; the train-test-split node must run first."
            )

        return result

    def _execute_model_config(self, config: Dict, inputs: Dict, node_type: str) -> Dict:
        """Configure model selection"""
        model_name = config.get("model_name")
        if not model_name:
            raise WorkflowExecutionError("No model name provided")

        result = {
            "type": "model_config",
            "model_name": model_name,
            "model_type": "quantum" if node_type == "quantum-model" else "classical",
        }

        # Merge input data if available
        if inputs:
            result.update(list(inputs.values())[0])

        return result

    def _execute_optuna_config(self, config: Dict, inputs: Dict) -> Dict:
        """Configure Optuna optimization parameters"""
        result = {
            "type": "optuna_config",
            "study_name": config.get("study_name", "workflow_study"),
            "n_trials": config.get("n_trials", 100),
            "db_name": config.get("db_name", "workflow_optimization.db"),
            "model_types": config.get("model_types"),
            "search_space": config.get("search_space"),
            "sampler": config.get("sampler", "tpe"),
            "sampler_seed": config.get("sampler_seed"),
            "pruner": config.get("pruner", "asha"),
            "pruner_min_resource": config.get("pruner_min_resource", 1),
            "pruner_reduction_factor": config.get("pruner_reduction_factor", 3),
            "intermediate_metric": config.get("intermediate_metric", "f1"),
            "max_steps": config.get("max_steps"),
            "convergence_interval": config.get("convergence_interval"),
            "max_vmap": config.get("max_vmap"),
            "dev_type": config.get("dev_type", "default.qubit"),
            "fairness_mode": config.get("fairness_mode", "off"),
            "fairness_metric": config.get("fairness_metric", "equal_opportunity_difference"),
            "fairness_threshold": config.get("fairness_threshold"),
            "sensitive_feature": config.get("sensitive_feature"),
            "dataset_id": config.get("dataset_id"),
        }

        # Merge input data if available
        if inputs:
            result.update(list(inputs.values())[0])

        return result

    def _execute_optimization(self, config: Dict, inputs: Dict) -> Dict:
        """Run Optuna optimization"""
        if not inputs:
            raise WorkflowExecutionError("No input configuration for optimization")

        opt_config = list(inputs.values())[0]

        # Store original DataFrames for later SHAP analysis. x_train here is
        # the inner (post-carve, resampled) train frame produced by the split
        # node — the exact data trials fit on, so Analyze reproduces them.
        x_train_df = opt_config["x_train"]
        x_test_df = opt_config["x_test"]
        y_train_df = opt_config["y_train"]
        y_test_df = opt_config["y_test"]
        x_val_df = opt_config.get("x_val")
        y_val_df = opt_config.get("y_val")

        def _as_array(frame):
            if frame is None:
                return None
            return frame.values if hasattr(frame, "values") else frame

        def _as_labels(frame):
            if frame is None:
                return None
            return frame.values.ravel() if hasattr(frame, "values") else np.asarray(frame).ravel()

        # Convert to numpy arrays for Optimizer (as shown in notebooks). The
        # validation split is passed in rather than carved by Optimizer: it
        # must be taken before resampling, which already happened upstream.
        data_dict = {
            "train_x": _as_array(x_train_df),
            "train_y": _as_labels(y_train_df),
            "test_x": _as_array(x_test_df),
            "test_y": _as_labels(y_test_df),
            "val_x": _as_array(x_val_df),
            "val_y": _as_labels(y_val_df),
        }

        task_spec = opt_config.get("task_spec")

        fairness_mode = opt_config.get("fairness_mode", "off")
        if (
            fairness_mode != "off"
            and task_spec
            and task_spec.get("kind") == "multiclass"
            and task_spec.get("favorable_code") is None
        ):
            raise WorkflowExecutionError(
                "Fairness-aware search on a multiclass target requires favorable_class"
            )
        # The search's disparity is measured on validation, so it needs the
        # sensitive column for those rows — carved in lockstep at split time.
        sensitive_val = None
        if fairness_mode != "off":
            column = opt_config.get("sensitive_feature")
            if not column:
                raise WorkflowExecutionError("Fairness-aware search requires a sensitive_feature")
            sens_val = opt_config.get("sensitive_val")
            if sens_val is None:
                raise WorkflowExecutionError(
                    "Fairness-aware search requires a validation split with the sensitive "
                    "column aligned to it; the dataset is too small to hold one out."
                )
            sensitive_val = np.asarray(sens_val).ravel()

        # Create optimizer (optional reduced search space, e.g. for tests)
        optimizer = Optimizer(
            db_name=opt_config.get("db_name", "workflow_optimization.db"),
            data=data_dict,
            study_name=opt_config.get("study_name", "workflow_study"),
            model_types=opt_config.get("model_types"),
            search_space=opt_config.get("search_space"),
            sampler=opt_config.get("sampler", "tpe"),
            sampler_seed=opt_config.get("sampler_seed"),
            pruner=opt_config.get("pruner", "asha"),
            pruner_min_resource=opt_config.get("pruner_min_resource", 1),
            pruner_reduction_factor=opt_config.get("pruner_reduction_factor", 3),
            intermediate_metric=opt_config.get("intermediate_metric", "f1"),
            max_steps=opt_config.get("max_steps"),
            convergence_interval=opt_config.get("convergence_interval"),
            max_vmap=opt_config.get("max_vmap"),
            dev_type=opt_config.get("dev_type", "default.qubit"),
            fairness_mode=fairness_mode,
            fairness_metric=opt_config.get("fairness_metric", "equal_opportunity_difference"),
            fairness_threshold=opt_config.get("fairness_threshold"),
            sensitive_val=sensitive_val,
            task_spec=task_spec,
        )

        # Run optimization
        n_trials = opt_config.get("n_trials", 100)
        model_name = opt_config.get("model_name", "DataReuploading")

        # Note: model_name is stored for reference, but Optuna will try different models automatically
        optimizer.optimize(n_trials=n_trials)

        # Get best trial
        if optimizer.study is None:
            raise RuntimeError("Optimization did not produce a study with a best trial")

        from optuna.trial import TrialState

        completed = [t for t in optimizer.study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            failed = [t for t in optimizer.study.trials if t.state == TrialState.FAIL]
            reason = failed[-1].user_attrs.get("error") if failed else "unknown error"
            raise WorkflowExecutionError(
                f"All {len(optimizer.study.trials)} trials failed. Last error: {reason}"
            )
        best_trial = study_best_trial(optimizer.study)

        pareto_trials = None
        if len(optimizer.study.directions) > 1:
            pareto_trials = [
                {
                    "trial": t.number,
                    "values": list(t.values),
                    "params": t.params,
                }
                for t in optimizer.study.best_trials
            ]

        return {
            "type": "optimization_result",
            "best_value": best_trial.values[0],
            "best_params": best_trial.params,
            "best_trial_number": best_trial.number,
            "pareto_trials": pareto_trials,
            "study_name": opt_config.get("study_name"),
            "db_name": opt_config.get("db_name"),
            "n_trials": n_trials,
            "model_name": model_name,
            # Store DataFrames for SHAP analysis. x_train/y_train are the
            # fitted frame (inner train, resampled) so build_xai reproduces
            # the trial's model; x_val/y_val let Analyze re-check the
            # objective it was selected on.
            "x_train": x_train_df,
            "x_test": x_test_df,
            "y_train": y_train_df,
            "y_test": y_test_df,
            "x_val": x_val_df,
            "y_val": y_val_df,
            "sensitive_train": opt_config.get("sensitive_train"),
            "sensitive_val": opt_config.get("sensitive_val"),
            "x_columns": opt_config.get("x_columns"),
            "y_column": opt_config.get("y_column"),
            "task_spec": task_spec,
            # Retraining a trial for analysis/deployment needs the same
            # training budget the search used; these are not in trial.params.
            "max_steps": opt_config.get("max_steps"),
            "convergence_interval": opt_config.get("convergence_interval"),
            "dev_type": opt_config.get("dev_type", "default.qubit"),
        }

    def _execute_shap_analysis(self, config: Dict, inputs: Dict) -> Dict:
        """Generate SHAP analysis"""
        # Check for inputs from dependencies first, then check for manually injected "input"
        if not inputs:
            if "input" in self.results:
                opt_result = self.results["input"]
            else:
                raise WorkflowExecutionError("No input data for SHAP analysis")
        else:
            opt_result = list(inputs.values())[0]

        # Load the best model from Optuna study
        import numpy as np
        from optuna import load_study

        from quoptuna.backend.models import create_model
        from quoptuna.server.services.storage import optuna_storage_url

        db_name = opt_result.get("db_name")
        study_name = opt_result.get("study_name")

        storage_location = optuna_storage_url(db_name)
        study = load_study(storage=storage_location, study_name=study_name)
        best_trial = study_best_trial(study)

        # Get DataFrames from opt_result
        x_train_df = opt_result["x_train"]
        x_test_df = opt_result["x_test"]
        y_train_df = opt_result["y_train"]
        y_test_df = opt_result["y_test"]

        # Recreate and fit the best model (model.fit needs numpy arrays)
        # Extract model_type and pass remaining params to avoid duplicate argument error
        task_spec = opt_result.get("task_spec")
        n_classes = int(task_spec["n_classes"]) if task_spec else 2
        params = {k: v for k, v in best_trial.params.items() if k != "model_type"}
        model = create_model(
            best_trial.params["model_type"],
            n_classes=n_classes,
            **_training_budget(opt_result),
            **params,
        )

        # Convert to numpy for model fitting. y MUST be 1-D — see build_xai.
        x_train_np = x_train_df.values if hasattr(x_train_df, "values") else x_train_df
        y_train_np = np.asarray(y_train_df).ravel()

        model.fit(x_train_np, y_train_np)

        # Prepare data dictionary for XAI (XAI expects DataFrames)
        data_dict = {
            "x_train": x_train_df,
            "x_test": x_test_df,
            "y_train": y_train_df,
            "y_test": y_test_df,
        }

        # Create XAI instance with XAIConfig
        xai_config = XAIConfig(use_proba=True, onsubset=True, subset_size=50)
        xai = XAI(model=model, data=data_dict, config=xai_config)

        # Generate SHAP plots
        plot_types = config.get("plot_types", ["bar", "beeswarm", "waterfall"])
        plots = {}

        for plot_type in plot_types:
            try:
                if plot_type == "bar":
                    plots[plot_type] = xai.get_bar_plot()
                elif plot_type == "beeswarm":
                    plots[plot_type] = xai.get_beeswarm_plot()
                elif plot_type == "waterfall":
                    plots[plot_type] = xai.get_waterfall_plot(index=0)
                elif plot_type == "violin":
                    plots[plot_type] = xai.get_violin_plot()
                elif plot_type == "heatmap":
                    plots[plot_type] = xai.get_heatmap_plot()
            except Exception as e:
                logger.error(f"Error generating {plot_type} plot: {e}")
                logger.exception("Full traceback:")

        # Get feature importance from SHAP values
        shap_values = xai.shap_values
        feature_importance = []

        if hasattr(shap_values, "values") and hasattr(shap_values, "data"):
            # Calculate mean absolute SHAP values for feature importance
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            # Get feature names from DataFrame columns
            feature_names = (
                list(x_train_df.columns)
                if hasattr(x_train_df, "columns")
                else [f"feature_{i}" for i in range(x_train_df.shape[1])]
            )

            for i, feature in enumerate(feature_names):
                feature_importance.append(
                    {
                        "feature": feature,
                        "importance": float(mean_abs_shap[i])
                        if len(mean_abs_shap.shape) == 1
                        else float(mean_abs_shap[i].mean()),
                    }
                )

            # Sort by importance
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "type": "shap_analysis",
            "plots": plots,
            "feature_importance": feature_importance,
            "feature_names": feature_names if feature_importance else [],
        }

    def _execute_confusion_matrix(self, config: Dict, inputs: Dict) -> Dict:
        """Generate confusion matrix"""
        if not inputs:
            raise WorkflowExecutionError("No input data for confusion matrix")

        result = list(inputs.values())[0]
        return {
            "type": "confusion_matrix",
            "message": "Confusion matrix generation not yet implemented",
            **result,
        }

    def _execute_feature_importance(self, config: Dict, inputs: Dict) -> Dict:
        """Calculate feature importance"""
        if not inputs:
            raise WorkflowExecutionError("No input data for feature importance")

        result = list(inputs.values())[0]

        if "plots" in result and "bar" in result["plots"]:
            # SHAP bar plot already shows feature importance
            return {
                "type": "feature_importance",
                "source": "shap",
                **result,
            }

        return {
            "type": "feature_importance",
            "message": "Feature importance analysis",
            **result,
        }

    def _execute_export_model(self, config: Dict, inputs: Dict) -> Dict:
        """Export trained model"""
        if not inputs:
            raise WorkflowExecutionError("No input data for model export")

        result = list(inputs.values())[0]
        export_path = config.get("export_path", f"./models/{result.get('study_name', 'model')}.pkl")

        return {
            "type": "model_export",
            "export_path": export_path,
            "message": f"Model would be exported to {export_path}",
            **result,
        }

    def _execute_generate_report(self, config: Dict, inputs: Dict) -> Dict:
        """Generate AI-powered report"""
        if not inputs:
            raise WorkflowExecutionError("No input data for report generation")

        result = list(inputs.values())[0]
        llm_provider = config.get("llm_provider", "openai")

        return {
            "type": "report",
            "llm_provider": llm_provider,
            "message": "AI report generation requires LLM API keys to be configured",
            **result,
        }

    def execute(self) -> Dict[str, Any]:
        """Execute the entire workflow"""
        logger.info(f"Starting workflow execution: {self.workflow.get('name', 'Unnamed')}")

        try:
            # Get execution order
            execution_order = self.topological_sort()

            # Execute nodes in order
            for node_id in execution_order:
                result = self.execute_node(node_id)
                self.results[node_id] = result
                logger.info(f"Node {node_id} executed successfully")

            # Return final results
            final_result = {
                "status": "completed",
                "workflow_id": self.workflow.get("id"),
                "workflow_name": self.workflow.get("name"),
                "execution_order": execution_order,
                "node_results": self.results,
            }

            logger.info("Workflow execution completed successfully")
            return final_result

        except Exception as e:
            logger.exception(f"Workflow execution failed: {e!s}")
            raise WorkflowExecutionError(f"Execution failed: {e!s}") from e
