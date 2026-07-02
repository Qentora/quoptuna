import logging
import time
from typing import Optional

from optuna import Trial, create_study, load_study
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score

from quoptuna.backend.models import create_model
from quoptuna.backend.utils.data_utils.data import load_data, preprocess_data
from quoptuna.backend.utils.storage import ensure_db_dir, optuna_db_path

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

    def load_and_preprocess_data(self):
        self.X, self.y = load_data(self.data_path)
        self.train_x, self.test_x, self.train_y, self.test_y = preprocess_data(self.X, self.y)

    def objective(self, trial: Trial):
        try:
            # Sample one value per hyperparameter from the (possibly reduced)
            # search space. Insertion order is preserved for reproducibility.
            params = {
                name: trial.suggest_categorical(name, choices)
                for name, choices in self.search_space.items()
            }

            model_type = trial.suggest_categorical("model_type", self.model_types)

            model = create_model(model_type, **params)

            model.fit(self.train_x, self.train_y)
            score = model.score(self.test_x, self.test_y)

            f_score_ = f1_score(self.test_y, model.predict(self.test_x))
            acc_ = accuracy_score(self.test_y, model.predict(self.test_x))

            self.log_user_attributes(
                model_type,
                {"accuracy": acc_, "f1_score": f_score_, "score": score},
                trial,
            )

            return f_score_  # noqa: TRY300
        except Exception:
            logger.exception("An error occurred")
            return 0

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
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return create_study(
                    storage=self.storage_location,
                    sampler=TPESampler(),
                    directions=["maximize"],
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
        # Final fallback: the study/tables exist, so just load it.
        try:
            return load_study(storage=self.storage_location, study_name=self.study_name)
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
        study.optimize(self.objective, n_trials=n_trials)
        return study, study.best_trials
