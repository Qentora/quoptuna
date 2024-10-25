import logging
import os
from typing import Optional

from optuna import Trial, create_study, load_study
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score

from quoptuna.backend.data import load_data, preprocess_data
from quoptuna.backend.models import create_model

logging.getLogger().setLevel(logging.INFO)


class Optimizer:
    def __init__(
        self,
        db_name: str,
        dataset_name: str = "",
        data: Optional[dict] = None,  # noqa: FA100
        study_name: str = "",
    ):
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
        self.data_path = f"db/{self.db_name}.db"
        if not os.path.exists("db"):  # noqa: PTH110
            os.makedirs("db")  # noqa: PTH103
        self.storage_location = f"sqlite:///{self.data_path}"
        self.study_name = study_name
        self.study = None

    def load_and_preprocess_data(self):
        self.X, self.y = load_data(self.data_path)
        self.train_x, self.test_x, self.train_y, self.test_y = preprocess_data(
            self.X, self.y
        )

    def objective(self, trial: Trial):
        try:
            # Define the hyperparameter search space
            params = {
                "max_vmap": trial.suggest_categorical("max_vmap", [1]),
                "batch_size": trial.suggest_categorical("batch_size", [32]),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", [0.001, 0.01, 0.1]
                ),
                "n_input_copies": trial.suggest_categorical(
                    "n_input_copies", [1, 2, 3]
                ),
                "n_layers": trial.suggest_categorical("n_layers", [1, 5, 10]),
                "observable_type": trial.suggest_categorical(
                    "observable_type", ["single", "half", "full"]
                ),
                "repeats": trial.suggest_categorical("repeats", [1, 5, 10]),
                "C": trial.suggest_categorical("C", [0.1, 1, 10, 100]),
                "gamma_factor": trial.suggest_categorical("gamma_factor", [0.1, 1, 10]),
                "trotter_steps": trial.suggest_categorical("trotter_steps", [1, 3, 5]),
                "t": trial.suggest_categorical("t", [0.01, 0.1, 1.0]),
                "n_qfeatures": trial.suggest_categorical(
                    "n_qfeatures", ["full", "half"]
                ),
                "n_episodes": trial.suggest_categorical(
                    "n_episodes", [10, 100, 500, 2000]
                ),
                "visible_qubits": trial.suggest_categorical(
                    "visible_qubits", ["single", "half", "full"]
                ),
                "temperature": trial.suggest_categorical("temperature", [1, 10, 100]),
                "encoding_layers": trial.suggest_categorical(
                    "encoding_layers", [1, 3, 5, 10]
                ),
                "degree": trial.suggest_categorical("degree", [2, 3, 4]),
                "n_qchannels": trial.suggest_categorical("n_qchannels", [1, 5, 10]),
                "qkernel_shape": trial.suggest_categorical("qkernel_shape", [2, 3]),
                "kernel_shape": trial.suggest_categorical("kernel_shape", [2, 3, 5]),
                "filter_name": trial.suggest_categorical(
                    "filter_name", ["edge_detect", "smooth", "sharpen"]
                ),
                "gamma": trial.suggest_categorical("gamma", [0.001, 0.01, 0.1, 1]),
                "alpha": trial.suggest_categorical("alpha", [0.01, 0.001, 0.0001]),
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes", ["[100,)", "(10, 10, 10, 10)", "(50, 10, 5)"]
                ),
                "eta0": trial.suggest_categorical("eta0", [0.1, 1, 10]),
            }

            model_type = trial.suggest_categorical(
                "model_type",
                [
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
                ],
            )

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
            import logging

            logging.exception("An error occurred")  # Use logging instead of print
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

        study = create_study(
            storage=self.storage_location,
            sampler=TPESampler(),
            directions=["maximize"],
            study_name=self.study_name,
        )
        self.study = study
        study.optimize(self.objective, n_trials=n_trials)
        return study, study.best_trials
