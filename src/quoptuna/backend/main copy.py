import logging
import time

from data import load_data, preprocess_data
from models import create_model
from optuna import Trial, create_study
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score

logging.getLogger().setLevel(logging.INFO)


def objective(trial: Trial):
    try:
        # Define the hyperparameter search space
        params = {
            "max_vmap": trial.suggest_categorical("max_vmap", [1]),
            "batch_size": trial.suggest_categorical("batch_size", [32]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.001, 0.01, 0.1]
            ),
            "n_input_copies": trial.suggest_categorical("n_input_copies", [1, 2, 3]),
            "n_layers": trial.suggest_categorical("n_layers", [1, 5, 10]),
            "observable_type": trial.suggest_categorical(
                "observable_type", ["single", "half", "full"]
            ),
            "repeats": trial.suggest_categorical("repeats", [1, 5, 10]),
            "C": trial.suggest_categorical("C", [0.1, 1, 10, 100]),
            "gamma_factor": trial.suggest_categorical("gamma_factor", [0.1, 1, 10]),
            "trotter_steps": trial.suggest_categorical("trotter_steps", [1, 3, 5]),
            "t": trial.suggest_categorical("t", [0.01, 0.1, 1.0]),
            "n_qfeatures": trial.suggest_categorical("n_qfeatures", ["full", "half"]),
            "n_episodes": trial.suggest_categorical("n_episodes", [10, 100, 500, 2000]),
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

        X, y = load_data("notebook/GaN Crystallinity.csv")
        train_X, test_X, train_y, test_y = preprocess_data(X, y)

        model.fit(train_X, train_y)
        score = model.score(test_X, test_y)

        f_score_ = f1_score(test_y, model.predict(test_X))
        acc_ = accuracy_score(test_y, model.predict(test_X))

        log_user_attributes(
            model_type, {"accuracy": acc_, "f1_score": f_score_, "score": score}, trial
        )

        return score, f_score_, acc_
    except Exception as e:
        print(e)
        return 0, 0, 0


def log_user_attributes(model_type, eval_scores, trial):
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


if __name__ == "__main__":
    import threading
    import time
    from wsgiref.simple_server import make_server

    import optuna
    from optuna_dashboard import wsgi

    port = 8020
    dataset = "Crystalline"
    DB = f"sqlite:///{dataset}.db"
    storage = optuna.storages.RDBStorage(DB)
    app = wsgi(storage)
    httpd = make_server("localhost", port, app)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    time.sleep(3)  # Wait until the server startup
    study = create_study(
        storage=DB,
        study_name=dataset,
        sampler=TPESampler(),
        directions=["maximize", "maximize", "maximize"],
    )
    study.optimize(objective, n_trials=100)
    httpd.shutdown()
