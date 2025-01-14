from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Task(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str
    status: str
    progress: float
    result: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataReference(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    data_id: str
    file_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# import sklearn models
import ast

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from quoptuna.backend.base.pennylane_models import (
    CircuitCentricClassifier,
    ConvolutionalNeuralNetwork,
    DataReuploadingClassifier,
    DataReuploadingClassifierSeparable,
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierSeparable,
    IQPKernelClassifier,
    # ParallelGradients,
    ProjectedQuantumKernel,
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable,
    QuantumKitchenSinks,
    QuantumMetricLearner,
    QuanvolutionalNeuralNetwork,
    SeparableKernelClassifier,
    SeparableVariationalClassifier,
    TreeTensorClassifier,
    WeiNet,
)


# Define a custom exception class
class UnknownModelTypeError(ValueError):
    def __init__(self, model_type):
        super().__init__(f"Unknown model type: {model_type}")


def create_model(model_type, **kwargs):
    model_constructors = {
        "CircuitCentricClassifier": (
            CircuitCentricClassifier,
            ["max_vmap", "batch_size", "learning_rate", "n_input_copies", "n_layers"],
        ),
        "DataReuploadingClassifier": (
            DataReuploadingClassifier,
            ["max_vmap", "batch_size", "learning_rate", "n_layers", "observable_type"],
        ),
        "DataReuploadingClassifierSeparable": (
            DataReuploadingClassifierSeparable,
            ["max_vmap", "batch_size", "learning_rate", "n_layers", "observable_type"],
        ),
        "DressedQuantumCircuitClassifier": (
            DressedQuantumCircuitClassifier,
            ["max_vmap", "batch_size", "learning_rate", "n_layers"],
        ),
        "DressedQuantumCircuitClassifierSeparable": (
            DressedQuantumCircuitClassifierSeparable,
            ["max_vmap", "batch_size", "learning_rate", "n_layers"],
        ),
        "IQPKernelClassifier": (IQPKernelClassifier, ["max_vmap", "repeats", "C"]),
        "ProjectedQuantumKernel": (
            ProjectedQuantumKernel,
            ["max_vmap", "gamma_factor", "C", "trotter_steps", "t"],
        ),
        "QuantumKitchenSinks": (
            QuantumKitchenSinks,
            ["max_vmap", "n_qfeatures", "n_episodes"],
        ),
        "QuantumMetricLearner": (
            QuantumMetricLearner,
            ["max_vmap", "batch_size", "learning_rate", "n_layers"],
        ),
        "QuantumBoltzmannMachine": (
            QuantumBoltzmannMachine,
            [
                "max_vmap",
                "batch_size",
                "learning_rate",
                "visible_qubits",
                "temperature",
            ],
        ),
        "QuantumBoltzmannMachineSeparable": (
            QuantumBoltzmannMachineSeparable,
            [
                "max_vmap",
                "batch_size",
                "learning_rate",
                "visible_qubits",
                "temperature",
            ],
        ),
        "TreeTensorClassifier": (
            TreeTensorClassifier,
            ["max_vmap", "batch_size", "learning_rate"],
        ),
        "QuanvolutionalNeuralNetwork": (
            QuanvolutionalNeuralNetwork,
            [
                "max_vmap",
                "batch_size",
                "learning_rate",
                "n_qchannels",
                "qkernel_shape",
                "kernel_shape",
            ],
        ),
        "WeiNet": (WeiNet, ["max_vmap", "batch_size", "learning_rate", "filter_name"]),
        "SeparableVariationalClassifier": (
            SeparableVariationalClassifier,
            ["batch_size", "learning_rate", "encoding_layers"],
        ),
        "SeparableKernelClassifier": (
            SeparableKernelClassifier,
            ["C", "encoding_layers"],
        ),
        "ConvolutionalNeuralNetwork": (
            ConvolutionalNeuralNetwork,
            ["batch_size", "learning_rate", "kernel_shape"],
        ),
        "SVC": (SVC, ["gamma", "C"]),
        "SVClinear": (LinearSVC, ["C"]),
        "MLPClassifier": (
            MLPClassifier,
            ["batch_size", "learning_rate", "hidden_layer_sizes", "alpha"],
        ),
        "Perceptron": (Perceptron, ["eta0"]),
    }

    if model_type not in model_constructors:
        raise UnknownModelTypeError(model_type)

    model_class, param_keys = model_constructors[model_type]
    params = {key: kwargs.get(key) for key in param_keys}

    if model_type == "MLPClassifier":
        params["hidden_layer_sizes"] = ast.literal_eval(params["hidden_layer_sizes"])

    return model_class(**params)
