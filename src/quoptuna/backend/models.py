# import sklearn models
import ast
import inspect

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
from quoptuna.backend.base.pennylane_models.ovr import wrap_one_vs_rest


# Define a custom exception class
class UnknownModelTypeError(ValueError):
    def __init__(self, model_type):
        super().__init__(f"Unknown model type: {model_type}")


# Quantum models whose readout is binary by construction (sign of an
# expectation value, {-1,+1} labels). For K>2 targets these are wrapped in a
# OneVsRestClassifier; everything else (classical sklearn models and the
# kernel-head quantum models whose final classifier is an SVC/LogisticRegression)
# handles multiclass natively.
VARIATIONAL_BINARY_MODELS = {
    "CircuitCentricClassifier",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierSeparable",
    "QuantumMetricLearner",
    "QuantumBoltzmannMachine",
    "QuantumBoltzmannMachineSeparable",
    "TreeTensorClassifier",
    "QuanvolutionalNeuralNetwork",
    "WeiNet",
    "SeparableVariationalClassifier",
    "ConvolutionalNeuralNetwork",
}

BINARY_N_CLASSES = 2


def create_model(model_type, n_classes: int = BINARY_N_CLASSES, **kwargs):
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
            ["batch_size", "hidden_layer_sizes", "alpha"],
        ),
        "Perceptron": (Perceptron, ["eta0"]),
    }

    if model_type not in model_constructors:
        raise UnknownModelTypeError(model_type)

    model_class, param_keys = model_constructors[model_type]
    params = {key: kwargs.get(key) for key in param_keys}

    if model_type == "MLPClassifier":
        hidden_layer_sizes = params["hidden_layer_sizes"]
        if hidden_layer_sizes is None:
            msg = "hidden_layer_sizes is required for MLPClassifier"
            raise ValueError(msg)
        params["hidden_layer_sizes"] = (
            ast.literal_eval(hidden_layer_sizes)
            if isinstance(hidden_layer_sizes, str)
            else hidden_layer_sizes
        )
        params["learning_rate_init"] = kwargs.get("learning_rate")

    # Simulator device selection for quantum models. Detected by signature
    # rather than a whitelist so classical sklearn models are auto-excluded.
    dev_type = kwargs.get("dev_type")
    if dev_type is not None and "dev_type" in inspect.signature(model_class.__init__).parameters:
        params["dev_type"] = dev_type

    model = model_class(**params)

    # Training-budget knobs for the iterative (JAX-trained) models. Applied
    # post-construction (and before any OvR wrapping) because not every
    # constructor accepts them.
    for knob in ("max_steps", "convergence_interval"):
        value = kwargs.get(knob)
        if value is not None and hasattr(model, knob):
            setattr(model, knob, value)

    if n_classes > BINARY_N_CLASSES and model_type in VARIATIONAL_BINARY_MODELS:
        model = wrap_one_vs_rest(model)

    return model
