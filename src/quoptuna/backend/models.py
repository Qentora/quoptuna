# import sklearn models
from sklearn.linear_model import Perceptron

# import MLPClassifier
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


def create_model(model_type, **kwargs):
    # Define the model with the hyperparameters
    if model_type == "CircuitCentricClassifier":
        return CircuitCentricClassifier(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_input_copies=kwargs.get("n_input_copies"),
            n_layers=kwargs.get("n_layers"),
        )
    elif model_type == "DataReuploadingClassifier":
        return DataReuploadingClassifier(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_layers=kwargs.get("n_layers"),
            observable_type=kwargs.get("observable_type"),
        )
    elif model_type == "DataReuploadingClassifierSeparable":
        return DataReuploadingClassifierSeparable(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_layers=kwargs.get("n_layers"),
            observable_type=kwargs.get("observable_type"),
        )
    elif model_type == "DressedQuantumCircuitClassifier":
        return DressedQuantumCircuitClassifier(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_layers=kwargs.get("n_layers"),
        )
    elif model_type == "DressedQuantumCircuitClassifierSeparable":
        return DressedQuantumCircuitClassifierSeparable(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_layers=kwargs.get("n_layers"),
        )
    elif model_type == "IQPKernelClassifier":
        return IQPKernelClassifier(
            max_vmap=kwargs.get("max_vmap"),
            repeats=kwargs.get("repeats"),
            C=kwargs.get("C"),
        )
    elif model_type == "ProjectedQuantumKernel":
        return ProjectedQuantumKernel(
            max_vmap=kwargs.get("max_vmap"),
            gamma_factor=kwargs.get("gamma_factor"),
            C=kwargs.get("C"),
            trotter_steps=kwargs.get("trotter_steps"),
            t=kwargs.get("t"),
        )
    elif model_type == "QuantumKitchenSinks":
        return QuantumKitchenSinks(
            max_vmap=kwargs.get("max_vmap"),
            n_qfeatures=kwargs.get("n_qfeatures"),
            n_episodes=kwargs.get("n_episodes"),
        )
    elif model_type == "QuantumMetricLearner":
        return QuantumMetricLearner(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_layers=kwargs.get("n_layers"),
        )
    elif model_type == "QuantumBoltzmannMachine":
        return QuantumBoltzmannMachine(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            visible_qubits=kwargs.get("visible_qubits"),
            temperature=kwargs.get("temperature"),
        )
    elif model_type == "QuantumBoltzmannMachineSeparable":
        return QuantumBoltzmannMachineSeparable(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            visible_qubits=kwargs.get("visible_qubits"),
            temperature=kwargs.get("temperature"),
        )
    elif model_type == "TreeTensorClassifier":
        return TreeTensorClassifier(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
        )
    elif model_type == "QuanvolutionalNeuralNetwork":
        return QuanvolutionalNeuralNetwork(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            n_qchannels=kwargs.get("n_qchannels"),
            qkernel_shape=kwargs.get("qkernel_shape"),
            kernel_shape=kwargs.get("kernel_shape"),
        )
    elif model_type == "WeiNet":
        return WeiNet(
            max_vmap=kwargs.get("max_vmap"),
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            filter_name=kwargs.get("filter_name"),
        )
    elif model_type == "SeparableVariationalClassifier":
        return SeparableVariationalClassifier(
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            encoding_layers=kwargs.get("encoding_layers"),
        )
    elif model_type == "SeparableKernelClassifier":
        return SeparableKernelClassifier(
            C=kwargs.get("C"),
            encoding_layers=kwargs.get("encoding_layers"),
        )
    elif model_type == "ConvolutionalNeuralNetwork":
        return ConvolutionalNeuralNetwork(
            batch_size=kwargs.get("batch_size"),
            learning_rate=kwargs.get("learning_rate"),
            kernel_shape=kwargs.get("kernel_shape"),
        )
    elif model_type == "SVC":
        return SVC(
            gamma=kwargs.get("gamma"),
            C=kwargs.get("C"),
        )
    elif model_type == "SVClinear":
        return LinearSVC(C=kwargs.get("C"))
    elif model_type == "MLPClassifier":
        return MLPClassifier(
            batch_size=kwargs.get("batch_size"),
            learning_rate_init=kwargs.get("learning_rate"),
            hidden_layer_sizes=eval(kwargs.get("hidden_layer_sizes")),
            alpha=kwargs.get("alpha"),
        )
    elif model_type == "Perceptron":
        return Perceptron(eta0=kwargs.get("eta0"))
    msg = f"Unknown model type: {model_type}"
    raise ValueError(msg)
