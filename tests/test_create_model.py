import pytest
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from quoptuna.backend.models import (
    CircuitCentricClassifier,
    ConvolutionalNeuralNetwork,
    DataReuploadingClassifier,
    DataReuploadingClassifierSeparable,
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierSeparable,
    IQPKernelClassifier,
    ProjectedQuantumKernel,
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable,
    QuantumKitchenSinks,
    QuantumMetricLearner,
    QuanvolutionalNeuralNetwork,
    SeparableKernelClassifier,
    SeparableVariationalClassifier,
    TreeTensorClassifier,
    UnknownModelTypeError,
    WeiNet,
    create_model,
)


def test_create_circuit_centric_classifier():
    model = create_model(
        "CircuitCentricClassifier",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_input_copies=2,
        n_layers=3,
    )
    assert isinstance(model, CircuitCentricClassifier)


def test_create_data_reuploading_classifier():
    model = create_model(
        "DataReuploadingClassifier",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_layers=3,
        observable_type="Z",
    )
    assert isinstance(model, DataReuploadingClassifier)


def test_create_data_reuploading_classifier_separable():
    model = create_model(
        "DataReuploadingClassifierSeparable",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_layers=3,
        observable_type="Z",
    )
    assert isinstance(model, DataReuploadingClassifierSeparable)


def test_create_dressed_quantum_circuit_classifier():
    model = create_model(
        "DressedQuantumCircuitClassifier",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_layers=3,
    )
    assert isinstance(model, DressedQuantumCircuitClassifier)


def test_create_dressed_quantum_circuit_classifier_separable():
    model = create_model(
        "DressedQuantumCircuitClassifierSeparable",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_layers=3,
    )
    assert isinstance(model, DressedQuantumCircuitClassifierSeparable)


def test_create_iqp_kernel_classifier():
    model = create_model("IQPKernelClassifier", max_vmap=1, repeats=5, C=1.0)
    assert isinstance(model, IQPKernelClassifier)


def test_create_projected_quantum_kernel():
    model = create_model(
        "ProjectedQuantumKernel",
        max_vmap=1,
        gamma_factor=0.5,
        C=1.0,
        trotter_steps=10,
        t=0.1,
    )
    assert isinstance(model, ProjectedQuantumKernel)


def test_create_quantum_kitchen_sinks():
    model = create_model("QuantumKitchenSinks", max_vmap=1, n_qfeatures=5, n_episodes=10)
    assert isinstance(model, QuantumKitchenSinks)


def test_create_quantum_metric_learner():
    model = create_model(
        "QuantumMetricLearner",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_layers=3,
    )
    assert isinstance(model, QuantumMetricLearner)


def test_create_quantum_boltzmann_machine():
    model = create_model(
        "QuantumBoltzmannMachine",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        visible_qubits=4,
        temperature=0.1,
    )
    assert isinstance(model, QuantumBoltzmannMachine)


def test_create_quantum_boltzmann_machine_separable():
    model = create_model(
        "QuantumBoltzmannMachineSeparable",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        visible_qubits=4,
        temperature=0.1,
    )
    assert isinstance(model, QuantumBoltzmannMachineSeparable)


def test_create_tree_tensor_classifier():
    model = create_model("TreeTensorClassifier", max_vmap=1, batch_size=32, learning_rate=0.01)
    assert isinstance(model, TreeTensorClassifier)


def test_create_quanvolutional_neural_network():
    model = create_model(
        "QuanvolutionalNeuralNetwork",
        max_vmap=1,
        batch_size=32,
        learning_rate=0.01,
        n_qchannels=2,
        qkernel_shape=(3, 3),
        kernel_shape=(3, 3),
    )
    assert isinstance(model, QuanvolutionalNeuralNetwork)


def test_create_weinet():
    model = create_model(
        "WeiNet", max_vmap=1, batch_size=32, learning_rate=0.01, filter_name="default"
    )
    assert isinstance(model, WeiNet)


def test_create_separable_variational_classifier():
    model = create_model(
        "SeparableVariationalClassifier",
        batch_size=32,
        learning_rate=0.01,
        encoding_layers=3,
    )
    assert isinstance(model, SeparableVariationalClassifier)


def test_create_separable_kernel_classifier():
    model = create_model("SeparableKernelClassifier", C=1.0, encoding_layers=3)
    assert isinstance(model, SeparableKernelClassifier)


def test_create_convolutional_neural_network():
    model = create_model(
        "ConvolutionalNeuralNetwork",
        batch_size=32,
        learning_rate=0.01,
        kernel_shape=(3, 3),
    )
    assert isinstance(model, ConvolutionalNeuralNetwork)


def test_create_svc():
    model = create_model("SVC", gamma="scale", C=1.0)
    assert isinstance(model, SVC)


def test_create_linear_svc():
    model = create_model("SVClinear", C=1.0)
    assert isinstance(model, LinearSVC)


def test_create_mlp_classifier():
    model = create_model(
        "MLPClassifier", hidden_layer_sizes="(100,)", learning_rate=0.01, alpha=0.0001
    )
    assert isinstance(model, MLPClassifier)
    assert model.hidden_layer_sizes == (100,)


def test_create_perceptron():
    model = create_model("Perceptron", eta0=0.1)
    assert isinstance(model, Perceptron)
    assert model.eta0 == 0.1  # noqa: PLR2004


def test_unknown_model_type():
    with pytest.raises(UnknownModelTypeError):
        create_model("UnknownModel")
