import pytest
import shap
from sklearn.datasets import load_iris

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
from quoptuna.backend.models import MLPClassifier
from quoptuna.backend.xai import XAI


@pytest.fixture
def iris_data():
    # Load the Iris dataset
    data = load_iris()
    return {"x_train": data.data, "y_train": data.target}


@pytest.fixture
def trained_model(iris_data):
    # Train a simple model on the Iris dataset
    model = MLPClassifier()
    model.fit(iris_data["x_train"], iris_data["y_train"])
    return model


def test_xai_initialization(trained_model, iris_data):
    # Test the initialization of the XAI class
    xai = XAI(model=trained_model, data=iris_data)
    assert xai.model == trained_model
    assert xai.data == iris_data
    assert xai.explainer_type == "shap-auto"


def test_get_explainer(trained_model, iris_data):
    # Test the get_explainer method
    xai = XAI(model=trained_model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)


def test_validate_predict_proba(trained_model):
    # Test the validate_predict_proba method
    xai = XAI(model=trained_model, data={})
    assert xai.validate_predict_proba() is True


def test_invalid_model():
    # Test initialization with an invalid model
    msg = "Model does not have a predict_proba method"
    with pytest.raises(ValueError, match=msg):
       XAI(model=None, data={})  # Instantiate the XAI object


# test the shap-auto explainer with all models in the backend.models module
def test_xai_circuit_centric_classifier(iris_data):
    model = CircuitCentricClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)


def test_xai_convolutional_neural_network(iris_data):
    model = ConvolutionalNeuralNetwork()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)


def test_xai_data_reuploading_classifier(iris_data):
    model = DataReuploadingClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_data_reuploading_classifier_separable(iris_data):
    model = DataReuploadingClassifierSeparable()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_dressed_quantum_circuit_classifier(iris_data):
    model = DressedQuantumCircuitClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_dressed_quantum_circuit_classifier_separable(iris_data):
    model = DressedQuantumCircuitClassifierSeparable()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_iqp_kernel_classifier(iris_data):
    model = IQPKernelClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_projected_quantum_kernel(iris_data):
    model = ProjectedQuantumKernel()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_quantum_boltzmann_machine(iris_data):
    model = QuantumBoltzmannMachine()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_quantum_boltzmann_machine_separable(iris_data):
    model = QuantumBoltzmannMachineSeparable()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_quantum_kitchen_sinks(iris_data):
    model = QuantumKitchenSinks()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_quantum_metric_learner(iris_data):
    model = QuantumMetricLearner()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_quanvolutional_neural_network(iris_data):
    model = QuanvolutionalNeuralNetwork()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_separable_kernel_classifier(iris_data):
    model = SeparableKernelClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_separable_variational_classifier(iris_data):
    model = SeparableVariationalClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_tree_tensor_classifier(iris_data):
    model = TreeTensorClassifier()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)

def test_xai_weinet(iris_data):
    model = WeiNet()
    xai = XAI(model=model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)
