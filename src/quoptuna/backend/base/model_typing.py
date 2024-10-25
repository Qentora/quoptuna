from typing import Type, Union

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

# Define a type alias for the model classes
ModelType = Type[Union[
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
    WeiNet,
    Perceptron,
    MLPClassifier,
    SVC,
    LinearSVC,
]]
