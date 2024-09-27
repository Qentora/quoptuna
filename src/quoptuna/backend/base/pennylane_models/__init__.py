from . import qml_benchmarks
from .qml_benchmarks.models.circuit_centric import CircuitCentricClassifier
from .qml_benchmarks.models.convolutional_neural_network import (
    ConvolutionalNeuralNetwork,
)
from .qml_benchmarks.models.data_reuploading import (
    DataReuploadingClassifier,
    DataReuploadingClassifierNoCost,
    DataReuploadingClassifierNoScaling,
    DataReuploadingClassifierNoTrainableEmbedding,
    DataReuploadingClassifierSeparable,
)
from .qml_benchmarks.models.dressed_quantum_circuit import (
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierOnlyNN,
    DressedQuantumCircuitClassifierSeparable,
)
from .qml_benchmarks.models.iqp_kernel import IQPKernelClassifier
from .qml_benchmarks.models.iqp_variational import IQPVariationalClassifier
from .qml_benchmarks.models.projected_quantum_kernel import ProjectedQuantumKernel
from .qml_benchmarks.models.quantum_boltzmann_machine import (
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable,
)
from .qml_benchmarks.models.quantum_kitchen_sinks import QuantumKitchenSinks
from .qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner
from .qml_benchmarks.models.quanvolutional_neural_network import (
    QuanvolutionalNeuralNetwork,
)
from .qml_benchmarks.models.separable import (
    SeparableKernelClassifier,
    SeparableVariationalClassifier,
)
from .qml_benchmarks.models.tree_tensor import TreeTensorClassifier
from .qml_benchmarks.models.vanilla_qnn import VanillaQNN
from .qml_benchmarks.models.weinet import WeiNet

# import ParallelGradients
# from .qml_benchmarks.models.parallel_gradients import ParallelGradients

__all__ = [
    "qml_benchmarks",
    "CircuitCentricClassifier",
    "ConvolutionalNeuralNetwork",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierNoCost",
    "DataReuploadingClassifierNoScaling",
    "DataReuploadingClassifierNoTrainableEmbedding",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierOnlyNN",
    "DressedQuantumCircuitClassifierSeparable",
    "IQPKernelClassifier",
    "IQPVariationalClassifier",
    "ProjectedQuantumKernel",
    "QuantumBoltzmannMachine",
    "QuantumBoltzmannMachineSeparable",
    "QuantumKitchenSinks",
    "QuantumMetricLearner",
    "QuanvolutionalNeuralNetwork",
    "SeparableKernelClassifier",
    "SeparableVariationalClassifier",
    "TreeTensorClassifier",
    "VanillaQNN",
    "WeiNet",
]
