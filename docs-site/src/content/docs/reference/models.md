---
title: Model catalog
description: Quantum and classical models available in QuOptuna and their searchable hyperparameters.
---

QuOptuna searches over quantum models (PennyLane) and classical models (scikit-learn). Searchable hyperparameters below come from `MODEL_PARAM_KEYS`.

:::note
**Kernel** models (IQPKernelClassifier, ProjectedQuantumKernel, QuantumKitchenSinks, SeparableKernelClassifier) have no iterative training steps and are **not prunable**. **Variational** models are trained iteratively and **are prunable**. Variational quantum models are OvR-wrapped for multiclass. `TreeTensorClassifier` is a **quantum** model despite the name.
:::

## Quantum models (PennyLane)

| Model | Type | Description | Searchable hyperparameters |
| --- | --- | --- | --- |
| CircuitCentricClassifier | variational | Circuit-centric variational classifier | max_vmap, batch_size, learning_rate, n_input_copies, n_layers |
| DataReuploadingClassifier | variational | Data re-uploading variational classifier | max_vmap, batch_size, learning_rate, n_layers, observable_type |
| DataReuploadingClassifierSeparable | variational | Separable data re-uploading classifier | max_vmap, batch_size, learning_rate, n_layers, observable_type |
| DressedQuantumCircuitClassifier | variational | Dressed quantum circuit (classical + quantum layers) | max_vmap, batch_size, learning_rate, n_layers |
| DressedQuantumCircuitClassifierSeparable | variational | Separable dressed quantum circuit | max_vmap, batch_size, learning_rate, n_layers |
| QuantumBoltzmannMachineSeparable | variational | Separable quantum Boltzmann machine | max_vmap, batch_size, learning_rate, visible_qubits, temperature |
| TreeTensorClassifier | variational | Tree-tensor-network circuit classifier | max_vmap, batch_size, learning_rate |
| IQPKernelClassifier | kernel | IQP-embedding quantum kernel classifier | max_vmap, repeats, C |
| ProjectedQuantumKernel | kernel | Projected quantum kernel classifier | max_vmap, gamma_factor, C, trotter_steps, t |
| QuantumKitchenSinks | kernel | Quantum kitchen sinks feature map | max_vmap, n_qfeatures, n_episodes |
| QuantumMetricLearner | variational | Quantum metric learning classifier | max_vmap, batch_size, learning_rate, n_layers |
| SeparableVariationalClassifier | variational | Separable variational classifier | batch_size, learning_rate, encoding_layers |
| SeparableKernelClassifier | kernel | Separable quantum kernel classifier | C, encoding_layers |

## Classical models (scikit-learn)

| Model | Description | Searchable hyperparameters |
| --- | --- | --- |
| SVC | Support vector classifier (RBF) | gamma, C, class_weight |
| SVClinear | Linear SVC (`LinearSVC`) | C, class_weight |
| MLPClassifier | Multi-layer perceptron (`hidden_layer_sizes` required) | batch_size, hidden_layer_sizes, alpha, learning_rate |
| Perceptron | Linear perceptron | eta0, class_weight |

## See also

- [Python API reference](/reference/python-api/)
- [CLI reference](/reference/cli/)
- [Configuration reference](/reference/configuration/)
