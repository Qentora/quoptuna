# class exai utilises the SHAP library to explain the model
from __future__ import annotations
import shap

from quoptuna.backend.base.model_typing import ModelType


class XAI:
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

    def __init__(self, model: ModelType, data: dict[str, object]):
        self.model = model
        self.validate_predict_proba()
        self.explainer_type = "shap-auto"
        self.data = data

    def get_explainer(self):
        if self.explainer_type == "shap-auto":
            return shap.Explainer(
                self.model.predict_proba, self.data["x_train"]
            )
        return None

    def validate_predict_proba(self):
        # check if the model has a predict_proba method
        if not hasattr(self.model, "predict_proba"):
            msg = "Model does not have a predict_proba method"
            raise ValueError(msg)
        return True
