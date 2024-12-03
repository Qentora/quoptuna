from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import shap
from shap import Explainer

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from quoptuna.backend.data_typing import DataSet

EXPECTED_SHAP_VALUES_DIM = 2
DATA_KEY = "x_train"


class XAI:
    def __init__(
        self,
        model: BaseEstimator,
        data: DataSet,
        use_proba: bool = True,  # noqa: FBT001, FBT002
        onsubset: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        if model is None:
            msg = "Model cannot be None"
            raise TypeError(msg)
        self.model = model
        self.data = data
        self.onsubset = onsubset
        self.use_proba = use_proba
        self.classes = self.get_classes()
        if self.use_proba:
            self.validate_predict_proba()
        self.explainer = self.get_explainer()
        self.shap_values = self.get_shap_values()
        self.shap_values_each_class = self.set_shap_values_classes()

    def set_shap_values_classes(self) -> dict[str, shap.Explanation] | None:
        if not (
            self.shap_values and self.shap_values.values.ndim > EXPECTED_SHAP_VALUES_DIM  # noqa: PD011
        ):
            return None
        return self.get_shap_values_each_class(self.shap_values)

    def get_classes(self) -> dict[int, str]:
        if not hasattr(self.model, "classes_"):
            msg = "Model does not have a classes_ attribute"
            raise TypeError(msg)
        return self.model.classes_

    def get_explainer(self) -> Explainer:
        predict_method = self.model.predict_proba if self.use_proba else self.model.predict
        data = self.data.get(DATA_KEY)
        if not isinstance(data, pd.DataFrame):
            msg = f"Expected {DATA_KEY} to be a pandas DataFrame"
            raise TypeError(msg)
        return Explainer(predict_method, data)

    def validate_predict_proba(self) -> bool:
        if not hasattr(self.model, "predict_proba"):
            msg = "Model does not have a predict_proba method"
            raise TypeError(msg)
        return True

    def get_shap_values(self) -> shap.Explanation:
        if not isinstance(self.explainer, Explainer):
            msg = "explainer is not of type shap.Explainer"
            raise TypeError(msg)
        data = self.data.get(DATA_KEY)
        if not isinstance(data, pd.DataFrame):
            msg = f"Expected {DATA_KEY} to be a pandas DataFrame"
            raise TypeError(msg)
        if self.onsubset:
            data = data.iloc[:100]
        return self.explainer(data)

    def get_shap_values_each_class(
        self, shap_values: shap.Explanation
    ) -> dict[str, shap.Explanation]:
        if shap_values.values.ndim < EXPECTED_SHAP_VALUES_DIM:  # noqa: PD011
            msg = "shap_values has less than 2 dimensions"
            raise TypeError(msg)
        return {c: shap_values[:, :, i] for i, c in self.classes.items()}

    def get_bar_plot(self):
        if self.shap_values.values.ndim == EXPECTED_SHAP_VALUES_DIM:  # noqa: PD011
            return self.custom_shap_bar_plot(self.shap_values, max_display=20)
        first_class = next(iter(self.classes))
        return self.custom_shap_bar_plot(self.shap_values_each_class[first_class], max_display=20)

    def get_beeswarm_plot(self):
        if self.shap_values.values.ndim == EXPECTED_SHAP_VALUES_DIM:  # noqa: PD011
            return self.custom_shap_beeswarm_plot(self.shap_values, max_display=20)
        first_class = next(iter(self.classes))
        return self.custom_shap_beeswarm_plot(
            self.shap_values_each_class[first_class], max_display=20
        )

    def custom_shap_beeswarm_plot(self, shap_values, max_display=20):
        # Call the original shap.plots.beeswarm function
        shap.plots.beeswarm(shap_values, max_display=max_display)

        # Capture the current figure
        return plt.gcf()

    def custom_shap_bar_plot(self, shap_values, max_display=20):
        # Call the original shap.plots.bar function
        shap.plots.bar(shap_values, max_display=max_display)

        # Capture the current figure
        return plt.gcf()
