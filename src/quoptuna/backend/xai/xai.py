from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import pandas as pd
import shap
from shap import Explainer

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from quoptuna.backend.typing.data_typing import DataSet

# Constants
EXPECTED_SHAP_VALUES_DIM = 2
DATA_KEY = "x_train"
DEFAULT_MAX_DISPLAY = 20
DEFAULT_SUBSET_SIZE = 100
PlotType = Literal["bar", "beeswarm"]


class XAI:
    def __init__(
        self,
        model: BaseEstimator,
        data: DataSet,
        use_proba: bool = True,  # noqa: FBT001, FBT002
        onsubset: bool = True,  # noqa: FBT001, FBT002
        feature_names: list[str] | None = None,
        subset_size: int = DEFAULT_SUBSET_SIZE,
        max_display: int = DEFAULT_MAX_DISPLAY,
        data_key: str = DATA_KEY,
    ) -> None:
        if model is None:
            msg = "Model cannot be None"
            raise TypeError(msg)
        self.model = model
        self.data = data
        self.onsubset = onsubset
        self.subset_size = subset_size
        self.max_display = max_display
        self.data_key = data_key
        self.use_proba = use_proba
        self._classes = self.get_classes  # Store the property value
        self.feature_names = feature_names
        if self.use_proba:
            self.validate_predict_proba()

        # Initialize these as None, they'll be computed on demand
        self._explainer: Explainer | None = None
        self._shap_values: shap.Explanation | None = None
        self._shap_values_each_class: dict[str, shap.Explanation] | None = None

    @property
    def explainer(self) -> Explainer:
        if self._explainer is None:
            self._explainer = self._get_explainer()
        return self._explainer

    @property
    def shap_values(self) -> shap.Explanation:
        if self._shap_values is None:
            self._shap_values = self._get_shap_values()
        return self._shap_values

    @property
    def shap_values_each_class(self) -> dict[str, shap.Explanation] | None:
        if self._shap_values_each_class is None:
            self._shap_values_each_class = self._set_shap_values_classes()
        return self._shap_values_each_class

    def _set_shap_values_classes(self) -> dict[str, shap.Explanation] | None:
        if not (self.shap_values and self.shap_values.values.ndim > EXPECTED_SHAP_VALUES_DIM):  # noqa: PD011
            return None
        return self._get_shap_values_each_class(self.shap_values)

    def get_classes(self) -> dict[int, str]:
        """Get model classes."""
        if not hasattr(self.model, "classes_"):
            msg = "Model does not have a classes_ attribute"
            raise TypeError(msg)
        return self.model.classes_

    def _get_explainer(self) -> Explainer:
        predict_method = self.model.predict_proba if self.use_proba else self.model.predict
        data = self._validate_and_get_data()
        if self.onsubset:
            data = data.iloc[: self.subset_size]
        return Explainer(model=predict_method, masker=data, feature_names=self.feature_names)

    def validate_predict_proba(self) -> bool:
        if not hasattr(self.model, "predict_proba"):
            msg = "Model does not have a predict_proba method"
            raise TypeError(msg)
        return True

    def _validate_and_get_data(self) -> pd.DataFrame:
        data = self.data.get(DATA_KEY)
        if not isinstance(data, pd.DataFrame):
            msg = f"Expected {DATA_KEY} to be a pandas DataFrame"
            raise TypeError(msg)
        return data

    def _get_shap_values(self) -> shap.Explanation:
        data = self._validate_and_get_data()
        if self.onsubset:
            data = data.iloc[: self.subset_size]
        return self.explainer(data)

    def _get_shap_values_each_class(
        self, shap_values: shap.Explanation
    ) -> dict[str, shap.Explanation]:
        if shap_values.values.ndim < EXPECTED_SHAP_VALUES_DIM:  # noqa: PD011
            msg = "shap_values has less than 2 dimensions"
            raise TypeError(msg)
        return {str(i): shap_values[:, :, i] for i in self.get_classes()}

    def _validate_shap_values_class(self) -> shap.Explanation:
        """Validate and get SHAP values for class-specific case."""
        if self.shap_values_each_class is None:
            msg = "No class-specific SHAP values available"
            raise ValueError(msg)
        first_class = next(iter(self.get_classes()))
        return self.shap_values_each_class[str(first_class)]

    def _handle_plot_error(self, plot_type: PlotType, error: Exception) -> None:
        """Handle plot generation errors."""
        if isinstance(error, (ValueError, TypeError, KeyError)):
            raise error
        msg = f"Error generating {plot_type} plot: {error!s}"
        raise RuntimeError(msg) from error

    def get_plot(
        self,
        plot_type: PlotType,
        max_display: int = DEFAULT_MAX_DISPLAY,
        class_index: int = -1,
        index: int = 0,
    ):
        """Generic method to get either bar or beeswarm plot."""
        try:
            # Initialize values with default
            values = self.shap_values

            # Override values if class-specific case
            if self.shap_values.values.ndim > EXPECTED_SHAP_VALUES_DIM and class_index >= 0:  # noqa: PD011
                values = self.shap_values_each_class[str(class_index)]

            if plot_type != "waterfall":
                plot_func = {
                    "bar": shap.plots.bar,
                    "beeswarm": shap.plots.beeswarm,
                    "heatmap": shap.plots.heatmap,
                    "violin": shap.plots.violin,
                }[plot_type]

                plot_func(values, max_display=max_display)
                return plt.gcf()
            plot_func = shap.plots.waterfall
            plot_func(values[index])
            return plt.gcf()
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            self._handle_plot_error(plot_type, e)

    def get_bar_plot(self, max_display: int = DEFAULT_MAX_DISPLAY, class_index: int = -1):
        return self.get_plot("bar", max_display, class_index=class_index)

    def get_beeswarm_plot(self, max_display: int = DEFAULT_MAX_DISPLAY, class_index: int = -1):
        return self.get_plot("beeswarm", max_display, class_index=class_index)

    def get_waterfall_plot(
        self, max_display: int = DEFAULT_MAX_DISPLAY, index: int = 0, class_index: int = -1
    ):
        return self.get_plot("waterfall", max_display, index=index, class_index=class_index)

    def get_violin_plot(self, max_display: int = DEFAULT_MAX_DISPLAY, class_index: int = -1):
        return self.get_plot("violin", max_display, class_index=class_index)

    def get_heatmap_plot(self, max_display: int = DEFAULT_MAX_DISPLAY, class_index: int = -1):
        return self.get_plot("heatmap", max_display, class_index=class_index)
