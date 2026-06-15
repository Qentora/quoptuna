from __future__ import annotations

import base64
import io
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import pandas as pd
import shap
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from shap import Explainer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from quoptuna.backend.typing.data_typing import DataSet

# Constants
EXPECTED_SHAP_VALUES_DIM = 2
DATA_KEY = "x_train"
DEFAULT_MAX_DISPLAY = 20
DEFAULT_SUBSET_SIZE = 100
NON_CLASS_SHAP_PLOT_TYPES = Literal["bar", "beeswarm", "violin", "heatmap"]
CLASS_SHAP_PLOT_TYPES = Literal["waterfall"]
CONFUSION_MATRIX_PLOT_TYPES = Literal["confusion_matrix"]
PlotType = Literal[NON_CLASS_SHAP_PLOT_TYPES, CLASS_SHAP_PLOT_TYPES, CONFUSION_MATRIX_PLOT_TYPES]


@dataclass
class XAIConfig:
    use_proba: bool = True
    onsubset: bool = True
    feature_names: list[str] | None = None
    subset_size: int = DEFAULT_SUBSET_SIZE
    max_display: int = DEFAULT_MAX_DISPLAY
    data_key: str = DATA_KEY
    x_test_key: str = "x_test"
    y_test_key: str = "y_test"


class XAI:
    def __init__(
        self,
        model: BaseEstimator,
        data: DataSet,
        config: XAIConfig | None = None,
    ) -> None:
        if model is None:
            msg = "Model cannot be None"
            raise TypeError(msg)

        self.config = config or XAIConfig()
        self.model = model
        self.data = data

        # Explicitly declare instance attributes
        self.use_proba: bool = self.config.use_proba
        self.onsubset: bool = self.config.onsubset
        self.feature_names: list[str] | None = self.config.feature_names
        self.subset_size: int = self.config.subset_size
        self.max_display: int = self.config.max_display
        self.data_key: str = self.config.data_key
        self.x_test_key: str = self.config.x_test_key
        self.y_test_key: str = self.config.y_test_key

        self._classes = self.get_classes
        data_frame = self.data.get(self.data_key)
        if self.feature_names is None and isinstance(data_frame, pd.DataFrame):
            self.feature_names = list(data_frame.columns)

        if self.use_proba:
            self.validate_predict_proba()

        # Initialize these as None, they'll be computed on demand
        self._explainer: Explainer | None = None
        self._shap_values: shap.Explanation | None = None
        self._shap_values_each_class: dict[str, shap.Explanation] | None = None
        self._x_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None
        self._predictions: pd.Series | None = None
        self._predictions_proba: pd.DataFrame | None = None

    @property
    def x_test(self) -> pd.DataFrame:
        if self._x_test is None:
            self._x_test = self.data.get(self.x_test_key)
        return self._x_test

    @property
    def y_test(self) -> pd.Series:
        if self._y_test is None:
            self._y_test = self.data.get(self.y_test_key)
        return self._y_test

    @property
    def predictions(self) -> pd.Series:
        if self._predictions is None:
            if not hasattr(self.model, "predict"):
                msg = "Model does not have a predict method"
                raise TypeError(msg)
            self._predictions = self.model.predict(self.x_test)
        return self._predictions

    @property
    def predictions_proba(self) -> pd.DataFrame:
        if self._predictions_proba is None:
            if not hasattr(self.model, "predict_proba"):
                msg = "Model does not have a predict_proba method"
                raise TypeError(msg)
            self._predictions_proba = self.model.predict_proba(self.x_test)
        return self._predictions_proba

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
        save_config: dict | None = None,
    ):
        """Generate plot with given configuration."""
        try:
            values = self._get_plot_values(class_index)
            return self._generate_plot(plot_type, values, max_display, index, save_config)
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            self._handle_plot_error(plot_type, e)

    def _generate_plot(
        self, plot_type: PlotType, values, max_display: int, index: int, save_config: dict | None
    ) -> str:
        plt.figure()
        if plot_type != "waterfall":
            plot_func = self._get_plot_function(plot_type)
            plot_func(values, max_display=max_display, show=False)
        else:
            shap.plots.waterfall(values[index], show=False)

        base64_code = self._save_plot_to_base64()

        if save_config is not None:  # Check if save_config exists
            self._save_plot_to_file(save_config)

        plt.close()
        return base64_code

    def _get_plot_values(self, class_index: int) -> shap.Explanation:
        if class_index == -1:
            return self.shap_values
        if self.shap_values.values.ndim > EXPECTED_SHAP_VALUES_DIM:  # noqa: PD011
            if self.shap_values_each_class is None:
                msg = "No class-specific SHAP values available"
                raise ValueError(msg)
            return self.shap_values_each_class[str(class_index)]
        return self.shap_values

    def _get_plot_function(self, plot_type: PlotType):
        return {
            "bar": shap.plots.bar,
            "beeswarm": shap.plots.beeswarm,
            "heatmap": shap.plots.heatmap,
            "violin": shap.plots.violin,
        }[plot_type]

    def _save_plot_to_base64(self) -> str:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def _save_plot_to_file(self, save_config: dict) -> None:
        save_path = save_config.get("save_path")
        save_name = save_config.get("save_name")
        if not save_path or not save_name:
            return

        plt.savefig(
            Path(save_path) / save_name,
            format=save_config.get("save_format", "png"),
            dpi=save_config.get("save_dpi", 300),
            bbox_inches="tight",
        )

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

    def save_state(self, file_path: str):
        """Saves the state of the class and its variables in a pkl file."""
        with Path(file_path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_state(cls, file_path: str):
        """Loads the state of the class from a pkl file.
        Warning: Only use this method with trusted data sources as pickle can be unsafe.
        """
        with Path(file_path).open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def get_confusion_matrix(self):
        """Get the confusion matrix of the model."""
        return confusion_matrix(self.y_test, self.predictions)

    def get_classification_report(self):
        """Get the classification report of the model."""
        return classification_report(self.y_test, self.predictions)

    def get_roc_curve(self):
        """Get the roc curve of the model."""
        return roc_curve(self.y_test, self.predictions_proba)

    def get_roc_auc_score(self):
        """Get the roc auc score of the model."""
        return roc_auc_score(self.y_test, self.predictions_proba)

    def get_precision_recall_curve(self):
        """Get the precision recall curve of the model."""
        return precision_recall_curve(self.y_test, self.predictions_proba)

    def get_average_precision_score(self):
        """Get the average precision score of the model."""
        return average_precision_score(self.y_test, self.predictions_proba)

    def get_f1_score(self):
        """Get the f1 score of the model."""
        return f1_score(self.y_test, self.predictions)

    def get_mcc(self):
        """Get the mcc of the model."""
        return matthews_corrcoef(self.y_test, self.predictions)

    def get_log_loss(self):
        """Get the log loss of the model."""
        return log_loss(self.y_test, self.predictions_proba)

    def get_cohens_kappa(self):
        """Get the cohens kappa of the model."""
        return cohen_kappa_score(self.y_test, self.predictions)

    def get_precision(self):
        """Get the precision of the model."""
        return precision_score(self.y_test, self.predictions)

    def get_recall(self):
        """Get the recall of the model."""
        return recall_score(self.y_test, self.predictions)

    def get_report(self):
        """Get the report of the model."""
        report = {}
        metrics = {
            "confusion_matrix": self.get_confusion_matrix,
            "classification_report": self.get_classification_report,
            "roc_curve": self.get_roc_curve,
            "roc_auc_score": self.get_roc_auc_score,
            "precision_recall_curve": self.get_precision_recall_curve,
            "average_precision_score": self.get_average_precision_score,
            "f1_score": self.get_f1_score,
            "mcc": self.get_mcc,
            "log_loss": self.get_log_loss,
            "cohens_kappa": self.get_cohens_kappa,
            "precision": self.get_precision,
            "recall": self.get_recall,
        }

        try:
            for key, func in metrics.items():
                report[key] = func()
        except (ValueError, TypeError) as e:
            report[key] = str(e)
        return report

    def plot_confusion_matrix(self, plot_config: dict | None = None):
        """Plot confusion matrix with given configuration."""
        from sklearn.metrics import ConfusionMatrixDisplay

        config = {
            "include_values": True,
            "cmap": "viridis",
            "xticks_rotation": "horizontal",
            "values_format": None,
            "ax": None,
            "colorbar": True,
            "im_kw": None,
            "text_kw": None,
            **(plot_config or {}),
        }

        cm = self.get_confusion_matrix()
        ConfusionMatrixDisplay(cm).plot(**config)

        if plot_config and plot_config.get("save_path"):
            plt.savefig(
                Path(plot_config["save_path"]) / plot_config["save_name"],
                format=plot_config.get("save_format", "png"),
                dpi=plot_config.get("save_dpi", 300),
                bbox_inches="tight",
            )

        return plt.gcf()

    def __str__(self):
        return str(self.get_report())

    def generate_report_with_langchain(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        provider: str = "google",
        num_waterfall_plots: int = 5,
    ):
        """Generate comprehensive report using LangChain and multimodal LLM."""
        chat = self._initialize_chat(api_key, model_name, provider)

        report = self.get_report()
        images = self._generate_report_images(num_waterfall_plots)

        prompt2 = Path("prompt.txt").read_text()
        return self._generate_final_report(chat, report, images, prompt2)

    def _initialize_chat(self, api_key: str, model_name: str, provider: str):
        if provider == "google":
            return ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name)
        if provider == "openai":
            return ChatOpenAI(openai_api_key=api_key, model_name=model_name)
        msg = "Invalid provider"
        raise ValueError(msg)

    def _generate_report_images(self, num_waterfall_plots: int):
        images: dict[str, str] = {}  # Change type hint to allow string keys
        plot_types: list[PlotType] = ["bar", "beeswarm", "violin", "heatmap"]

        try:
            for plot_type in plot_types:
                images[plot_type] = self.get_plot(plot_type)

            if self.onsubset:
                num_waterfall_plots = min(num_waterfall_plots, self.subset_size)
            else:
                num_waterfall_plots = min(num_waterfall_plots, len(self.x_test))

            indices = sorted(random.sample(range(num_waterfall_plots), num_waterfall_plots))
            for i in indices:
                waterfall_plot_type: PlotType = "waterfall"
                images[f"{waterfall_plot_type}_{i}"] = self.get_plot(waterfall_plot_type, index=i)

            fig = self.plot_confusion_matrix()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
            images["confusion_matrix"] = f"data:image/png;base64,{img_base64}"
            plt.close(fig)

        except Exception as e:
            msg = f"Error generating plots: {e}"
            raise ValueError(msg) from e

        return images

    def _generate_final_report(self, chat, report, images, prompt2):
        prompt_template = ChatPromptTemplate(
            messages=[
                SystemMessage(content=prompt2),
                HumanMessage(content="Model Evaluation Report:\n```\n{report}\n```"),
                MessagesPlaceholder(variable_name="images"),
            ]
        )
        image_messages = []
        for plot_type, image_url in images.items():
            image_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Here is a {plot_type.replace('_', ' ')} plot:"},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                    ]
                )
            )

        final_prompt = prompt_template.format_messages(report=str(report), images=image_messages)

        response = chat(final_prompt)
        return response.content
