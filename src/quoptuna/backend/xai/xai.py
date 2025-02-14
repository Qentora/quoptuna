from __future__ import annotations

import base64
import io
import pickle
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import pandas as pd
import shap
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
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
        x_test_key: str = "x_test",
        y_test_key: str = "y_test",
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
        if feature_names is None and isinstance(self.data.get(data_key), pd.DataFrame):
            self.feature_names = list(self.data.get(data_key).columns)
        else:
            self.feature_names = feature_names
        if self.use_proba:
            self.validate_predict_proba()
        self.x_test_key = x_test_key
        self.y_test_key = y_test_key

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
        save_path: str | None = None,
        save_name: str | None = None,
        save_format: str = "png",
        save_dpi: int = 300,
        plot_title: str | None = None,
    ):
        """Generic method to get either bar or beeswarm plot."""
        try:
            # Initialize values with default
            values = self.shap_values

            # Override values if class-specific case
            if self.shap_values.values.ndim > EXPECTED_SHAP_VALUES_DIM:  # noqa: PD011
                values = self.shap_values_each_class[str(class_index)]

            plt.figure()  # Create a new figure for the plot
            plt.title(plot_title)  # Add the title to the empty plot

            if plot_type != "waterfall":
                plot_func = {
                    "bar": shap.plots.bar,
                    "beeswarm": shap.plots.beeswarm,
                    "heatmap": shap.plots.heatmap,
                    "violin": shap.plots.violin,
                }[plot_type]

                plot_func(values, max_display=max_display, show=False)
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format="png")
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
                base64_code = f"data:image/png;base64,{img_base64}"
                if save_path and save_name:
                    # save the plot
                    plt.savefig(Path(save_path) / save_name, format=save_format, dpi=save_dpi)
                plt.close()
                return base64_code
            plot_func = shap.plots.waterfall
            plot_func(values[index], show=False)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
            base64_code = f"data:image/png;base64,{img_base64}"
            if save_path and save_name:
                # save the plot
                plt.savefig(Path(save_path) / save_name, format=save_format, dpi=save_dpi)
            return base64_code
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            return values
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

    def get_average_precision_score(self):
        """Get the average precision score of the model."""
        return average_precision_score(self.y_test, self.predictions_proba)

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

    def get_classification_report(self):
        """Get the classification report of the model."""
        return classification_report(self.y_test, self.predictions)

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

        for key, func in metrics.items():
            try:
                report[key] = func()
            except Exception as e:
                report[key] = f"Error retrieving {key.replace('_', ' ')}: {e!s}"
        return report

    def plot_confusion_matrix(
        self,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        colorbar=True,
        im_kw=None,
        text_kw=None,
    ):
        """Plot the confusion matrix of the model."""
        from sklearn.metrics import ConfusionMatrixDisplay

        cm = self.get_confusion_matrix()
        ConfusionMatrixDisplay(cm).plot(
            include_values=include_values,
            cmap=cmap,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            ax=ax,
            colorbar=colorbar,
            im_kw=im_kw,
            text_kw=text_kw,
        )
        plt.title("Confusion Matrix")
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
        """Generates a comprehensive report using LangChain and a multimodal LLM."""

        if provider == "google":
            chat = ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name)
        elif provider == "openai":
            chat = ChatOpenAI(openai_api_key=api_key, model_name=model_name)
        else:
            raise ValueError(f"Invalid provider: {provider}")

        report = self.get_report()
        report_string = str(report)

        images = {}

        # --- Other Plots (Bar, Beeswarm, etc.) ---
        for plot_type in ["bar", "beeswarm", "violin", "heatmap"]:  # Exclude waterfall here
            try:
                images[plot_type] = self.get_plot(plot_type)
            except Exception as e:
                msg = f"Error generating or encoding {plot_type} plot: {e}"
                raise ValueError(msg) from e

        # --- Waterfall Plots (Selected Randomly) ---
        try:
            if self.onsubset:
                num_waterfall_plots = min(num_waterfall_plots, self.subset_size)
            else:
                num_waterfall_plots = min(num_waterfall_plots, len(self.x_test))
            indices = sorted(random.sample(range(num_waterfall_plots), num_waterfall_plots))
            for i in indices:
                images[f"waterfall_{i}"] = self.get_plot("waterfall", index=i)
        except Exception as e:
            msg = f"Error generating or encoding waterfall plots: {e}"
            raise ValueError(msg) from e

        # --- Confusion Matrix ---
        try:
            fig = self.plot_confusion_matrix()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png")
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
            images["confusion_matrix"] = f"data:image/png;base64,{img_base64}"
            plt.close(fig)
        except Exception as e:
            msg = f"Error generating or encoding confusion matrix: {e}"
            raise ValueError(msg) from e

        # --- LangChain Prompt and LLM Call ---
        prompt1 = """
            You are an AI assistant that analyzes model evaluation reports and generates insightful summaries. 
            You are provided with evaluation metrics and explanations of feature importance from SHAP values, along with visualizations. 
            Use all of the available information to generate a comprehensive report. 
            Be sure to explain what the metrics mean in context, and how they relate to the model's performance. 
            For the SHAP values, explain which features are most important, and how they influence the model's predictions. 
            For the confusion matrix, explain what the different quadrants represent, and what the overall accuracy is. 
            If there are any potential problems with the model, be sure to point them out.   
            """
        prompt2 = """
            System Role:
            You are an AI assistant specializing in analyzing model evaluation reports to generate comprehensive, governance-oriented summaries. Your goal is to provide detailed, data-driven insights into the model’s performance, feature importance, and potential risks, aligning with AI governance principles such as transparency, fairness, and accountability.

            Key Instructions for Model Evaluation Report:
                1.	Evaluation Metrics Analysis:
                •	Clearly explain each metric (e.g., True Positives (TP), False Negatives (FN), Precision, Recall, F1-score, Accuracy).
                •	Interpret what these metrics indicate about the model’s performance, strengths, weaknesses, and potential biases.
                •	Provide precise calculations where applicable, referencing actual values (e.g., confusion matrix counts, accuracy percentages).
                2.	Detailed SHAP Value and Feature Importance Analysis:
            Analyze SHAP visualizations and bar plots to identify key features:
                •	For Each SHAP Plot (Bar Plot, Beeswarm Plot, Violin Plot, Heatmap, Waterfall Plot):
                •	Describe the plot—what is shown visually.
                •	Interpret the data accurately based on provided plots without adding extra assumptions.
                •	Numerically quantify feature importance where applicable (e.g., “Feature X has a SHAP value of 0.45, indicating strong positive influence on predictions”).
                •	In the Bar Plot, identify which features are most relevant to the model’s predictions and quantify how much they contribute (e.g., “Feature A contributes 30% of the total SHAP value importance”).
                3.	Risk and Fairness Assessment:
                •	Identify potential risks such as overfitting, bias, or data drift based on observed metrics and feature contributions.
                •	Highlight any indications of unfair bias in feature importance, especially if certain features dominate the model’s decisions disproportionately.
                •	Recommend further fairness audits where necessary.
                4.	Governance and Compliance Recommendations:
                •	Provide actionable recommendations for improving model robustness, fairness, and explainability.
                •	Suggest best practices for validation, monitoring, and accountability, aligned with regulatory standards.
                •	If perfect or near-perfect accuracy is observed, recommend robustness testing and cross-validation to rule out overfitting.
                5.	Model Lifecycle Context:
                •	Frame the report within the AI model lifecycle, including development, deployment, and monitoring phases.
                •	Emphasize continuous model evaluation, with recommendations for regular performance audits.

            Tone and Structure:
                •	Maintain a formal, precise, and governance-compliant writing style.
                •	Use clear section headers, bullet points, and actionable recommendations.
                •	Only include insights based on provided data—do not infer or assume beyond what the evaluation metrics and plots reveal.

            Primary Objective:
            Deliver a report that promotes trust, transparency, and accountability in AI systems. Your analysis should support technical teams, compliance officers, and governance stakeholders in making informed, data-driven decisions based on the provided evaluation metrics and visualizations.
            """
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

        final_prompt = prompt_template.format_messages(report=report_string, images=image_messages)

        response = chat(final_prompt)
        return response.content
