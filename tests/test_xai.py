import matplotlib.pyplot as plt
import pytest
import shap

from quoptuna import XAI
from quoptuna.backend.models import MLPClassifier


@pytest.fixture
def load_data():
    x, y = shap.datasets.adult(n_points=100)
    #  convert y to int bow y is a list of bool
    y = y.astype(int)
    return {"x_train": x, "y_train": y}


@pytest.fixture
def trained_model(load_data):
    # Train a simple model on the Iris dataset
    model = MLPClassifier()
    model.fit(load_data["x_train"], load_data["y_train"])
    return model


def test_load_data(load_data):
    assert load_data is not None
    # check if all values in y are 0 or 1 or -1
    assert all(y in [0, 1, -1] for y in load_data["y_train"])


def test_xai_initialization(trained_model, load_data):
    # Test the initialization of the XAI class
    xai = XAI(model=trained_model, data=load_data)
    assert isinstance(xai, XAI)
    assert xai.use_proba is True  # Check the default value of use_proba
    assert xai.onsubset is True  # Check the default value of onsubset


def test_get_explainer(trained_model, load_data):
    # Test the get_explainer method
    xai = XAI(model=trained_model, data=load_data)
    explainer = xai._get_explainer()
    assert isinstance(explainer, shap.Explainer)


def test_validate_predict_proba(trained_model, load_data):
    # Test the validate_predict_proba method
    xai = XAI(model=trained_model, data=load_data)
    assert xai.validate_predict_proba() is True


def test_invalid_model(load_data):
    # Test initialization with an invalid model
    with pytest.raises(TypeError):
        XAI(model=None, data=load_data)  # Instantiate the XAI object


# New tests added below
@pytest.fixture
def trained_model_sample(load_data):
    # Train a simple model
    model = MLPClassifier()
    model.fit(load_data["x_train"], load_data["y_train"])
    return model


def test_get_shap_values(trained_model_sample, load_data):
    # Test getting SHAP values
    xai = XAI(model=trained_model_sample, data=load_data, onsubset=True, subset_size=5)
    shap_values = xai._get_shap_values()
    assert shap_values is not None
    assert hasattr(shap_values, "values")


def test_trained_model_sample(load_data):
    # Train a simple model
    model = MLPClassifier()
    model.fit(load_data["x_train"], load_data["y_train"])
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "fit")
    assert hasattr(model, "classes_")
    assert model.classes_ is not None
    assert len(model.classes_) > 0


def test_get_classes(trained_model_sample, load_data):
    # Test getting classes
    model = MLPClassifier()
    model.fit(load_data["x_train"], load_data["y_train"])
    xai = XAI(model=model, data=load_data)
    classes = xai.get_classes()
    assert len(classes) > 0


# Tests for plotting methods
def test_get_bar_plot(trained_model_sample, load_data):
    # Test the get_bar_plot method
    xai = XAI(model=trained_model_sample, data=load_data, onsubset=True, subset_size=5)
    bar_plot = xai.get_bar_plot(class_index=0)
    assert isinstance(bar_plot, plt.Figure)  # Check if a figure is returned


def test_get_beeswarm_plot(trained_model_sample, load_data):
    # Test the get_beeswarm_plot method
    xai = XAI(model=trained_model_sample, data=load_data, onsubset=True, subset_size=5)
    beeswarm_plot = xai.get_beeswarm_plot(class_index=0)
    assert isinstance(beeswarm_plot, plt.Figure)  # Check if a figure is returned


def test_get_waterfall_plot(trained_model_sample, load_data):
    # Test the get_waterfall_plot method
    xai = XAI(model=trained_model_sample, data=load_data, onsubset=True, subset_size=5)
    waterfall_plot = xai.get_waterfall_plot(index=0, class_index=0)
    assert isinstance(waterfall_plot, plt.Figure)  # Check if a figure is returned


def test_get_violin_plot(trained_model_sample, load_data):
    # Test the get_violin_plot method
    xai = XAI(model=trained_model_sample, data=load_data, onsubset=True, subset_size=5)
    violin_plot = xai.get_violin_plot(class_index=0)
    assert isinstance(violin_plot, plt.Figure)  # Check if a figure is returned


def test_get_heatmap_plot(trained_model_sample, load_data):
    # Test the get_heatmap_plot method
    xai = XAI(model=trained_model_sample, data=load_data, onsubset=True, subset_size=5)
    heatmap_plot = xai.get_heatmap_plot(class_index=0)
    assert isinstance(heatmap_plot, plt.Figure)  # Check if a figure is returned
