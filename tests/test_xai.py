import pytest
import shap

from quoptuna.backend.models import MLPClassifier
from quoptuna.backend.xai import XAI


@pytest.fixture
def load_data():
    # Load the Iris dataset
    x, y = shap.datasets.adult(n_points=100)
    return {"x_train": x, "y_train": y}


@pytest.fixture
def trained_model(load_data):
    # Train a simple model on the Iris dataset
    model = MLPClassifier()
    model.fit(load_data["x_train"], load_data["y_train"])
    return model


def test_xai_initialization(trained_model, load_data):
    # Test the initialization of the XAI class
    xai = XAI(model=trained_model, data=load_data)
    assert isinstance(xai, XAI)
    assert xai.use_proba is True  # Check the default value of use_proba
    assert xai.onsubset is True  # Check the default value of onsubset


def test_get_explainer(trained_model, load_data):
    # Test the get_explainer method
    xai = XAI(model=trained_model, data=load_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)


def test_validate_predict_proba(trained_model, load_data):
    # Test the validate_predict_proba method
    xai = XAI(model=trained_model, data=load_data)
    assert xai.validate_predict_proba() is True


def test_invalid_model(load_data):
    # Test initialization with an invalid model
    with pytest.raises(TypeError):
        XAI(model=None, data=load_data)  # Instantiate the XAI object
