import pytest
import shap
from sklearn.datasets import load_iris

from quoptuna.backend.models import MLPClassifier
from quoptuna.backend.xai import XAI


@pytest.fixture
def iris_data():
    # Load the Iris dataset
    data = load_iris()
    return {"x_train": data.data, "y_train": data.target}


@pytest.fixture
def trained_model(iris_data):
    # Train a simple model on the Iris dataset
    model = MLPClassifier()
    model.fit(iris_data["x_train"], iris_data["y_train"])
    return model


def test_xai_initialization(trained_model, iris_data):
    # Test the initialization of the XAI class
    xai = XAI(model=trained_model, data=iris_data)
    assert xai.model == trained_model
    assert xai.data == iris_data
    assert xai.explainer_type == "shap-permutation"


def test_get_explainer(trained_model, iris_data):
    # Test the get_explainer method
    xai = XAI(model=trained_model, data=iris_data)
    explainer = xai.get_explainer()
    assert isinstance(explainer, shap.Explainer)


def test_validate_predict_proba(trained_model):
    # Test the validate_predict_proba method
    xai = XAI(model=trained_model, data={})
    assert xai.validate_predict_proba() is True


def test_invalid_model():
    # Test initialization with an invalid model
    msg = "Model does not have a predict_proba method"
    with pytest.raises(ValueError, match=msg):
       XAI(model=None, data={})  # Instantiate the XAI object
