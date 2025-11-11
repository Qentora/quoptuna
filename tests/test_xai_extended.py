"""Extended unit tests for XAI module."""


import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.linear_model import LogisticRegression

from quoptuna.backend.xai.xai import XAI, XAIConfig


@pytest.fixture
def xai_instance_with_data(train_test_split_dataframes):
    """Create an XAI instance with trained model and data."""
    model = LogisticRegression(random_state=42)
    model.fit(train_test_split_dataframes["x_train"], train_test_split_dataframes["y_train"])

    config = XAIConfig(use_proba=True, onsubset=True, subset_size=20)
    xai = XAI(model=model, data=train_test_split_dataframes, config=config)
    return xai


def test_xai_config_defaults():
    """Test XAIConfig default values."""
    config = XAIConfig()

    assert config.use_proba is True
    assert config.onsubset is True
    assert config.feature_names is None
    assert config.subset_size == 100
    assert config.max_display == 20
    assert config.data_key == "x_train"


def test_xai_config_custom_values():
    """Test XAIConfig with custom values."""
    config = XAIConfig(
        use_proba=False,
        onsubset=False,
        feature_names=["f1", "f2"],
        subset_size=50,
        max_display=10,
    )

    assert config.use_proba is False
    assert config.onsubset is False
    assert config.feature_names == ["f1", "f2"]
    assert config.subset_size == 50
    assert config.max_display == 10


def test_xai_initialization_none_model(train_test_split_dataframes):
    """Test XAI initialization with None model raises TypeError."""
    with pytest.raises(TypeError, match="Model cannot be None"):
        XAI(model=None, data=train_test_split_dataframes)


def test_xai_initialization_default_config(trained_classifier, train_test_split_dataframes):
    """Test XAI initialization with default config."""
    xai = XAI(model=trained_classifier, data=train_test_split_dataframes)

    assert xai.use_proba is True
    assert xai.onsubset is True
    assert xai.config is not None


def test_xai_feature_names_from_dataframe(trained_classifier, train_test_split_dataframes):
    """Test that feature names are extracted from DataFrame."""
    xai = XAI(model=trained_classifier, data=train_test_split_dataframes)

    assert xai.feature_names is not None
    assert len(xai.feature_names) == train_test_split_dataframes["x_train"].shape[1]


def test_xai_x_test_property(xai_instance_with_data, train_test_split_dataframes):
    """Test x_test property."""
    x_test = xai_instance_with_data.x_test

    assert x_test is not None
    pd.testing.assert_frame_equal(x_test, train_test_split_dataframes["x_test"])


def test_xai_y_test_property(xai_instance_with_data, train_test_split_dataframes):
    """Test y_test property."""
    y_test = xai_instance_with_data.y_test

    assert y_test is not None
    pd.testing.assert_series_equal(y_test, train_test_split_dataframes["y_test"])


def test_xai_predictions_property(xai_instance_with_data):
    """Test predictions property."""
    predictions = xai_instance_with_data.predictions

    assert predictions is not None
    assert len(predictions) == len(xai_instance_with_data.x_test)


def test_xai_predictions_proba_property(xai_instance_with_data):
    """Test predictions_proba property."""
    predictions_proba = xai_instance_with_data.predictions_proba

    assert predictions_proba is not None
    assert len(predictions_proba) == len(xai_instance_with_data.x_test)


def test_xai_explainer_property(xai_instance_with_data):
    """Test explainer property."""
    explainer = xai_instance_with_data.explainer

    assert explainer is not None
    assert isinstance(explainer, shap.Explainer)


def test_xai_shap_values_property(xai_instance_with_data):
    """Test shap_values property."""
    shap_values = xai_instance_with_data.shap_values

    assert shap_values is not None
    assert hasattr(shap_values, "values")


def test_xai_get_classes(xai_instance_with_data):
    """Test get_classes method."""
    classes = xai_instance_with_data.get_classes()

    assert classes is not None
    assert len(classes) > 0


def test_xai_model_without_predict_proba(train_test_split_dataframes):
    """Test XAI with model without predict_proba method."""

    class DummyModel:
        def predict(self, X):
            return np.ones(len(X))

        @property
        def classes_(self):
            return np.array([0, 1])

    model = DummyModel()
    config = XAIConfig(use_proba=True)

    with pytest.raises(TypeError, match="Model does not have a predict_proba method"):
        XAI(model=model, data=train_test_split_dataframes, config=config)


def test_xai_without_use_proba(train_test_split_dataframes):
    """Test XAI without using predict_proba."""
    model = LogisticRegression(random_state=42)
    model.fit(train_test_split_dataframes["x_train"], train_test_split_dataframes["y_train"])

    config = XAIConfig(use_proba=False, onsubset=True, subset_size=20)
    xai = XAI(model=model, data=train_test_split_dataframes, config=config)

    assert xai.use_proba is False
    assert xai.explainer is not None


def test_xai_confusion_matrix(xai_instance_with_data):
    """Test get_confusion_matrix method."""
    cm = xai_instance_with_data.get_confusion_matrix()

    assert cm is not None
    assert cm.shape == (2, 2)  # Binary classification


def test_xai_classification_report(xai_instance_with_data):
    """Test get_classification_report method."""
    report = xai_instance_with_data.get_classification_report()

    assert report is not None
    assert isinstance(report, str)


def test_xai_f1_score(xai_instance_with_data):
    """Test get_f1_score method."""
    f1 = xai_instance_with_data.get_f1_score()

    assert f1 is not None
    assert 0 <= f1 <= 1


def test_xai_precision(xai_instance_with_data):
    """Test get_precision method."""
    precision = xai_instance_with_data.get_precision()

    assert precision is not None
    assert 0 <= precision <= 1


def test_xai_recall(xai_instance_with_data):
    """Test get_recall method."""
    recall = xai_instance_with_data.get_recall()

    assert recall is not None
    assert 0 <= recall <= 1


def test_xai_mcc(xai_instance_with_data):
    """Test get_mcc method."""
    mcc = xai_instance_with_data.get_mcc()

    assert mcc is not None
    assert -1 <= mcc <= 1


def test_xai_cohens_kappa(xai_instance_with_data):
    """Test get_cohens_kappa method."""
    kappa = xai_instance_with_data.get_cohens_kappa()

    assert kappa is not None
    assert -1 <= kappa <= 1


def test_xai_get_report(xai_instance_with_data):
    """Test get_report method."""
    report = xai_instance_with_data.get_report()

    assert report is not None
    assert isinstance(report, dict)
    assert "confusion_matrix" in report
    # Note: Some metrics may fail depending on the data, so we just check the dict is returned


def test_xai_save_and_load_state(xai_instance_with_data, tmp_path):
    """Test save_state and load_state methods."""
    file_path = tmp_path / "xai_state.pkl"

    # Save state
    xai_instance_with_data.save_state(str(file_path))

    # Verify file was created
    assert file_path.exists()

    # Load state
    loaded_xai = XAI.load_state(str(file_path))

    # Verify loaded instance has same configuration
    assert loaded_xai.use_proba == xai_instance_with_data.use_proba
    assert loaded_xai.onsubset == xai_instance_with_data.onsubset
    assert loaded_xai.subset_size == xai_instance_with_data.subset_size


def test_xai_plot_confusion_matrix(xai_instance_with_data):
    """Test plot_confusion_matrix method."""
    fig = xai_instance_with_data.plot_confusion_matrix()

    assert fig is not None


def test_xai_str_method(xai_instance_with_data):
    """Test __str__ method."""
    str_repr = str(xai_instance_with_data)

    assert str_repr is not None
    assert isinstance(str_repr, str)


def test_xai_with_custom_keys(trained_classifier):
    """Test XAI with custom data keys."""
    # Create data with custom keys
    X, y = shap.datasets.adult(n_points=100)
    y = y.astype(int)

    data = {
        "custom_train_x": X,
        "custom_train_y": y,
        "custom_test_x": X[:20],
        "custom_test_y": y[:20],
    }

    config = XAIConfig(
        data_key="custom_train_x",
        x_test_key="custom_test_x",
        y_test_key="custom_test_y",
        onsubset=True,
        subset_size=20,
    )

    xai = XAI(model=trained_classifier, data=data, config=config)

    assert xai.data_key == "custom_train_x"
    assert xai.x_test_key == "custom_test_x"
    assert xai.y_test_key == "custom_test_y"


# Note: Plot tests are already covered in test_xai.py and depend on specific
# data shapes and SHAP library behavior, so we don't duplicate them here
