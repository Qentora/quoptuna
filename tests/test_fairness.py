"""Tests for the fairlearn-based fairness audit utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from quoptuna import DataPreparation
from quoptuna.backend.xai import fairness as fm


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = 400
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=["c1", "c2", "c3", "c4"])
    sens = pd.Series(rng.choice(["g1", "g2"], size=n), name="group")
    y = np.where(x["c1"] + (sens == "g1") * 0.8 + rng.normal(scale=0.5, size=n) > 0, 1, -1)
    model = LogisticRegression().fit(x.values, y)
    return x, y, sens, model


def test_compute_fairness_structure(synthetic):
    x, y, sens, model = synthetic
    result = fm.compute_fairness(y, model.predict(x.values), sens)

    assert set(result) == {"by_group", "overall", "disparities"}
    assert set(result["by_group"]["accuracy"]) == {"g1", "g2"}
    assert 0 <= result["disparities"]["demographic_parity_ratio"] <= 1
    assert result["by_group"]["count"]["g1"] + result["by_group"]["count"]["g2"] == len(y)


def test_plot_group_metrics_returns_data_urls(synthetic):
    x, y, sens, model = synthetic
    result = fm.compute_fairness(y, model.predict(x.values), sens)
    plots = fm.plot_group_metrics(result)

    assert "selection_rate" in plots
    assert "count" not in plots
    assert all(url.startswith("data:image/png;base64,") for url in plots.values())


def test_threshold_optimizer_mitigation(synthetic):
    x, y, sens, model = synthetic
    result = fm.mitigate_with_threshold_optimizer(model, x, y, sens, x, y, sens)

    assert set(result) == {"constraint", "before", "after", "comparison_plot"}
    # Equalized-odds constraint should not worsen the EO gap.
    assert (
        result["after"]["disparities"]["equalized_odds_difference"]
        <= result["before"]["disparities"]["equalized_odds_difference"] + 0.05
    )
    assert result["comparison_plot"].startswith("data:image/png;base64,")


def test_split_index_alignment_with_raw_dataframe():
    """The seeded split's indices must be positional rows into the raw dataframe.

    This invariant is what lets the /fairness endpoint align a raw-CSV column
    (never seen by DataPreparation) with x_test rows.
    """
    rng = np.random.default_rng(1)
    n = 200
    raw = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "group": rng.choice(["a", "b"], size=n),
            "target": rng.choice([0, 1], size=n),
        }
    )
    prep = DataPreparation(
        dataset={"x": raw[["f1", "f2"]], "y": raw["target"]},
        x_cols=["f1", "f2"],
        y_col="target",
    )
    # Recover raw f1 values by positional index and undo the standard scaling.
    recovered = prep.x_test["f1"] * raw["f1"].std(ddof=0) + raw["f1"].mean()
    expected = raw["f1"].iloc[prep.x_test.index]
    assert np.allclose(recovered.values, expected.values)
    # A raw categorical column aligns the same way.
    sens_test = raw["group"].iloc[prep.x_test.index]
    assert len(sens_test) == len(prep.x_test)
