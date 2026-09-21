"""SHAP row progress must reach the caller, not only the server's terminal.

An explanation on a variational model is minutes of work. Without this the
browser shows an indeterminate "SHAP" step for the whole of it while the tqdm
bar ticks away in a terminal the user cannot see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from quoptuna.backend.xai import shap_progress

ROWS = 6
FEATURES = 3


def _explain(rows: int = ROWS):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(20, FEATURES)), columns=[f"f{i}" for i in range(FEATURES)]
    )
    model = LogisticRegression().fit(frame, (frame["f0"] > 0).astype(int))
    explainer = shap.Explainer(
        model.predict,
        masker=shap.maskers.Independent(frame, max_samples=5),
        algorithm="permutation",
    )
    return explainer(frame.iloc[:rows], max_evals=2 * FEATURES + 1)


def test_every_explained_row_is_reported():
    seen: list[tuple[int, int]] = []
    with shap_progress.report_progress(lambda done, total: seen.append((done, total))):
        _explain()

    assert [done for done, _ in seen] == list(range(1, ROWS + 1))
    assert {total for _, total in seen} == {ROWS}


def test_progress_is_not_reported_outside_the_context():
    seen: list[tuple[int, int]] = []
    with shap_progress.report_progress(lambda done, total: seen.append((done, total))):
        pass
    _explain()

    assert seen == []


def test_a_failing_sink_does_not_break_the_explanation():
    """Progress reporting is a side channel; a broken one must not lose a run."""

    def explode(done: int, total: int) -> None:
        raise RuntimeError("sink is down")

    with shap_progress.report_progress(explode):
        values = _explain()

    assert values.values.shape == (ROWS, FEATURES)
