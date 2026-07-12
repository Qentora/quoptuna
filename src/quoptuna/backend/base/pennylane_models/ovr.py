"""One-vs-rest wrapping for binary-by-construction quantum models.

The variational quantum classifiers predict via the sign of an observable's
expectation value and hard-require labels in {-1, +1}. sklearn's
``OneVsRestClassifier`` fits one sub-estimator per class on 0/1 indicator
labels (via ``LabelBinarizer``), which those models reject. The adapter below
translates between the two conventions so any {-1,+1} binary estimator can be
used as an OvR sub-estimator unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.multiclass import OneVsRestClassifier


class PlusMinusAdapter(BaseEstimator, ClassifierMixin):
    """Lets OneVsRestClassifier's 0/1 sub-labels drive a {-1,+1} estimator."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # [0, 1] from LabelBinarizer
        y_pm = np.where(np.asarray(y).ravel() == 1, 1, -1)
        self.estimator_ = clone(self.estimator)
        try:
            self.estimator_.fit(X, y_pm)
            self.converged_ = True
        except ConvergenceWarning:
            # The training loop hit max_steps without meeting the flat-loss
            # criterion; the partially-trained parameters are kept on the model.
            # Swallowing it here matters: raising would abort the remaining
            # OvR sub-fits, discarding every other class's training.
            self.converged_ = False
        return self

    def predict_proba(self, X):
        # Inner columns are ordered [P(-1), P(+1)], which matches [0, 1].
        return np.asarray(self.estimator_.predict_proba(X))

    def decision_function(self, X):
        p = np.asarray(self.estimator_.predict_proba(X))
        return p[:, 1] - p[:, 0]

    def predict(self, X):
        return (np.asarray(self.estimator_.predict(X)) == 1).astype(int)


class ConvergenceAwareOvR(OneVsRestClassifier):
    """OneVsRestClassifier exposing an aggregated convergence flag.

    Each PlusMinusAdapter swallows its sub-fit's ConvergenceWarning (raising
    would abort the sibling class fits); ``converged_`` surfaces whether ALL
    sub-models met the convergence criterion so trial bookkeeping stays honest.
    """

    @property
    def converged_(self) -> bool:
        estimators = getattr(self, "estimators_", None)
        if not estimators:
            return True
        return all(getattr(e, "converged_", True) for e in estimators)


def wrap_one_vs_rest(model) -> OneVsRestClassifier:
    """OvR-wrap a {-1,+1} binary estimator for a K>2 target.

    ``n_jobs=1``: the JAX-trained models are not safe to fit in parallel
    processes. Training knobs (max_steps, ...) must be set on ``model``
    before wrapping — the OvR object deliberately does not proxy them, which
    also keeps the optimizer's pruning callback (gated on ``max_steps``)
    from attaching to a wrapper whose K interleaved loss series would not be
    comparable.
    """
    return ConvergenceAwareOvR(PlusMinusAdapter(model), n_jobs=1)
