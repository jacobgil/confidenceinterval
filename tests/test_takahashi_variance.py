"""Delta-method variances of the Takahashi-style intervals.

Reference: the published example of Takahashi et al. (2022), Table 1, and a
numerical delta method with the multinomial covariance diag(p) - p p^T.
"""
import numpy as np
import pytest
from scipy.stats import norm

from confidenceinterval.takahashi_methods import (
    binary_f1_score_takahashi, macro_f1_score_takahashi,
    precision_score_takahashi, recall_score_takahashi)


def _labels(cm):
    y_true, y_pred = [], []
    for i, row in enumerate(cm):
        for j, c in enumerate(row):
            y_true += [i] * c
            y_pred += [j] * c
    return np.array(y_true), np.array(y_pred)


def _delta_se(metric, cm):
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    p = (cm / n).ravel()
    r = cm.shape[0]
    grad = np.empty_like(p)
    h = 1e-6
    for k in range(p.size):
        up, dn = p.copy(), p.copy()
        up[k] += h
        dn[k] -= h
        grad[k] = (metric(up.reshape(r, r)) - metric(dn.reshape(r, r))) / (2 * h)
    cov = np.diag(p) - np.outer(p, p)
    return np.sqrt(grad @ cov @ grad / n)


def _f1_binary(p):
    return 2 * p[1, 1] / (p[0, 1] + p[1, 0] + 2 * p[1, 1])


def _precision_macro(p):
    return np.mean(np.diag(p) / p.sum(axis=0))


def _recall_macro(p):
    return np.mean(np.diag(p) / p.sum(axis=1))


def _f1_macro(p):
    return np.mean(2 * np.diag(p) / (p.sum(axis=0) + p.sum(axis=1)))


def _se_from_ci(ci, level=0.95):
    return (ci[1] - ci[0]) / (2 * norm.ppf(1 - (1 - level) / 2))


def test_macro_f1_matches_takahashi_table1():
    # Takahashi et al. (2022), Table 1: rows predicted, columns true.
    paper = np.array([[2, 2, 2], [5, 70, 2], [0, 2, 15]])
    y_true, y_pred = _labels(paper.T)
    f1, ci = macro_f1_score_takahashi(y_true, y_pred)
    assert f1 == pytest.approx(0.689, abs=5e-4)
    assert _se_from_ci(ci) == pytest.approx(0.0650, abs=5e-5)
    assert ci[0] == pytest.approx(0.562, abs=5e-4)
    assert ci[1] == pytest.approx(0.817, abs=5e-4)


CM3 = [[30, 5, 5], [10, 20, 10], [2, 8, 10]]
CM2 = [[50, 10], [20, 20]]


@pytest.mark.parametrize("func, metric, kwargs, cm", [
    (binary_f1_score_takahashi, _f1_binary, {}, CM2),
    (binary_f1_score_takahashi, _f1_binary, {}, [[9, 0], [5, 138]]),
    (precision_score_takahashi, _precision_macro, {"average": "macro"}, CM3),
    (recall_score_takahashi, _recall_macro, {"average": "macro"}, CM3),
    (recall_score_takahashi, _recall_macro, {"average": "macro"}, [[641, 55], [0, 2]]),
    (macro_f1_score_takahashi, _f1_macro, {}, CM3),
    (macro_f1_score_takahashi, _f1_macro, {}, CM2),
])
def test_delta_method_se(func, metric, kwargs, cm):
    y_true, y_pred = _labels(cm)
    _, ci = func(y_true, y_pred, **kwargs)
    assert _se_from_ci(ci) == pytest.approx(_delta_se(metric, cm), rel=1e-4)
