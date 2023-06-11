import pytest
from confidenceinterval import accuracy_score, \
    ppv_score, \
    npv_score, \
    tpr_score, \
    fpr_score, \
    tnr_score, \
    precision_score, \
    recall_score, \
    f1_score, \
    roc_auc_score

import sklearn.metrics
import numpy as np


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0])])
def test_accuracy(data):
    y_true, y_pred = data
    sklearn_result = sklearn.metrics.accuracy_score(y_true, y_pred)
    accuracy, ci = accuracy_score(y_true, y_pred)
    assert sklearn_result == accuracy


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0])])
def test_micro_precision(data):
    y_true, y_pred = data
    sklearn_result = sklearn.metrics.precision_score(
        y_true, y_pred, average='micro')
    precision, ci = precision_score(y_true, y_pred, average='micro')
    assert sklearn_result == precision


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0])])
@pytest.mark.parametrize("metric",
                         [accuracy_score, ppv_score, npv_score, tpr_score, fpr_score, tnr_score, precision_score, recall_score, roc_auc_score, f1_score])
def test_run_metrics(data, metric):
    y_true, y_pred = data
    result, ci = metric(y_true, y_pred)


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0])])
@pytest.mark.parametrize("metric",
                         [accuracy_score, ppv_score, npv_score, tpr_score, fpr_score, tnr_score, precision_score, f1_score])
def test_run_metrics_bootstrap(data, metric):
    y_true, y_pred = data
    result, ci = metric(
        y_true, y_pred, method="bootstrap_bca", n_resamples=100)


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0],
                           [0, 0, 1, 0, 1, 0, 1, 1, 0])])
def test_macro_f1(data):
    y_true, y_pred = data
    sklearn_result = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    f1, ci = f1_score(y_true, y_pred, average='macro')
    assert pytest.approx(sklearn_result, 0.01) == f1


def test_auc():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0] * 10)
    y_pred = y_true + np.random.randn(len(y_true)) * 1
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.minimum(y_pred, 1)

    auc, ci = roc_auc_score(y_true, y_pred, method='delong')
    auc_bootstrap, ci_bootstrap = roc_auc_score(
        y_true, y_pred, method='bootstrap_bca')
    assert pytest.approx(auc, 0.01) == auc_bootstrap


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0] * 30,
                           [0, 0, 1, 0, 1, 0, 1, 1, 0] * 30)])
def test_binary_f1(data):
    y_true, y_pred = data
    sklearn_result = sklearn.metrics.f1_score(y_true, y_pred)
    f1, ci = f1_score(y_true, y_pred, average='binary')
    assert pytest.approx(sklearn_result, 0.01) == f1


@pytest.mark.parametrize("data",
                         [([0, 0, 0, 0, 1, 1, 1, 1, 0] * 30,
                           [0, 0, 1, 0, 1, 0, 1, 1, 0] * 30)])
def test_macro_recall(data):
    y_true, y_pred = data
    sklearn_result = sklearn.metrics.recall_score(
        y_true, y_pred, average='macro')
    recall, ci = recall_score(y_true, y_pred, average='macro')

    assert pytest.approx(sklearn_result, 0.01) == recall


@pytest.mark.parametrize("data",
                         [([1, 1, 0, 0, 1, 1, 1, 1, 0] * 30,
                           [0, 0, 1, 0, 1, 0, 1, 1, 0] * 30)])
def test_macro_precision(data):
    y_true, y_pred = data
    sklearn_result = sklearn.metrics.precision_score(
        y_true, y_pred, average='macro')
    precision, ci = precision_score(y_true, y_pred, average='macro')

    assert pytest.approx(sklearn_result, 0.01) == precision
