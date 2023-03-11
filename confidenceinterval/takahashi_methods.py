from typing import List, Callable
from functools import partial
import numpy as np
from scipy.stats import norm
from confidenceinterval.utils import get_positive_negative_counts
from confidenceinterval.bootstrap import bootstrap_ci, bootstrap_methods


def precision_score_takahashi(y_true: List,
                              y_pred: List,
                              confidence_level: int = 0.95,
                              average='micro',
                              compute_ci=True):

    # TBD, do the math for macro precision based on the paper
    assert average in ['micro']

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)

    if average == 'micro':
        precision = TP.sum() / len(y_pred)
        if compute_ci:
            variance = precision * (1 - precision) / len(y_pred)
            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = precision - z * \
                np.sqrt(variance), precision + z * np.sqrt(variance)
            return precision, ci
        else:
            return precision


def precision_score_bootstrap(y_true: List,
                              y_pred: List,
                              confidence_level: int = 0.95,
                              average='micro',
                              method: str = 'bootstrap_bca',
                              n_resamples=9999,
                              random_state=None):
    precision_score_no_ci = partial(
        precision_score_takahashi,
        average=average,
        compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=precision_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def precision_score(y_true: List,
                    y_pred: List,
                    confidence_level: int = 0.95,
                    average='micro',
                    method: str = 'takahashi',
                    *args,
                    **kwargs):
    if method in bootstrap_methods:
        return precision_score_bootstrap(
            y_true, y_pred, confidence_level, average, method, *args, **kwargs)
    else:
        return precision_score_takahashi(
            y_true, y_pred, confidence_level, average, *args, **kwargs)


def recall_score_takahashi(y_true: List,
                           y_pred: List,
                           confidence_level: int = 0.95,
                           average='micro',
                           compute_ci=True):

    # TBD, do the math for macro recall based on the paper
    assert average in ['micro']

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)

    if average == 'micro':
        recall = TP.sum() / len(y_pred)
        if compute_ci:
            variance = recall * (1 - recall) / len(y_pred)
            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = recall - z * np.sqrt(variance), recall + z * np.sqrt(variance)
            return recall, ci
        else:
            return recall


def recall_score_bootstrap(y_true: List,
                           y_pred: List,
                           confidence_level: int = 0.95,
                           average='micro',
                           method: str = 'bootstrap_bca',
                           n_resamples=9999,
                           random_state=None):
    recall_score_no_ci = partial(
        recall_score_takahashi,
        average=average,
        compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=recall_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def recall_score(y_true: List,
                 y_pred: List,
                 confidence_level: int = 0.95,
                 average='micro',
                 method: str = 'takahashi',
                 *args,
                 **kwargs):
    if method in bootstrap_methods:
        return recall_score_bootstrap(
            y_true, y_pred, confidence_level, average, method, *args, **kwargs)
    else:
        return recall_score_takahashi(
            y_true, y_pred, confidence_level, average, *args, **kwargs)


def macro_f1_score_takahashi(y_true: List,
                             y_pred: List,
                             confidence_level: int = 0.95,
                             compute_ci=True):
    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    f1i = 2 * TP / (TP + FP + TP + FN + 1e-7)
    f1 = f1i.mean()
    if compute_ci:
        # Breaking down the equation for Var(MACROF1) in 3.2
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1

        a = (TP + FP + TP + FN - 2 * TP)
        b = (TP + FP + TP + FN + 1e-7)
        first_term = f1i * a / (b**2) + (f1i / 2 + a / b)

        second_term = 0
        for i in range(CM.shape[0]):
            for j in range(CM.shape[1]):
                if i == j:
                    continue
                second_term += CM[j][i] * f1i[i] * f1i[j] / (b[i] + b[j])

        variance = 2 * (first_term.sum() + second_term) / (len(f1i) ** 2)
        variance = variance / len(y_true)

        alpha = 1 - confidence_level
        z = norm.ppf(1 - alpha / 2)
        ci = f1 - z * np.sqrt(variance), f1 + z * np.sqrt(variance)

        return f1, ci
    else:
        return f1


def macro_f1_score_bootstrap(y_true: List,
                             y_pred: List,
                             confidence_level: int = 0.95,
                             average='micro',
                             method: str = 'bootstrap_bca',
                             n_resamples=9999,
                             random_state=None):
    f1_score_no_ci = partial(macro_f1_score_takahashi, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=f1_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def f1_score(y_true: List,
             y_pred: List,
             confidence_level: int = 0.95,
             average='micro',
             method: str = 'takahashi',
             *args,
             **kwargs):
    assert average in ['micro', 'macro']

    if method in bootstrap_methods:
        if average == 'macro':
            return macro_f1_score_bootstrap(
                y_true, y_pred, confidence_level, average, method, *args, **kwargs)
        else:
            # micro precision, recall and f1 are all the same
            return precision_score_bootstrap(
                y_true, y_pred, confidence_level, average, method, *args, **kwargs)
    else:
        if average == 'macro':
            return macro_f1_score_takahashi(
                y_true, y_pred, confidence_level, average, *args, **kwargs)
        else:
            # micro precision, recall and f1 are all the same
            return precision_score_takahashi(
                y_true, y_pred, confidence_level, average, *args, **kwargs)
