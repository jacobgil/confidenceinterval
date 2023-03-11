import statsmodels
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix
from scipy.stats import bootstrap
from typing import List, Callable
from functools import partial
import numpy as np
from scipy.stats import norm

proportion_conf_methods = [
    'wilson',
    'normal',
    'agresti_coull',
    'beta',
    'jeffreys',
    'binom_test']
boostrap_conf_methods = [
    'bootstrap_bca',
    'bootstrap_percentile',
    'bootstrap_basic']
precision_recall_f1_methods = ['takahashi']


def get_positive_negative_counts(y_true: List,
                                 y_pred: List):
    """ Return the False positives, false negatives, true positives, true negatives.
        Taken from
        https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
        answered by lucidv01d
    """

    max_category = max(np.max(y_true), np.max(y_pred))
    # There are always at least two categories:
    if max_category == 0:
        max_category = 1

    cm = np.array(confusion_matrix(y_true, y_pred,
                  labels=np.arange(0, 1 + max_category)))
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return FP, FN, TP, TN


def bootstrap_ci(y_true: List,
                 y_pred: List,
                 metric: Callable,
                 confidence_level: int = 0.95,
                 n_resamples=9999,
                 method='bootstrap_bca',
                 random_state=None):

    def statistic(*indices):
        indices = np.array(indices)[0, :]
        return metric(np.array(y_true)[indices], np.array(y_pred)[indices])

    assert method in boostrap_conf_methods, f'Bootstrap ci method {method} not in {boostrap_conf_methods}'

    indices = (np.arange(len(y_true)), )
    bootstrap_res = bootstrap(indices,
                              statistic=statistic,
                              n_resamples=n_resamples,
                              confidence_level=confidence_level,
                              method=method.split('bootstrap_')[1],
                              random_state=random_state)
    result = metric(y_true, y_pred)
    ci = bootstrap_res.confidence_interval.low, bootstrap_res.confidence_interval.high
    return result, ci


def accuracy_score_binomial_ci(y_true: List,
                               y_pred: List,
                               confidence_level: int = 0.95,
                               method: str = 'wilson',
                               compute_ci=True
                               ):
    """ Compute the accuracy score and the confidence interval.
        The confidence interval is computed as a binomial proportion,
        for more information see
        https://www.statsmodels.org/devel/generated/statsmodels.stats.proportion.proportion_confint.html

    """
    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'

    correct = np.sum(np.array(y_pred) == np.array(y_true))
    acc = correct / len(y_pred)
    if compute_ci:
        interval = proportion_confint(
            correct,
            len(y_pred),
            alpha=1 -
            confidence_level,
            method=method)
        return acc, interval
    else:
        return acc


def accuracy_score_bootstrap(y_true: List,
                             y_pred: List,
                             confidence_level: int = 0.95,
                             method: str = 'bootstrap_BCa',
                             n_resamples=9999,
                             random_state=None):
    accuracy_score_no_ci = partial(
        accuracy_score_binomial_ci,
        compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=accuracy_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def accuracy_score(y_true: List,
                   y_pred: List,
                   confidence_level: int = 0.95,
                   method: str = 'wilson',
                   *args, **kwargs):
    if method in boostrap_conf_methods:
        return accuracy_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return accuracy_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def ppv_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True):

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)
    TP, FP = TP[1], FP[1]

    result = TP / (TP + FP)
    if compute_ci:
        interval = proportion_confint(
            TP, TP + FP, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def ppv_score_bootstrap(y_true: List,
                        y_pred: List,
                        confidence_level: int = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples=9999,
                        random_state=None):
    ppv_score_no_ci = partial(ppv_score_binomial_ci, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=ppv_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def ppv_score(y_true: List,
              y_pred: List,
              confidence_level: int = 0.95,
              method: str = 'wilson',
              *args, **kwargs):
    if method in boostrap_conf_methods:
        return ppv_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return ppv_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def npv_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True):

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)
    TN, FN = TN[1], FN[1]

    result = TN / (TN + FN)
    if compute_ci:
        interval = proportion_confint(
            TN, TN + FN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def npv_score_bootstrap(y_true: List,
                        y_pred: List,
                        confidence_level: int = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples=9999,
                        random_state=None):
    npv_score_no_ci = partial(npv_score_binomial_ci, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=npv_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def npv_score(y_true: List,
              y_pred: List,
              confidence_level: int = 0.95,
              method: str = 'wilson',
              *args, **kwargs):
    if method in boostrap_conf_methods:
        return npv_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return npv_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def tpr_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True):

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)
    TP, FN = TP[1], FN[1]

    result = TP / (TP + FN)
    if compute_ci:
        interval = proportion_confint(
            TP, TP + FN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def tpr_score_bootstrap(y_true: List,
                        y_pred: List,
                        confidence_level: int = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples=9999,
                        random_state=None):
    tpr_score_no_ci = partial(tpr_score_binomial_ci, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=tpr_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def tpr_score(y_true: List,
              y_pred: List,
              confidence_level: int = 0.95,
              method: str = 'wilson',
              *args, **kwargs):
    if method in boostrap_conf_methods:
        return tpr_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return tpr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def fpr_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True):

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)
    FP, TN = FP[1], TN[1]

    result = FP / (FP + TN)
    if compute_ci:
        interval = proportion_confint(
            FP, FP + TN, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def fpr_score_bootstrap(y_true: List,
                        y_pred: List,
                        confidence_level: int = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples=9999,
                        random_state=None):
    fpr_score_no_ci = partial(fpr_score_binomial_ci, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=fpr_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def fpr_score(y_true: List,
              y_pred: List,
              confidence_level: int = 0.95,
              method: str = 'wilson',
              *args,
              **kwargs):
    if method in boostrap_conf_methods:
        return fpr_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return fpr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def tnr_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True):

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)
    TN, FP = TN[1], FP[1]

    result = TN / (TN + FP)
    if compute_ci:
        interval = proportion_confint(
            TN, TN + FP, alpha=1 - confidence_level, method=method)
        return result, interval
    else:
        return result


def tnr_score_bootstrap(y_true: List,
                        y_pred: List,
                        confidence_level: int = 0.95,
                        method: str = 'bootstrap_bca',
                        n_resamples=9999,
                        random_state=None):
    tnr_score_no_ci = partial(tnr_score_binomial_ci, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=tnr_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def tnr_score(y_true: List,
              y_pred: List,
              confidence_level: int = 0.95,
              method: str = 'wilson',
              *args,
              **kwargs):
    if method in boostrap_conf_methods:
        return tnr_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return tnr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def precision_score_takahashi(y_true: List,
                              y_pred: List,
                              confidence_level: int = 0.95,
                              average='micro',
                              compute_ci=True):

    # TBD, do the math for macro precision based on the paper
    assert average in ['micro']

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)

    if average == 'micro':
        precision = TP.sum() / len(y_pred)
        if compute_ci:
            variance = precision * (1 - precision) / len(y_pred)
            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = precision - z * variance, precision + z * variance
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
    if method in boostrap_conf_methods:
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

    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)

    if average == 'micro':
        recall = TP.sum() / len(y_pred)
        if compute_ci:
            variance = recall * (1 - recall) / len(y_pred)
            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = recall - z * variance, recall + z * variance
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
    if method in boostrap_conf_methods:
        return recall_score_bootstrap(
            y_true, y_pred, confidence_level, average, method, *args, **kwargs)
    else:
        return recall_score_takahashi(
            y_true, y_pred, confidence_level, average, *args, **kwargs)


def macro_f1_score_takahashi(y_true: List,
                             y_pred: List,
                             confidence_level: int = 0.95,
                             compute_ci=True):
    FP, FN, TP, TN = get_positive_negative_counts(y_true, y_pred)
    f1i = 2 * TP / (TP + FP + TP + FN)
    f1 = f1i.mean()
    if compute_ci:
        return f1, [0, 1]
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

    if method in boostrap_conf_methods:
        if average == 'macro':
            return macro_f1_score_bootstrap(
                y_true, y_pred, confidence_level, average, method, *args, **kwargs)
    else:
        if average == 'macro':
            return macro_f1_score_takahashi(
                y_true, y_pred, confidence_level, average, *args, **kwargs)
