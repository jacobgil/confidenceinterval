from ast import Call
import statsmodels
from statsmodels.stats.proportion import proportion_confint
from typing import List, Callable, Tuple, Union
from functools import partial
import numpy as np
from confidenceinterval.utils import get_positive_negative_counts
from confidenceinterval.bootstrap import bootstrap_ci, bootstrap_methods

proportion_conf_methods = [
    'wilson',
    'normal',
    'agresti_coull',
    'beta',
    'jeffreys',
    'binom_test']


def accuracy_score_binomial_ci(y_true: List,
                               y_pred: List,
                               confidence_level: int = 0.95,
                               method: str = 'wilson',
                               compute_ci=True
                               ) -> Union[float, Tuple[float, float]]:
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
                             method: str = 'bootstrap_bca',
                             n_resamples: int = 9999,
                             random_state: Callable = None) -> Tuple[float, float]:
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
                   *args, **kwargs) -> Union[float, Tuple[float, float]]:
    if method in bootstrap_methods:
        return accuracy_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return accuracy_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def ppv_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, float]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FP = TP[1], FP[1]

    result = TP / (TP + FP + 1e-7)
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
                        n_resamples: int = 9999,
                        random_state: Callable = None) -> Tuple[float, float]:
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
              *args, **kwargs) -> Union[float, Tuple[float, float]]:
    if method in bootstrap_methods:
        return ppv_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return ppv_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def npv_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, float]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TN, FN = TN[1], FN[1]

    result = TN / (TN + FN + 1e-7)
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
                        n_resamples: int = 9999,
                        random_state: Callable = None) -> Tuple[float, float]:
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
              *args, **kwargs) -> Union[float, Tuple[float, float]]:
    if method in bootstrap_methods:
        return npv_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return npv_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def tpr_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, float]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TP, FN = TP[1], FN[1]

    result = TP / (TP + FN + 1e-7)
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
                        n_resamples: int = 9999,
                        random_state: Callable = None) -> Tuple[float, float]:
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
              *args,
              **kwargs) -> Union[float, Tuple[float, float]]:
    if method in bootstrap_methods:
        return tpr_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return tpr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def fpr_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, float]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    FP, TN = FP[1], TN[1]

    result = FP / (FP + TN + 1e-7)
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
                        n_resamples: int = 9999,
                        random_state: Callable = None) -> Tuple[float, float]:
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
              **kwargs) -> Union[float, Tuple[float, float]]:
    if method in bootstrap_methods:
        return fpr_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return fpr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)


def tnr_score_binomial_ci(y_true: List,
                          y_pred: List,
                          confidence_level: int = 0.95,
                          method: str = 'wilson',
                          compute_ci=True) -> Union[float, Tuple[float, float]]:

    assert method in proportion_conf_methods, f'Proportion CI method {method} not in {proportion_conf_methods}'
    assert 0 <= confidence_level <= 1, f'confidence_level has to be between 0 and 1 but is {confidence_level}'
    assert np.min(y_true) >= 0 and np.max(
        y_true) <= 1, 'This metric is supported only for binary classification'
    assert np.min(y_pred) >= 0 and np.max(
        y_pred) <= 1, 'This metric is supported only for binary classification'

    FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
    TN, FP = TN[1], FP[1]

    result = TN / (TN + FP + 1e-7)
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
                        n_resamples: int = 9999,
                        random_state: Callable = None) -> Tuple[float, float]:
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
              **kwargs) -> Union[float, Tuple[float, float]]:
    if method in bootstrap_methods:
        return tnr_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
    else:
        return tnr_score_binomial_ci(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
