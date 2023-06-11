import numpy as np
from typing import Callable, List, Tuple
from .delong import delong_roc_variance
import sklearn.metrics
from scipy.stats import norm

from confidenceinterval.bootstrap import bootstrap_ci, bootstrap_methods


def roc_auc_score_bootstrap(y_true: List,
                            y_pred: List,
                            confidence_level: float = 0.95,
                            method: str = 'bootstrap_bca',
                            n_resamples: int = 9999,
                            random_state: Callable = None) -> Tuple[float, float]:
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=sklearn.metrics.roc_auc_score,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def roc_auc_score(y_true: List,
                  y_pred: List,
                  confidence_level: float = 0.95,
                  method: str = 'delong',
                  *args, **kwargs) -> Tuple[float, float]:
    assert method in [
        'delong'] + bootstrap_methods, f"Method {method} not in {['delong'] + bootstrap_methods}"

    if method == 'delong':
        auc, variance = delong_roc_variance(np.array(y_true), np.array(y_pred))
        alpha = 1 - confidence_level
        z = norm.ppf(1 - alpha / 2)
        ci = auc - z * np.sqrt(variance), auc + z * np.sqrt(variance)
        return auc, ci
    elif method in bootstrap_methods:
        return roc_auc_score_bootstrap(
            y_true, y_pred, confidence_level, method, *args, **kwargs)
