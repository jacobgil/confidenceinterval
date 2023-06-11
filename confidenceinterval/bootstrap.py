from scipy.stats import bootstrap
import numpy as np
from typing import List, Callable, Optional, Tuple

bootstrap_methods = [
    'bootstrap_bca',
    'bootstrap_percentile',
    'bootstrap_basic']


class BootstrapParams:
    n_resamples: int
    random_state: Optional[np.random.RandomState]


def bootstrap_ci(y_true: List[int],
                 y_pred: List[int],
                 metric: Callable,
                 confidence_level: float = 0.95,
                 n_resamples: int = 9999,
                 method: str = 'bootstrap_bca',
                 random_state: Optional[np.random.RandomState] = None) -> Tuple[float, Tuple[float, float]]:

    def statistic(*indices):
        indices = np.array(indices)[0, :]
        return metric(np.array(y_true)[indices], np.array(y_pred)[indices])

    assert method in bootstrap_methods, f'Bootstrap ci method {method} not in {bootstrap_methods}'

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
