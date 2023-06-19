""" Confidence intervals for the F1, precision and recall scores, based on the Takahashi paper.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1
"""

from typing import List, Callable, Tuple, Union, Optional

from functools import partial
import numpy as np
from scipy.stats import norm
from confidenceinterval.utils import get_positive_negative_counts
from confidenceinterval.bootstrap import bootstrap_ci, bootstrap_methods, BootstrapParams
from confidenceinterval.binary_metrics import tpr_score, ppv_score

from sklearn.metrics import confusion_matrix


def precision_score_takahashi(y_true: List[int],
                              y_pred: List[int],
                              confidence_level: float = 0.95,
                              average: str = 'micro',
                              compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the precision score according to the takahasi paper.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1

    The macro case isn't derived in the paper, so instead we derive it ourselves

    The derivation is based on 3 parts:
    1. Assuming the confusion matrix p distribution is a multinomial distribution,
    2. Use the delta method to compute the variance of
    the precision
    n * variance(f1) = (∂Precision/∂p)T * [diag(p) - ppT] * (∂∂Precision/∂p)
    3. To compute ∂∂Precision/∂p:
    ∂Precision / ∂p_ii = (1 - Precision]) / categories
    ∂Precision / ∂p_ji = -(∂Precision_i) / (#categories * ∑k (p_ki) )

    p_ij are the fraction of the detections from category i that were classified as category j



    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level, by default 0.95
    average : str, optional
        The avarging method accross multiple categories., by default 'micro'.
        Currently only 'micro' is supported, 'macro' to be implemented.

    compute_ci : bool, optional
        If tue, the confidence level will be computed as well, otherwise only the metric.
        By default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """
    # TBD, do the math for macro precision based on the paper
    assert average in ['micro', 'macro']

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
    elif average == 'macro':
        p = confusion_matrix(y_true, y_pred)
        p = p / p.sum()

        # The [recosopm] per class
        total_detected_as_each_category = p.sum(axis=0)
        P_i = np.diag(p / total_detected_as_each_category)
        precision_macro = P_i.sum() / len(p)

        if compute_ci:
            # The variance
            derivative = np.diag((1 - P_i) / len(p))
            for i in range(len(p)):
                indices = [j for j in range(len(p)) if i != j]
                derivative[indices, i] -= P_i[i] / \
                    (len(p) * total_detected_as_each_category[i])
            p = p.flatten()
            derivative = derivative.flatten()
            variance = np.diag(p) - p * p.transpose()

            delta_method_variance = derivative.transpose().dot(variance).dot(derivative)
            delta_method_variance = delta_method_variance / len(y_true)

            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = precision_macro - z * \
                np.sqrt(delta_method_variance), precision_macro + \
                z * np.sqrt(delta_method_variance)

            return precision_macro, ci
        else:
            return precision_macro

    else:
        raise NotImplementedError(
            "Only micro averaging is currently supported for multi-class precision score.")


def precision_score_bootstrap(y_true: List[int],
                              y_pred: List[int],
                              confidence_level: float = 0.95,
                              average: str = 'micro',
                              method: str = 'bootstrap_bca',
                              n_resamples: int = 9999,
                              random_state: Optional[np.random.RandomState] = None) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """The bootstrap for the precision score.

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level, by default 0.95
    average : str, optional
        The avarging method accross multiple categories., by default 'micro'.
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducability, by default None

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """
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


def precision_score(y_true: List[int],
                    y_pred: List[int],
                    confidence_level: float = 0.95,
                    average: str = 'micro',
                    method: str = 'takahashi',
                    compute_ci: bool = True,
                    **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Return the precision score. Supports micro/binary precision methods.

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level , by default 0.95
    average : str, optional
        The avarging method accross multiple categories., by default 'micro'.
        Supports binary / micro
    method : str, optional
        The method for the confidence interval. If one of the bootstrap methods, bootstrap will be used.
        The values supported are 'takahashi', 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'.
    compute_ci : bool, optional
        If tue, the confidence level will be computed as well, otherwise only the metric.
        By default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """

    if average == 'binary':
        return ppv_score(y_true, y_pred, confidence_level, method, **kwargs)

    if method in bootstrap_methods:
        return precision_score_bootstrap(
            y_true, y_pred, confidence_level, average, method, **kwargs)
    else:
        return precision_score_takahashi(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, average=average, compute_ci=compute_ci)


def recall_score_takahashi(y_true: List[int],
                           y_pred: List[int],
                           confidence_level: float = 0.95,
                           average: str = 'micro',
                           compute_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the precision score according to the takahasi paper.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1

    The macro case isn't derived in the paper, so instead we derive it ourselves

    The derivation is based on 3 parts:
    1. Assuming the confusion matrix p distribution is a multinomial distribution,
    2. Use the delta method to compute the variance of
    the recall
    n * variance(f1) = (∂Recall/∂p)T * [diag(p) - ppT] * (∂Recall/∂p)
    3. To compute ∂Recall/∂p:
    ∂Recall / ∂p_ii = (1 - Recall_i) / categories
    ∂Recall / ∂p_ij = -(Recall_i) / (#categories * ∑j (p_ij) )

    p_ij are the fraction of the detections from category i that were classified as category j

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level, by default 0.95
    average : str, optional
        The avarging method accross multiple categories., by default 'micro'.
        Currently only 'micro' is supported, 'macro' to be implemented.
    compute_ci : bool, optional
        If tue, the confidence level will be computed as well, otherwise only the metric.
        By default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """

    # TBD, do the math for macro recall based on the paper
    assert average in ['micro', 'macro']

    if average == 'micro':
        FP, FN, TP, TN, CM = get_positive_negative_counts(y_true, y_pred)
        recall = TP.sum() / len(y_pred)
        if compute_ci:
            variance = recall * (1 - recall) / len(y_pred)
            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = recall - z * np.sqrt(variance), recall + z * np.sqrt(variance)
            return recall, ci
        else:
            return recall
    elif average == 'macro':
        p = confusion_matrix(y_true, y_pred)

        p = p / p.sum()

        # The recall per class
        total_from_each_category = p.sum(axis=-1)
        R_i = np.diag(p / total_from_each_category)
        recall_macro = R_i.sum() / len(p)

        if compute_ci:
            # The variance
            derivative = np.diag((1 - R_i) / len(p))
            for i in range(len(p)):
                indices = [j for j in range(len(p)) if i != j]
                derivative[i, indices] -= R_i[i] / \
                    (len(p) * total_from_each_category[i])
            p = p.flatten()
            derivative = derivative.flatten()
            variance = np.diag(p) - p * p.transpose()

            delta_method_variance = derivative.transpose().dot(variance).dot(derivative)
            delta_method_variance = delta_method_variance / len(y_true)

            alpha = 1 - confidence_level
            z = norm.ppf(1 - alpha / 2)
            ci = recall_macro - z * \
                np.sqrt(delta_method_variance), recall_macro + \
                z * np.sqrt(delta_method_variance)

            return recall_macro, ci
        else:
            return recall_macro
    else:
        raise NotImplementedError(
            "Only micro averaging is currently supported for the multi-class recall score.")


def recall_score_bootstrap(y_true: List[int],
                           y_pred: List[int],
                           confidence_level: float = 0.95,
                           average: str = 'micro',
                           method: str = 'bootstrap_bca',
                           n_resamples: int = 9999,
                           random_state: Optional[np.random.RandomState] = None) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """The bootstrap for the recall score.

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level, by default 0.95
    average : str, optional
        The avarging method accross multiple categories., by default 'micro'.
        Currently only 'micro' is supported, 'macro' to be implemented.
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducability, by default None

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """
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


def recall_score(y_true: List[int],
                 y_pred: List[int],
                 confidence_level: float = 0.95,
                 average: str = 'micro',
                 method: str = 'takahashi',
                 compute_ci: bool = True,
                 **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Return the precision score. Supports micro/binary precision methods.

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level , by default 0.95
    average : str, optional
        The avarging method accross multiple categories., by default 'micro'.
        Supports binary / micro
    method : str, optional
        The method for the confidence interval. If one of the bootstrap methods, bootstrap will be used.
        The values supported are 'takahashi', 'bootstrap_bca', 'bootstrap_percentile', 'bootstrap_basic'.
    compute_ci : bool, optional
        If tue, the confidence level will be computed as well, otherwise only the metric.
        By default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """
    if average == 'binary':
        return tpr_score(y_true, y_pred, confidence_level, method, **kwargs)

    if method in bootstrap_methods:
        return recall_score_bootstrap(
            y_true, y_pred, confidence_level, average, method, **kwargs)
    else:
        return recall_score_takahashi(
            y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, average=average, compute_ci=compute_ci)


def binary_f1_score_takahashi(y_true: List[int],
                              y_pred: List[int],
                              confidence_level: float = 0.95,
                              compute_ci=True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the binary f1 and it's confidence interval in the spirit of the takahashi paper.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1
    The paper covers macro/micro F1, however does not cover the binary case.
    The paper uses this technique for deriving the variances of different metrics:
    1. Assuming the confusion matrix p distribution is a multinomial distribution,
    2. Use the delta method to compute the variance of
    the F1 score for the binary case :
    n * variance(f1) = derivative(p)T * [diag(p) - ppT] * derivative(p)

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    compute_ci : bool, optional
        If tue, the confidence level will be computed as well, otherwise only the metric.
        By default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """

    p = confusion_matrix(y_true, y_pred)
    p = p / p.sum()
    denom = p[0][1] + p[1][0] + 2 * p[1][1]
    f1 = 2 * p[1][1] / denom

    if compute_ci:
        derivative = np.array(
            [0, -f1 / denom, -f1 / denom, 2 * (1 - f1) / denom])
        derivative = np.float32(derivative)
        p = p.flatten()
        variance = np.diag(p) - p * p.transpose()
        delta_method_variance = derivative.transpose().dot(variance).dot(derivative)
        delta_method_variance = delta_method_variance / len(y_true)

        alpha = 1 - confidence_level
        z = norm.ppf(1 - alpha / 2)
        ci = f1 - z * np.sqrt(delta_method_variance), f1 + \
            z * np.sqrt(delta_method_variance)
        return f1, ci
    return f1


def macro_f1_score_takahashi(y_true: List[int],
                             y_pred: List[int],
                             confidence_level: float = 0.95,
                             compute_ci=True) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the Macro F1 score using the Takahashi method:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP1

    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence interval level, by default 0.95
    compute_ci : bool, optional
        If tue, the confidence level will be computed as well, otherwise only the metric.
        By default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The metric, and (optionally) the confidence interval low, high tuple.
    """
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


def binary_f1_score_bootstrap(y_true: List[int],
                              y_pred: List[int],
                              confidence_level: float = 0.95,
                              method: str = 'bootstrap_bca',
                              n_resamples: int = 9999,
                              random_state: Optional[np.random.RandomState] = None) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the binary f1 score with bootstrapping.
    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducability, by default None

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The binary F1 score, and (optionally) the confidence interval low, high tuple.
    """

    f1_score_no_ci = partial(binary_f1_score_takahashi, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=f1_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def macro_f1_score_bootstrap(y_true: List[int],
                             y_pred: List[int],
                             confidence_level: float = 0.95,
                             method: str = 'bootstrap_bca',
                             n_resamples: int = 9999,
                             random_state: Optional[np.random.RandomState] = None) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the macro f1 score with bootstrapping.
    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    confidence_level : float, optional
        The confidence level, by default 0.95
    method : str, optional
        The bootstrap method, by default 'bootstrap_bca'
    n_resamples : int, optional
        The number of bootstrap resamples, by default 9999
    random_state : Optional[np.random.RandomState], optional
        The random state for reproducability, by default None

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The F1 macro score, and (optionally) the confidence interval low, high tuple.
    """

    f1_score_no_ci = partial(macro_f1_score_takahashi, compute_ci=False)
    return bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=f1_score_no_ci,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        method=method,
                        random_state=random_state)


def f1_score(y_true: List[int],
             y_pred: List[int],
             confidence_level: float = 0.95,
             average: str = 'micro',
             method: str = 'takahashi',
             compute_ci: bool = True,
             **kwargs) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """Compute the F1 score and optionally the confidence interval.
    For non bootstrapping methods, the micro and macro F1 averaging modes, this following the method described in the paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/
    For the binary F1 averaging mode, the confidence interval is computed using a derivation made by us in the spirit of the paper above,
    using the delta method and the derivatives of F1 score with respect to the 2x2 confusion matrix parameters.

    Parameters
    ----------
    y_true : List[int]
        _description_
    y_pred : List[int]
        _description_
    confidence_level : float, optional
        _description_, by default 0.95
    average : str, optional
        _description_, by default 'micro'
    method : str, optional
        _description_, by default 'takahashi'
    compute_ci : bool, optional
        _description_, by default True

    Returns
    -------
    Union[float, Tuple[float, Tuple[float, float]]]
        The F1 score, and (optionally) the confidence interval low, high tuple.
    """

    assert average in ['micro', 'macro',
                       'binary'], 'average method {average} not supported'

    if method in bootstrap_methods:
        if average == 'binary':
            return binary_f1_score_bootstrap(
                y_true=y_true,
                y_pred=y_pred,
                confidence_level=confidence_level,
                method=method,
                **kwargs)
        elif average == 'macro':
            return macro_f1_score_bootstrap(
                y_true=y_true,
                y_pred=y_pred,
                confidence_level=confidence_level,
                method=method,
                **kwargs)
        elif average == 'micro':
            # micro precision, recall and f1 are all the same
            return precision_score_bootstrap(
                y_true=y_true, y_pred=y_pred, confidence_level=confidence_level, average=average, method=method, **kwargs)
        else:
            raise NotImplementedError(
                f'average method {average} not supported')
    else:
        if average == 'binary':
            return binary_f1_score_takahashi(y_true=y_true,
                                             y_pred=y_pred,
                                             confidence_level=confidence_level,
                                             compute_ci=compute_ci)
        elif average == 'macro':
            return macro_f1_score_takahashi(
                y_true=y_true,
                y_pred=y_pred,
                confidence_level=confidence_level,
                compute_ci=compute_ci)
        elif average == 'micro':
            # micro precision, recall and f1 are all the same
            return precision_score_takahashi(
                y_true=y_true,
                y_pred=y_pred,
                confidence_level=confidence_level,
                average=average,
                compute_ci=compute_ci)
        else:
            raise NotImplementedError(
                f'average method {average} not supported')
