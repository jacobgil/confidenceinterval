import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Callable


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
    return FP, FN, TP, TN, cm
