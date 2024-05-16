""" Classification report similar to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
"""

from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from confidenceinterval.takahashi_methods import  precision_score,recall_score, f1_score

def round_tuple(t, decimals=3):
    return tuple(round(num, decimals) for num in t)

def classification_report_with_ci(y_true: List[int], y_pred: List[int], 
                                  binary_method: str = 'wilson',
                                  round_ndigits: int = 3,
                                  numerical_to_label_map: Optional[Dict[int, str]] = None, 
                                  confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Parameters
    ----------
    y_true : List[int]
        The ground truth labels.
    y_pred : List[int]
        The predicted categories.
    binary_method: str = 'wilson'
        The method to calculate the CI for binary proportions.
    round_ndigits: int = 3
        Number of digits to return after the decimal point.
    numerical_to_label_map: Optional[Dict[int, str]]
        Mapping from class indices to descriptive names.
    confidence_level: float, optional
        The confidence level, by default 0.95

    Returns
    -------
    pd.DataFrame
        A DataFrame containing precision, recall, F1-score, and their confidence intervals for each class,
        as well as micro and macro averages.
    """

    # Unique classes in the dataset
    classes = np.unique(y_true)

    # Validate that all unique classes are covered in the numerical_to_label_map if provided
    if numerical_to_label_map is not None:
        missing_labels = [cls for cls in classes if cls not in numerical_to_label_map]
        if missing_labels:
            raise ValueError(f'Missing labels for classes: {missing_labels}')

    data = []  # List to store row dictionaries

    # Unique classes in the dataset
    classes = np.unique(y_true)

    # Calculate precision, recall, f1 for each class treated as binary
    for class_ in classes:
        y_true_binary = [1 if y == class_ else 0 for y in y_true]
        y_pred_binary = [1 if y == class_ else 0 for y in y_pred]

        # Calculate metrics
        precision, precision_ci = precision_score(y_true_binary, y_pred_binary, average='binary', method=binary_method)
        recall, recall_ci = recall_score(y_true_binary, y_pred_binary, average='binary', method=binary_method)
        binary_f1, binary_f1_ci = f1_score(y_true_binary, y_pred_binary, confidence_level=confidence_level, average='binary')

        class_name = numerical_to_label_map[class_] if (
                    numerical_to_label_map and class_ in numerical_to_label_map) else f'Class {class_}'
        support = sum(y_true_binary)

        # Create a new row as a DataFrame and append it to the main DataFrame
        # Append new row to the list
        data.append({
            'Class': class_name,
            'Precision': round(precision, round_ndigits),
            'Recall': round(recall, round_ndigits),
            'F1-Score': round(binary_f1, round_ndigits),
            'Precision CI': round_tuple(precision_ci, round_ndigits),
            'Recall CI': round_tuple(recall_ci, round_ndigits),
            'F1-Score CI': round_tuple(binary_f1_ci, round_ndigits),
            'Support': support
        })

    precision_micro, p_ci_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro, p_ci_macro = precision_score(y_true, y_pred, average='macro')

    recall_micro, r_ci_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro, r_ci_macro = recall_score(y_true, y_pred, average='macro')

    f1_micro, f1_ci_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro, f1_ci_macro = f1_score(y_true, y_pred, average='macro')

    data.append({
        'Class': 'micro',
        'Precision': round(precision_micro, round_ndigits),
        'Recall': round(recall_micro, round_ndigits),
        'F1-Score': round(f1_micro, round_ndigits),
        'Precision CI': round_tuple(p_ci_micro, round_ndigits),
        'Recall CI': round_tuple(r_ci_micro, round_ndigits),
        'F1-Score CI': round_tuple(f1_ci_micro, round_ndigits),
        'Support': len(y_true)
    })

    data.append({
        'Class': 'macro',
        'Precision': round(precision_macro, round_ndigits),
        'Recall': round(recall_macro, round_ndigits),
        'F1-Score': round(f1_macro, round_ndigits),
        'Precision CI': round_tuple(p_ci_macro, decimals=round_ndigits),
        'Recall CI': round_tuple(r_ci_macro, decimals=round_ndigits),
        'F1-Score CI': round_tuple(f1_ci_macro, decimals=round_ndigits),
        'Support': len(y_true)

    })

    df = pd.DataFrame(data)

    return df