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

    # Initialize DataFrame
    columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Precision CI', 'Recall CI', 'F1-Score CI', 'Support']
    df = pd.DataFrame(columns=columns)

    # Calculate precision, recall, F1 for each class treated as binary
    for class_ in classes:
        y_true_binary = [1 if y == class_ else 0 for y in y_true]
        y_pred_binary = [1 if y == class_ else 0 for y in y_pred]

        # Calculate metrics
        precision, precision_ci = precision_score(y_true_binary, y_pred_binary, average='binary', method=binary_method)
        recall, recall_ci = recall_score(y_true_binary, y_pred_binary, average='binary', method=binary_method)
        binary_f1, binary_f1_ci = f1_score(y_true_binary, y_pred_binary, confidence_level=confidence_level, average='binary')

        class_name = numerical_to_label_map[class_] if (numerical_to_label_map and class_ in numerical_to_label_map) else f'Class {class_}'
        support = sum(y_true_binary)

        # Create a new row as a DataFrame and append it to the main DataFrame
        new_row = pd.DataFrame({
            'Class': class_name,
            'Precision': [round(precision, 3)],
            'Recall': [round(recall, 3)],
            'F1-Score': [round(binary_f1, 3)],
            'Precision CI': [round_tuple(precision_ci)],
            'Recall CI': [round_tuple(recall_ci)],
            'F1-Score CI': [round_tuple(binary_f1_ci)],
            'Support': [support]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        
    # Calculate micro and macro averages and append to DataFrame
    precision_micro, p_ci_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro, p_ci_macro = precision_score(y_true, y_pred, average='macro')
    
    recall_micro, r_ci_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro, r_ci_macro = recall_score(y_true, y_pred, average='macro')
    
    f1_micro, f1_ci_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro, f1_ci_macro = f1_score(y_true, y_pred, average='macro')

    # Append micro and macro average rows
    new_row_micro = pd.DataFrame({
        'Class': 'micro',
        'Precision': [round(precision_micro,3)],
        'Recall': [round(recall_micro,3)],
        'F1-Score': [round(f1_micro,3)],
        'Precision CI': [round_tuple(p_ci_micro)],
        'Recall CI': [round_tuple(r_ci_micro)],
        'F1-Score CI': [round_tuple(f1_ci_micro)],
        'Support' : [len(y_true)]
    })
    df = pd.concat([df, new_row_micro], ignore_index=True)
    
    new_row_macro = pd.DataFrame({
        'Class': 'macro',
        'Precision': [round(precision_macro,3)],
        'Recall': [round(recall_macro,3)],
        'F1-Score': [round(f1_macro,3)],
        'Precision CI': [round_tuple(p_ci_macro)],
        'Recall CI': [round_tuple(r_ci_macro)],
        'F1-Score CI': [round_tuple(f1_ci_macro)],
        'Support' : [len(y_true)]

    })
    df = pd.concat([df, new_row_macro], ignore_index=True)

    return df