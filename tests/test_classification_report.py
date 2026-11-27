from confidenceinterval import classification_report_with_ci, \
    precision_score, recall_score, f1_score

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]


def _rounded(ci):
    return tuple(round(x, 3) for x in ci)


def test_report_uses_confidence_level_for_every_ci():
    level = 0.8
    report = classification_report_with_ci(
        y_true, y_pred, confidence_level=level).set_index('Class')

    y_true_0 = [int(y == 0) for y in y_true]
    y_pred_0 = [int(y == 0) for y in y_pred]
    _, p_ci = precision_score(y_true_0, y_pred_0, confidence_level=level,
                              average='binary', method='wilson')
    _, r_ci = recall_score(y_true_0, y_pred_0, confidence_level=level,
                           average='binary', method='wilson')
    assert report.loc['Class 0', 'Precision CI'] == _rounded(p_ci)
    assert report.loc['Class 0', 'Recall CI'] == _rounded(r_ci)

    for average in ['micro', 'macro']:
        for name, fn in [('Precision', precision_score),
                         ('Recall', recall_score),
                         ('F1-Score', f1_score)]:
            _, ci = fn(y_true, y_pred, confidence_level=level, average=average)
            assert report.loc[average, f'{name} CI'] == _rounded(ci)
