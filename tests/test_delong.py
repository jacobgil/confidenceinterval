from confidenceinterval.delong import compute_ground_truth_statistics, delong_roc_variance
import numpy as np
import pytest
import sklearn.metrics


@pytest.mark.parametrize(
    'ground_truth',
    [
        (np.array([0,0,1,1,1])), # Values are integers
        (np.array([False, False, True, True, True])) # values are bools
        ]
)
def test_compute_ground_truth_statistics(ground_truth):
    sample_weight = np.array([1,1,1,1,1])
    
    expected_order = np.array([2,3,4,0,1])
    expected_label_1_count = 3
    expected_ordered_sample_weight = np.array([1,1,1,1,1])
    
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(ground_truth=ground_truth, sample_weight=sample_weight)
    
    assert np.array_equal(expected_order, order)
    assert expected_label_1_count == label_1_count
    assert np.array_equal(expected_ordered_sample_weight, ordered_sample_weight)


def test_delong_auc_does_not_exceed_one_for_perfect_separation():
    # Case from issue #15: every positive scores above every negative.
    ground_truth = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    predictions = np.array([0.00721543, 0.53823202, 0.8211329, 0.24135367, 0.61904833, 0.51323922, 0.50633062,
                            0.32820338, 0.0909322, 0.75047184, 0.68729921, 0.9778512, 0.71006862, 0.4194739,
                            0.31955396, 0.84841144, 0.75825972, 0.9168877, 0.86957666, 0.29215992, 0.77219873,
                            0.87827124, 0.57069033, 0.5206118, 0.73469778, 0.54911986, 0.63431484, 0.69279543,
                            0.96744707, 0.98569492, 0.82269733, 0.9818143, 0.57780473, 0.86393161, 0.69898409,
                            0.95322689, 0.66892306, 0.82946166, 0.78274066, 0.84924395, 0.87428222, 0.37256549,
                            0.33853178, 0.79422342, 0.14013082, 0.98968488, 0.2856893, 0.95359197, 0.13624191,
                            0.72447603])
    auc, _ = delong_roc_variance(ground_truth, predictions)
    assert auc == 1.0


@pytest.mark.parametrize("n", [1000, 200_000])
def test_delong_auc_matches_sklearn_in_double_precision(n):
    rng = np.random.default_rng(0)
    ground_truth = rng.integers(0, 2, n)
    predictions = np.round(rng.standard_normal(n) + ground_truth, 2)  # rounding creates ties
    auc, variance = delong_roc_variance(ground_truth, predictions)
    assert auc.dtype == np.float64
    assert auc == pytest.approx(sklearn.metrics.roc_auc_score(ground_truth, predictions), abs=1e-12)
    assert variance > 0
