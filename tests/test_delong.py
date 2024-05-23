from confidenceinterval.delong import compute_ground_truth_statistics
import numpy as np
import pytest


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
