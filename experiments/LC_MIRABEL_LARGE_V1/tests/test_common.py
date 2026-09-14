import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from common import empirical_upper_tail, matched_threshold, tpr_at_fpr, wilson


def test_empirical_tail_includes_plus_one():
    assert empirical_upper_tail([1.0, 2.0, 3.0], 4.0) == 0.25
    assert empirical_upper_tail([1.0, 2.0, 3.0], 2.0) == 0.75


def test_matched_threshold_does_not_exceed_budget():
    threshold, alarms = matched_threshold(list(range(100)), .03)
    assert alarms <= 3
    assert sum(value > threshold for value in range(100)) == alarms


def test_tpr_interpolation_bounds():
    value = tpr_at_fpr([.9, .8, .1], [.7, .6, .2], .5)
    assert 0 <= value <= 1


def test_wilson_contains_point_estimate():
    low, high = wilson(30, 1000)
    assert low < .03 < high
