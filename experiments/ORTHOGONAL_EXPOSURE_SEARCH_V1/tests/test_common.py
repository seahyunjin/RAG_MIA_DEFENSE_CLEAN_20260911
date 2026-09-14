#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from common import empirical_upper_tail, matched_binary_threshold, normalize_tokens, tpr_at_fpr


def test_unicode_normalization_and_numbers():
    assert normalize_tokens("Caf\u00e9, A_B 27.8") == {"caf\u00e9", "a", "b", "27", "8"}


def test_upper_tail_ties():
    assert empirical_upper_tail([1.0, 2.0, 2.0], 2.0) == 0.75


def test_binary_budget():
    threshold, alarms = matched_binary_threshold(list(map(float, range(100))), .03)
    assert alarms <= 3 and sum(x > threshold for x in range(100)) == alarms


def test_roc_interpolation_bounds():
    value = tpr_at_fpr([2, 3, 4], [0, 1, 5], .03)
    assert 0 <= value <= 1

