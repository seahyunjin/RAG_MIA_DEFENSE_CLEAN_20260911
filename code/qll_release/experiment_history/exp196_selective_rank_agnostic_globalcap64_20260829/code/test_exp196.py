#!/usr/bin/env python3
import pandas as pd

from run_exp196 import caps_for, metric_rows, quantile_threshold


def main():
    assert caps_for(["a", "b", "c", "d"], "a") == {"a": 64, "b": 662, "c": 661, "d": 661}
    assert caps_for(["a", "b", "c", "d"], "d") == {"d": 64, "a": 662, "b": 661, "c": 661}
    values = list(range(1000)); tau = quantile_threshold(values, .01)
    assert tau == 989 and sum(value > tau for value in values) == 10
    scores = pd.DataFrame({
        "condition": ["x"] * 5,
        "attack_family": ["IA"] * 5,
        "member": [0, 0, 1, 1, 1],
        "attack_score": ["0.1", "", "0.8", "0.9", ""],
    })
    metrics = metric_rows(scores).iloc[0]
    assert metrics.n == 3 and metrics.n_undefined == 2
    assert metrics.raw_auc == 1.0 and metrics.effective_auc == 1.0
    print("EXP196_UNIT_TESTS_PASS")


if __name__ == "__main__":
    main()
