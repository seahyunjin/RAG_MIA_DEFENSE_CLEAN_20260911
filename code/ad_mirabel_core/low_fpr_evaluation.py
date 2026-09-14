"""Session-level statistical utilities for Experiment 42."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta
from sklearn.metrics import auc, roc_curve

from .calibration_resolution import require_empirical_resolution


# Detector scores are produced by several NumPy/scikit-learn paths.  Values that
# are mathematically tied can therefore differ by one or two binary ULPs after a
# CSV round trip (for example, 2.699404081815337 vs 2.6994040818153375).  At a
# discrete empirical-p boundary that tiny representation difference can change a
# decision.  Quantising both sides before rank comparison gives ties one stable,
# conservative (">=") treatment without affecting meaningful score precision.
EMPIRICAL_SCORE_DECIMALS = 12


def empirical_upper_pvalues(
    calibration_scores: Sequence[float], evaluation_scores: Sequence[float]
) -> np.ndarray:
    reference = np.asarray(calibration_scores, dtype=float)
    values = np.asarray(evaluation_scores, dtype=float)
    if reference.ndim != 1 or len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("calibration_scores must be a nonempty finite vector")
    if not np.isfinite(values).all():
        raise ValueError("evaluation_scores must be finite")
    reference = np.sort(np.round(reference, decimals=EMPIRICAL_SCORE_DECIMALS))
    values = np.round(values, decimals=EMPIRICAL_SCORE_DECIMALS)
    ge = len(reference) - np.searchsorted(reference, values, side="left")
    return (ge + 1.0) / (len(reference) + 1.0)


def low_fpr_decisions(
    calibration_scores: Sequence[float],
    evaluation_scores: Sequence[float],
    target_fpr: float,
) -> tuple[np.ndarray, np.ndarray]:
    require_empirical_resolution(
        num_independent_sessions=len(calibration_scores), target_fpr=target_fpr
    )
    pvalues = empirical_upper_pvalues(calibration_scores, evaluation_scores)
    return pvalues <= float(target_fpr), pvalues


def assert_nested_decisions(decisions: Mapping[float, Sequence[bool]]) -> None:
    targets = sorted(float(value) for value in decisions)
    for lower, upper in zip(targets, targets[1:]):
        left = np.asarray(decisions[lower], dtype=bool)
        right = np.asarray(decisions[upper], dtype=bool)
        if left.shape != right.shape or np.any(left & ~right):
            raise AssertionError(f"blocked@{lower:g} is not a subset of blocked@{upper:g}")


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    k, n = int(successes), int(trials)
    if n <= 0 or k < 0 or k > n or not 0 < confidence < 1:
        raise ValueError("invalid binomial interval arguments")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def bootstrap_rate_interval(
    values: Sequence[bool | int | float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(int(resamples), dtype=float)
    chunk = 500
    for start in range(0, int(resamples), chunk):
        count = min(chunk, int(resamples) - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_bootstrap_difference(
    candidate: Sequence[bool | int | float],
    baseline: Sequence[bool | int | float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    left, right = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("paired vectors must be nonempty and aligned")
    return bootstrap_rate_interval(left - right, resamples=resamples, seed=seed)


def maximum_prefix(scores: Sequence[float], horizon: int | None = None) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("prefix scores must be a nonempty finite vector")
    if horizon is not None:
        if int(horizon) <= 0 or int(horizon) > len(values):
            raise ValueError("invalid horizon")
        values = values[: int(horizon)]
    return float(values.max())


def first_detection_turn(pvalues: Sequence[float], target_fpr: float) -> float:
    values = np.asarray(pvalues, dtype=float)
    hit = np.flatnonzero(values <= float(target_fpr))
    return float(hit[0] + 1) if len(hit) else float("nan")


def stable_bucket(value: object, buckets: int, *, salt: str = "exp42") -> int:
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()
    return int(digest[:16], 16) % int(buckets)


def assert_disjoint_groups(roles: Mapping[str, Iterable[object]]) -> None:
    seen: dict[str, str] = {}
    for role, values in roles.items():
        for value in values:
            key = str(value)
            if key in seen and seen[key] != role:
                raise AssertionError(f"group {key} overlaps roles {seen[key]} and {role}")
            seen[key] = role


def empirical_partial_auc(
    labels: Sequence[int], scores: Sequence[float], maximum_fpr: float
) -> float:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) != 2 or not 0 < maximum_fpr <= 1:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    stop = int(np.searchsorted(fpr, maximum_fpr, side="right"))
    x, z = fpr[:stop], tpr[:stop]
    if len(x) == 0 or x[-1] < maximum_fpr:
        index = min(stop, len(fpr) - 1)
        if index == 0:
            y_at = float(tpr[0])
        else:
            y_at = float(np.interp(maximum_fpr, fpr[index - 1 : index + 1], tpr[index - 1 : index + 1]))
        x = np.r_[x, maximum_fpr]
        z = np.r_[z, y_at]
    return float(auc(x, z) / maximum_fpr)
