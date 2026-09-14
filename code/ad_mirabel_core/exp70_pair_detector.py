"""Statistical and representation primitives for Exp70.

The module deliberately contains no dataset, family, membership, or domain
feature at inference.  Both candidates share the same pair interaction; the
only representational ablation is frozen MPNet versus shared MPNet LoRA.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def pair_interaction(query: np.ndarray, document: np.ndarray) -> np.ndarray:
    """Return [q, d, |q-d|, q*d] for aligned normalized embeddings."""
    q = np.asarray(query, dtype=np.float32)
    d = np.asarray(document, dtype=np.float32)
    if q.shape != d.shape or q.ndim != 2 or not np.isfinite(q).all() or not np.isfinite(d).all():
        raise ValueError("aligned finite [N,D] query/document embeddings required")
    return np.concatenate((q, d, np.abs(q - d), q * d), axis=1)


def empirical_upper_p(values: Sequence[float] | float,
                      reference: Sequence[float]) -> np.ndarray:
    observed = np.atleast_1d(np.asarray(values, dtype=np.float64))
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(ref) or not np.isfinite(ref).all() or not np.isfinite(observed).all():
        raise ValueError("finite non-empty observations/reference required")
    return (len(ref) - np.searchsorted(ref, observed, side="left") + 1.0) / (len(ref) + 1.0)


def empirical_lower_p(values: Sequence[float] | float,
                      reference: Sequence[float]) -> np.ndarray:
    observed = np.atleast_1d(np.asarray(values, dtype=np.float64))
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(ref) or not np.isfinite(ref).all() or not np.isfinite(observed).all():
        raise ValueError("finite non-empty observations/reference required")
    return (np.searchsorted(ref, observed, side="right") + 1.0) / (len(ref) + 1.0)


def bonferroni_minp(*pvalues: Sequence[float]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in pvalues]
    if not arrays or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("aligned p-values required")
    if any(np.any((value <= 0) | (value > 1)) for value in arrays):
        raise ValueError("p-values must lie in (0,1]")
    return np.minimum(1.0, len(arrays) * np.minimum.reduce(arrays))


def early_q1_features(scores: Sequence[float], margin: float) -> np.ndarray:
    """Exact frozen All6 Q1 mapping from heuristic_free_early.aggregate_prefix."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or not math.isfinite(margin):
        raise ValueError("finite top-k scores and Mirabel margin required")
    top1, background = float(values[0]), values[1:]
    spread = max(float(background.std(ddof=0)), np.finfo(float).eps)
    concentration = (top1 - float(background.mean())) / spread
    return np.asarray((top1, top1 - float(values[1]), concentration,
                       float(margin), top1, 1.0), dtype=np.float64)


def fit_early_q1(feature_reference: np.ndarray,
                 score_reference_rows: np.ndarray) -> dict[str, object]:
    feature = np.asarray(feature_reference, dtype=np.float64)
    score_rows = np.asarray(score_reference_rows, dtype=np.float64)
    if feature.ndim != 2 or feature.shape[1] != 6 or score_rows.ndim != 2 or score_rows.shape[1] != 6:
        raise ValueError("two finite [N,6] benign splits required")
    refs = tuple(np.sort(feature[:, index]) for index in range(6))
    p = np.column_stack([empirical_upper_p(score_rows[:, index], refs[index]) for index in range(6)])
    # The historical implementation uses weights 1/6.  Multiplying every
    # score by six is monotone; we preserve the exact weighted definition.
    fisher = -2.0 * np.mean(np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    return {"feature_references": refs, "fisher_reference": np.sort(fisher)}


def early_q1_p(features: np.ndarray, calibration: dict[str, object]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    refs = calibration["feature_references"]
    p = np.column_stack([empirical_upper_p(values[:, index], refs[index]) for index in range(6)])
    fisher = -2.0 * np.mean(np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    return empirical_upper_p(fisher, calibration["fisher_reference"])


def strict_threshold(scores: Sequence[float], target_fpr: float = .02) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or not 0 < target_fpr < 1:
        raise ValueError("finite scores and target FPR in (0,1) required")
    candidates = np.unique(values)
    feasible = [(int(np.count_nonzero(values > threshold)), float(threshold))
                for threshold in candidates if np.mean(values > threshold) <= target_fpr + 1e-15]
    alarms, threshold = max(feasible, key=lambda row: (row[0], -row[1]))
    return {"threshold": threshold, "alarms": alarms, "n": len(values),
            "observed_fpr": alarms / len(values), "comparator": ">"}


def wilson_upper(alarms: int, n: int, z: float = 1.959963984540054) -> float:
    if n <= 0 or not 0 <= alarms <= n:
        raise ValueError("valid binomial counts required")
    p = alarms / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return min(1.0, center + radius)


__all__ = ["bonferroni_minp", "early_q1_features", "early_q1_p",
           "empirical_lower_p", "empirical_upper_p", "fit_early_q1",
           "pair_interaction", "strict_threshold", "wilson_upper"]
