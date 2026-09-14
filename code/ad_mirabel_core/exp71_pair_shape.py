"""Frozen relative Pair-shape primitives for Exp71."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from src.exp70_pair_detector import empirical_upper_p, strict_threshold, wilson_upper


PAIR_SHAPE_FEATURES = (
    "mean_top1_similarity",
    "mean_top1_top2_gap",
    "mean_standardized_retrieval_concentration",
    "max_top1_similarity",
    "same_document_concentration",
)


def sigmoid(values: Sequence[float] | float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def pair_shape_prefix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Apply every directly transferable frozen Early feature.

    Vectors remain in retriever rank order.  They are never re-sorted by Pair
    score, matching the observation R(q)=[g(q,d1),...,g(q,dk)].
    """
    if not rows:
        raise ValueError("at least one query row required")
    top1, gaps, concentration, documents = [], [], [], []
    for row in rows:
        scores = np.asarray(row["pair_scores"], dtype=np.float64)
        if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
            raise ValueError("finite ordered top-k Pair scores required")
        background = scores[1:]
        spread = max(float(background.std(ddof=0)), np.finfo(float).eps)
        top1.append(float(scores[0]))
        gaps.append(float(scores[0] - scores[1]))
        concentration.append(float((scores[0] - background.mean()) / spread))
        documents.append(str(row["top_document_id"]))
    top = np.asarray(top1)
    return np.asarray((top.mean(), np.mean(gaps), np.mean(concentration), top.max(),
                       max(Counter(documents).values()) / len(documents)), dtype=np.float64)


def retrieval_shape_prefix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Exact six-feature frozen Early mapping, including Mirabel margin."""
    if not rows:
        raise ValueError("at least one query row required")
    top1, gaps, concentration, margins, documents = [], [], [], [], []
    for row in rows:
        scores = np.asarray(row["retrieved_scores"], dtype=np.float64)
        background = scores[1:]
        spread = max(float(background.std(ddof=0)), np.finfo(float).eps)
        top1.append(float(scores[0])); gaps.append(float(scores[0] - scores[1]))
        concentration.append(float((scores[0] - background.mean()) / spread))
        margins.append(float(row["mirabel_margin"])); documents.append(str(row["top_document_id"]))
    top = np.asarray(top1)
    return np.asarray((top.mean(), np.mean(gaps), np.mean(concentration), np.mean(margins),
                       top.max(), max(Counter(documents).values()) / len(documents)), dtype=np.float64)


def fit_empirical_shape(feature_reference: np.ndarray,
                        score_reference: np.ndarray) -> dict[str, object]:
    features = np.asarray(feature_reference, dtype=np.float64)
    scoring = np.asarray(score_reference, dtype=np.float64)
    if features.ndim != 2 or scoring.ndim != 2 or features.shape[1] != scoring.shape[1]:
        raise ValueError("aligned-dimensional [N,F] benign splits required")
    refs = tuple(np.sort(features[:, index]) for index in range(features.shape[1]))
    p = np.column_stack([empirical_upper_p(scoring[:, index], refs[index]) for index in range(features.shape[1])])
    fisher = -2.0 * np.mean(np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    return {"feature_references": refs, "score_reference": np.sort(fisher), "features": features.shape[1]}


def shape_p(features: np.ndarray, calibration: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    refs = calibration["feature_references"]
    p = np.column_stack([empirical_upper_p(values[:, index], refs[index]) for index in range(values.shape[1])])
    fisher = -2.0 * np.mean(np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    return empirical_upper_p(fisher, calibration["score_reference"])


def dual_minp(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or np.any((a <= 0) | (a > 1)) or np.any((b <= 0) | (b > 1)):
        raise ValueError("aligned p-values in (0,1] required")
    return np.minimum(1.0, 2.0 * np.minimum(a, b))


def binary_summary(alarm: Sequence[bool]) -> dict[str, float | int]:
    values = np.asarray(alarm, dtype=bool); n = len(values); count = int(values.sum())
    return {"n": n, "alarms": count, "rate": count / n, "wilson_upper": wilson_upper(count, n)}


def running_minp(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if not len(p) or np.any((p <= 0) | (p > 1)):
        raise ValueError("p-values in (0,1] required")
    return np.minimum.accumulate(p)


__all__ = ["PAIR_SHAPE_FEATURES", "binary_summary", "dual_minp", "fit_empirical_shape",
           "pair_shape_prefix", "retrieval_shape_prefix", "running_minp", "shape_p",
           "sigmoid", "strict_threshold"]
