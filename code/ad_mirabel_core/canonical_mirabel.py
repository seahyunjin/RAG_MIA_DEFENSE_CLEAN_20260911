"""Canonical MIRABEL extreme-value margin.

The official statistic excludes the largest similarity from the background
moments while retaining the *full* corpus size in the extreme-value term.
This module is intentionally free of tuned epsilons and scaling constants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np


MIRABEL_FORMULA_VERSION = "canonical_official_v2"
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class CanonicalMirabelStats:
    top1: float
    background_mean: float
    background_std: float
    corpus_size: int
    confidence: float
    threshold: float
    margin: float


def _validate_common(*, top1: float, corpus_size: int, confidence: float) -> None:
    if int(corpus_size) < 3:
        raise ValueError("canonical MIRABEL requires at least three corpus scores")
    if not math.isfinite(float(top1)):
        raise ValueError("top1 must be finite")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")


def _from_background(
    *,
    top1: float,
    background_mean: float,
    background_std: float,
    corpus_size: int,
    confidence: float,
) -> CanonicalMirabelStats:
    _validate_common(top1=top1, corpus_size=corpus_size, confidence=confidence)
    if not math.isfinite(float(background_mean)):
        raise ValueError("background mean must be finite")
    if not math.isfinite(float(background_std)) or float(background_std) < 0.0:
        raise ValueError("background standard deviation must be finite and nonnegative")
    root = math.sqrt(2.0 * math.log(int(corpus_size)))
    gumbel_quantile = -math.log(-math.log(float(confidence)))
    threshold = (
        float(background_mean)
        + float(background_std) * root
        + gumbel_quantile * float(background_std) / root
    )
    return CanonicalMirabelStats(
        top1=float(top1),
        background_mean=float(background_mean),
        background_std=float(background_std),
        corpus_size=int(corpus_size),
        confidence=float(confidence),
        threshold=float(threshold),
        margin=float(top1) - float(threshold),
    )


def canonical_mirabel_from_full_scores(
    sorted_scores: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> CanonicalMirabelStats:
    """Compute the official statistic from all descending corpus scores."""

    scores = np.asarray(sorted_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 3:
        raise ValueError("sorted_scores must be a one-dimensional full-corpus vector")
    if not np.isfinite(scores).all():
        raise ValueError("sorted_scores must contain only finite values")
    if np.any(scores[:-1] < scores[1:]):
        raise ValueError("sorted_scores must be in nonincreasing order")
    background = scores[1:]
    return _from_background(
        top1=float(scores[0]),
        background_mean=float(background.mean()),
        background_std=float(background.std(ddof=0)),
        corpus_size=int(scores.size),
        confidence=confidence,
    )


def canonical_mirabel_from_moments(
    *,
    top1: float,
    sum_all: float,
    sumsq_all: float,
    corpus_size: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> CanonicalMirabelStats:
    """Compute the official statistic without transferring all GPU scores."""

    _validate_common(top1=top1, corpus_size=corpus_size, confidence=confidence)
    if not math.isfinite(float(sum_all)) or not math.isfinite(float(sumsq_all)):
        raise ValueError("full-corpus moments must be finite")
    background_count = int(corpus_size) - 1
    background_sum = float(sum_all) - float(top1)
    background_sumsq = float(sumsq_all) - float(top1) ** 2
    background_mean = background_sum / background_count
    background_var = background_sumsq / background_count - background_mean**2
    # Equivalent to the prescribed tensor clamp_min(0).  This only protects
    # against a tiny negative caused by floating-point cancellation.
    background_var = max(0.0, float(background_var))
    return _from_background(
        top1=top1,
        background_mean=background_mean,
        background_std=math.sqrt(background_var),
        corpus_size=corpus_size,
        confidence=confidence,
    )


def canonical_cache_fields(stats: CanonicalMirabelStats) -> dict[str, object]:
    """Return the versioned fields required in a canonical retrieval row."""

    return {
        "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
        "mirabel_confidence": stats.confidence,
        "mirabel_corpus_n": stats.corpus_size,
        "mirabel_background_mean": stats.background_mean,
        "mirabel_background_std": stats.background_std,
        "mirabel_threshold": stats.threshold,
        "mirabel_margin": stats.margin,
    }


def require_canonical_cache_row(
    row: Mapping[str, object],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Reject legacy or incompatible cache rows instead of silently reusing them."""

    version = row.get("mirabel_formula_version")
    if version != MIRABEL_FORMULA_VERSION:
        raise ValueError(
            f"canonical MIRABEL cache required: expected {MIRABEL_FORMULA_VERSION!r}, "
            f"found {version!r}"
        )
    required = (
        "mirabel_confidence",
        "mirabel_corpus_n",
        "mirabel_background_mean",
        "mirabel_background_std",
        "mirabel_threshold",
        "mirabel_margin",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"canonical MIRABEL cache row is missing: {missing}")
    if not math.isclose(float(row["mirabel_confidence"]), float(confidence), abs_tol=0.0):
        raise ValueError("canonical MIRABEL cache confidence mismatch")
    if int(row["mirabel_corpus_n"]) < 3:
        raise ValueError("canonical MIRABEL cache corpus size is invalid")

