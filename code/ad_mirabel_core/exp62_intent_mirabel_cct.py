"""Auditable statistical fusion primitives for Exp62.

The module deliberately contains no learned fusion parameters.  It combines
the frozen Exp61 Intent score and canonical Mirabel score after independent
empirical-tail calibration, then applies an empirically calibrated threshold.
Session confirmation reuses the same query p-values with equal-weight CCT.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from src.exp61_intent_mirabel import (
    harmonic_member_nonmember,
    hide_and_backfill,
    higher_threshold,
    wilson_interval,
)


FPR_TARGETS = (0.005, 0.01, 0.02, 0.05)
POLICIES = ("P0_ORIGINAL_MIRABEL", "P1_INTENT_ONLY", "P2_Q1_CCT", "P3_OR_CONTROL")


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def empirical_pvalue(
    scores: Sequence[float] | float,
    reference: Sequence[float],
    *,
    tail: str = "upper",
) -> np.ndarray:
    """Finite-sample empirical p-values with the mandatory +1 correction.

    Upper-tail p-values use ``(1 + #{x_i >= s}) / (N + 1)``.  Ties are
    therefore conservative and tail direction is explicit rather than inferred
    from attack labels.
    """

    ref = np.sort(_finite_vector(reference, name="reference"))
    query = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(query).all():
        raise ValueError("scores must be finite")
    if tail == "upper":
        counts = len(ref) - np.searchsorted(ref, query, side="left")
    elif tail == "lower":
        counts = np.searchsorted(ref, query, side="right")
    else:
        raise ValueError("tail must be 'upper' or 'lower'")
    return (1.0 + counts.astype(np.float64)) / (len(ref) + 1.0)


def cct_pvalue(pvalues: Sequence[float] | np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Equal-weight Cauchy combination of already calibrated p-values."""

    values = np.asarray(pvalues, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("finite p-values required")
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in (0, 1]")
    # p=1 is a valid empirical p-value.  nextafter avoids an infinite tangent
    # while preserving the limiting CCT result to machine precision.
    upper = np.nextafter(1.0, 0.0)
    clipped = np.clip(values, np.finfo(np.float64).tiny, upper)
    statistic = np.mean(np.tan(np.pi * (0.5 - clipped)), axis=axis)
    combined = 0.5 - np.arctan(statistic) / np.pi
    return np.clip(combined, 0.0, 1.0)


def anomaly_from_pvalue(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("finite p-values in (0, 1] required")
    return -np.log10(values)


def q1_fusion(
    intent_scores: Sequence[float],
    mirabel_scores: Sequence[float],
    intent_reference: Sequence[float],
    mirabel_reference: Sequence[float],
) -> dict[str, np.ndarray]:
    intent = _finite_vector(intent_scores, name="intent_scores")
    mirabel = _finite_vector(mirabel_scores, name="mirabel_scores")
    if len(intent) != len(mirabel):
        raise ValueError("intent and Mirabel scores must align")
    p_intent = empirical_pvalue(intent, intent_reference, tail="upper")
    p_mirabel = empirical_pvalue(mirabel, mirabel_reference, tail="upper")
    p_cct = cct_pvalue(np.column_stack((p_intent, p_mirabel)), axis=1)
    return {
        "p_intent": p_intent,
        "p_mirabel": p_mirabel,
        "p_cct": p_cct,
        "score_intent": anomaly_from_pvalue(p_intent),
        "score_mirabel": anomaly_from_pvalue(p_mirabel),
        "score_cct": anomaly_from_pvalue(p_cct),
        "score_or": np.maximum(anomaly_from_pvalue(p_intent), anomaly_from_pvalue(p_mirabel)),
    }


def calibrated_threshold(calibration_scores: Sequence[float], target_fpr: float) -> float:
    if target_fpr not in FPR_TARGETS and not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")
    return higher_threshold(calibration_scores, float(target_fpr))


def alarm_at_threshold(scores: Sequence[float], threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or not math.isfinite(float(threshold)):
        raise ValueError("finite scores and threshold required")
    return values > float(threshold)


def session_prefix_cct(query_pvalues: Sequence[float]) -> np.ndarray:
    values = _finite_vector(query_pvalues, name="query_pvalues")
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("query p-values must lie in (0, 1]")
    output = np.empty(len(values), dtype=np.float64)
    for stop in range(1, len(values) + 1):
        output[stop - 1] = float(cct_pvalue(values[:stop], axis=0))
    return output


def session_prefix_minp(query_pvalues: Sequence[float]) -> np.ndarray:
    values = _finite_vector(query_pvalues, name="query_pvalues")
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("query p-values must lie in (0, 1]")
    return np.minimum.accumulate(values)


def anytime_session_score(query_pvalues: Sequence[float], *, aggregator: str = "CCT") -> np.ndarray:
    """Return the score at every prefix under one anytime-calibrated threshold.

    This deliberately does *not* latch an earlier alarm by applying a prefix
    maximum to the returned scores.  The calibration threshold is fitted to
    each benign session's maximum prefix score, which controls the anytime
    false-positive event.  Keeping the current-prefix score here also makes
    the full-session CCT score and decision invariant to permutation, as the
    protocol requires.  Detection time can still depend on ordering.
    """
    if aggregator == "CCT":
        pvalues = session_prefix_cct(query_pvalues)
    elif aggregator == "MINP":
        pvalues = session_prefix_minp(query_pvalues)
    else:
        raise ValueError("aggregator must be CCT or MINP")
    return anomaly_from_pvalue(pvalues)


def calibrate_anytime_threshold(
    benign_sessions: Iterable[Sequence[float]],
    target_fpr: float,
    *,
    aggregator: str = "CCT",
) -> float:
    maxima = []
    for session in benign_sessions:
        scores = anytime_session_score(session, aggregator=aggregator)
        maxima.append(float(np.max(scores)))
    return calibrated_threshold(maxima, target_fpr)


def permutation_final_audit(query_pvalues: Sequence[float], permutations: Sequence[Sequence[int]]) -> dict[str, float]:
    values = _finite_vector(query_pvalues, name="query_pvalues")
    final = []
    for order in permutations:
        indices = np.asarray(order, dtype=int)
        if sorted(indices.tolist()) != list(range(len(values))):
            raise ValueError("each permutation must contain every index exactly once")
        final.append(float(session_prefix_cct(values[indices])[-1]))
    array = np.asarray(final, dtype=np.float64)
    return {"std": float(array.std()), "range": float(array.max() - array.min())}


@dataclass(frozen=True)
class DetectionMetrics:
    member_tpr: float
    nonmember_tpr: float
    adr: float
    gap: float


def detection_metrics(member_labels: Sequence[int], alarms: Sequence[bool]) -> DetectionMetrics:
    labels = np.asarray(member_labels, dtype=int)
    flags = np.asarray(alarms, dtype=bool)
    if labels.shape != flags.shape or set(np.unique(labels)) != {0, 1}:
        raise ValueError("aligned binary member labels with both classes required")
    member = float(flags[labels == 1].mean())
    nonmember = float(flags[labels == 0].mean())
    return DetectionMetrics(member, nonmember, harmonic_member_nonmember(member, nonmember), abs(member - nonmember))


def fpr_with_wilson(alarms: Sequence[bool]) -> dict[str, float | int]:
    flags = np.asarray(alarms, dtype=bool)
    if flags.ndim != 1 or not len(flags):
        raise ValueError("non-empty alarm vector required")
    count = int(flags.sum())
    low, high = wilson_interval(count, len(flags))
    return {"false_positives": count, "normal_n": len(flags), "fpr": float(flags.mean()),
            "wilson_low": low, "wilson_upper": high}


def defend_ranked_documents(
    ranked_document_ids: Sequence[str],
    *,
    alarm: bool,
    candidate_document: str,
    top_k: int = 3,
) -> tuple[tuple[str, ...], str | None, bool]:
    hidden = str(candidate_document) if bool(alarm) else None
    after, backfilled = hide_and_backfill(
        ranked_document_ids,
        top_k=top_k,
        hidden_document=hidden,
    )
    return after, hidden, backfilled
